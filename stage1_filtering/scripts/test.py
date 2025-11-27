#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Dual-Branch Build + Unlabeled Ensemble Inference (EVT)
 - 입력 CSV → dedup → 임베딩/윈도 저장 → 앙상블 추론(EVT 임계값) → 라벨 역매핑 → 원본 CSV에 label 추가 저장
 - 0=정상, 1=이상
"""

import os, sys, json, argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import genpareto  # EVT

# ── 외부 모델 의존 (사용자 프로젝트 경로 구조 가정) ───────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))   # ../models/model.py 접근
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))  # LORIN 패키지 접근

from models.model import load_teacher_ckpt, RevStudent  # noqa: E402

from sentence_transformers import SentenceTransformer

# ── 중앙 집중식 설정 로드 (LORIN 패키지에서) ───────────────────────────────────
try:
    from LORIN.config.settings import get_stage1_settings
    _stage1_cfg = get_stage1_settings()
    WORKERS = _stage1_cfg.workers
except ImportError:
    # LORIN 패키지가 없는 경우 기본값 사용
    WORKERS = int(os.getenv("STAGE1_WORKERS", "4"))

# ── 하이퍼파라미터(추론에 필요한 범위) ─────────────────────────────────────────
MASK_RAND, LOW_FREQ_RATIO = 0.3, 0.01
STUD_HID, STUD_HEADS, STUD_LAYERS, STUD_DROPOUT = 512, 8, 4, 0.2
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
#  유틸 & 데이터셋
# =========================
def sbert_encode(texts, model, bs):
    """문장 리스트 → SBERT float16 임베딩"""
    arr = model.encode(
        texts,
        batch_size=bs,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    return arr.astype(np.float16)

def dedup_consecutive_with_groups(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    연속된 동일 EventId 구간을 하나로 dedup.
    반환:
      (dedup_df, orig_to_dedup_idx, groups_df)
        * dedup_df: 연속 구간 첫 행만 남긴 DataFrame (dedup_idx 포함)
        * orig_to_dedup_idx: 원본 각 행 → dedup 인덱스(0..G-1) 매핑 배열
        * groups_df: 각 group_id -> dedup_idx, first_row 저장 테이블
    """
    df = df.copy()
    df["_row"] = np.arange(len(df))  # 원본 순서 보존용
    groups = (df["EventId"] != df["EventId"].shift()).cumsum()  # 1..G
    df["group_id"] = groups

    # 그룹 등장 순서 보존한 고유 group_id → 0..G-1 dedup_idx 부여
    uniq_groups = df["group_id"].drop_duplicates().tolist()
    gid2didx = {gid: i for i, gid in enumerate(uniq_groups)}
    df["dedup_idx"] = df["group_id"].map(gid2didx)

    # dedup 결과 (각 그룹의 첫 행)
    dedup_df = (
        df.sort_values("_row")
          .groupby("group_id", sort=False)
          .agg({
              "_row": "first",
              "EventId": "first",
              "dedup_idx": "first"
          })
          .sort_values("_row")
          .reset_index(drop=False)
    )
    dedup_df = dedup_df[["dedup_idx", "group_id", "_row", "EventId"]].rename(
        columns={"_row": "first_row_idx"}
    )

    # 원본 각 행 → dedup_idx 매핑
    orig_to_dedup_idx = df.sort_values("_row")["dedup_idx"].to_numpy()

    # 그룹 메타
    groups_df = dedup_df.copy()

    return dedup_df, orig_to_dedup_idx, groups_df

class TripletNPY(Dataset):
    """저장된 NPY (ctx, ae) 로더: (emb, ids)만 반환"""
    def __init__(self, ctx_path: Path, ids_path: Path):
        self.ctx = np.load(ctx_path, mmap_mode="r")
        self.ids = np.load(ids_path, mmap_mode="r")
    def __len__(self): return self.ctx.shape[0]
    def __getitem__(self, i):
        emb = torch.from_numpy(self.ctx[i].astype(np.float32))
        ids = torch.from_numpy(self.ids[i].astype(np.int64))
        return emb, ids

def mask_batch_per_seq(x: torch.Tensor,
                       ids: torch.Tensor,
                       rand_ratio: float,
                       lowf_ratio: float):
    """
    x: (B, S, E), ids: (B, S)
    배치 저빈도 + 시퀀스별 무작위 마스크 혼합
    """
    B, S, _ = x.shape
    device = x.device

    flat = ids.view(-1)
    uniq, cnt = torch.unique(flat, return_counts=True)
    cnt_sorted, order = torch.sort(cnt)
    cum_cnt = torch.cumsum(cnt_sorted, 0)
    low_ids = uniq[order[cum_cnt < flat.numel() * lowf_ratio]]
    low_mask = torch.isin(ids, low_ids)  # (B,S)

    k = max(1, int((rand_ratio * S) + 0.9999))  # ceil
    mask = low_mask.clone()
    for b in range(B):
        need = k - int(mask[b].sum())
        if need > 0:
            cand = (~mask[b]).nonzero(as_tuple=False).squeeze(1)
            if cand.numel():
                pick = cand[torch.randperm(cand.numel(), device=device)[:need]]
                mask[b, pick] = True

    x_corr = x.clone()
    x_corr[mask.unsqueeze(-1).expand_as(x_corr)] = 0.0
    return x_corr, mask

def kd_seq(pred, tgt, msk, alpha=0.30, eps=1e-6):
    """
    (alpha*MSE + (1-alpha)*cosine) 마스크 평균 → (B,)
    """
    mse = F.mse_loss(pred, tgt, reduction="none").mean(-1)
    cos = 1 - F.cosine_similarity(pred, tgt, dim=-1)
    num = (alpha * mse + (1 - alpha) * cos) * msk
    den = msk.sum(1) + eps
    return torch.nan_to_num(num.sum(1) / den, nan=0.0, posinf=1e6, neginf=1e6)

def evt_threshold(scores: np.ndarray, tail_q: float = 0.95, p_extreme: float = 0.999) -> float:
    """
    EVT(GPD) 기반 임계값:
      1) 상위 tail_q 분위수 q 선택
      2) tail = scores[scores>q] - q
      3) tail에 GPD 적합 → p_extreme 분위수로 역변환
    tail 샘플이 너무 적으면 보수적으로 99퍼센타일 사용
    """
    q = np.quantile(scores, tail_q)
    tail = scores[scores > q] - q
    if len(tail) < 10:
        return np.percentile(scores, 99)
    c, _, s = genpareto.fit(tail, floc=0)
    return q + genpareto.ppf(p_extreme, c, loc=0, scale=s)


# =========================
#  메인 파이프라인
# =========================
def build_dual_branch_and_save(args) -> Tuple[int, np.ndarray, List[int]]:
    """
    - 템플릿 CSV 로드 → SBERT 임베딩
    - 구조화 로그 CSV 로드 → dedup (매핑 보관)
    - dedup 시퀀스로 윈도 생성 (ctx/ae/mask), 저장
    - 반환: N_dedup, orig_to_dedup_idx, window_starts
    """
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # 1) SBERT 로드 & 템플릿 임베딩
    print(f"[INFO] Loading SBERT: {args.embed_model}")
    sbert = SentenceTransformer(args.embed_model)
    tmpl_df = pd.read_csv(args.template_csv, dtype={"EventId": str})
    tmpl_ids = tmpl_df["EventId"].tolist()
    tmpl_txts = tmpl_df["EventTemplate"].tolist()
    print(f"[INFO] unique templates = {len(tmpl_ids)} | encoding templates…")
    tmpl_emb = sbert_encode(tmpl_txts, sbert, args.sbert_batch)
    id2idx = {eid: i for i, eid in enumerate(tmpl_ids)}
    np.save(out / "template_embeddings.npy", tmpl_emb)

    # 2) 원본 로그 로드 → dedup (그룹 보존)
    log_df_orig = pd.read_csv(args.struct_csv, dtype={"EventId": str})
    log_df_orig = log_df_orig.reset_index(drop=True)
    dedup_df, orig_to_dedup_idx, groups_df = dedup_consecutive_with_groups(log_df_orig)
    print(f"[INFO] total rows (orig) = {len(log_df_orig)}")
    print(f"[INFO] total rows (dedup) = {len(dedup_df)}")

    # dedup 시퀀스의 템플릿 인덱스
    event_idx = np.array([id2idx[e] for e in dedup_df["EventId"]], dtype=np.int32)

    # 3) 슬라이딩 윈도 생성
    WSIZE, STRIDE = args.window_size, args.stride
    ctx_win, ae_win, mask_win, win_starts = [], [], [], []
    for st in tqdm(range(0, len(event_idx) - WSIZE + 1, STRIDE), desc="windows"):
        ed = st + WSIZE
        win_ids = event_idx[st:ed]
        ctx_win.append(tmpl_emb[win_ids])
        ae_win.append(win_ids)
        mask_win.append(np.ones(WSIZE, dtype=np.int8))
        win_starts.append(st)

    ctx_win  = np.asarray(ctx_win,  dtype=np.float16) # (W,S,E)
    ae_win   = np.asarray(ae_win,   dtype=np.int32)   # (W,S)
    mask_win = np.asarray(mask_win, dtype=np.int8)    # (W,S)

    # 4) 저장
    np.save(out / "X_test_ctx.npy",  ctx_win)
    np.save(out / "X_test_ae.npy",   ae_win)
    np.save(out / "X_test_mask.npy", mask_win)
    np.save(out / "orig_to_dedup_idx.npy", orig_to_dedup_idx)
    groups_df.to_csv(out / "dedup_groups.csv", index=False)
    with open(out / "build_meta.json", "w") as f:
        json.dump({
            "N_orig": int(len(log_df_orig)),
            "N_dedup": int(len(dedup_df)),
            "window_size": int(WSIZE),
            "stride": int(STRIDE),
            "num_windows": int(len(ctx_win))
        }, f, indent=2)
    print(f"[SAVE] ctx={ctx_win.shape} ae={ae_win.shape} mask={mask_win.shape}")

    # 원본 CSV(라벨 추가 전) 사본 저장
    log_df_orig.to_csv(out / "Android.log_structured_clean.original_copy.csv", index=False)

    return len(dedup_df), orig_to_dedup_idx, win_starts


@torch.no_grad()
def infer_ensemble_unlabeled_and_label_rows(args,
                                            N_dedup: int,
                                            orig_to_dedup_idx: np.ndarray,
                                            win_starts: List[int]) -> None:
    """
    - 저장된 NPY 로더로 앙상블 추론 (윈도당 점수)
    - (EVT 대신) 고정 임계값 0.3으로 윈도 라벨링 → dedup 포지션 점수/라벨로 역매핑(최대값)
    - 원본 CSV에 label(0/1) 추가 저장
    """
    out = args.out
    # 1) 데이터 로더
    ds = TripletNPY(out / "X_test_ctx.npy", out / "X_test_ae.npy")
    dl = DataLoader(ds, args.batch, shuffle=False, num_workers=WORKERS, pin_memory=True)

    # 2) 모델 로드 (teacher + 앙상블 student들)
    teacher = load_teacher_ckpt(args.teacher_ckpt, device=DEV)
    models = []

    # ---- A) 개별 ckpt 파일 직접 지정 ----
    if hasattr(args, "student_ckpt_files") and args.student_ckpt_files.strip():
        ckpt_paths = [Path(p.strip()) for p in args.student_ckpt_files.split(",") if p.strip()]
        print("[CKPT] using explicit files:", [str(p) for p in ckpt_paths])
        for p in ckpt_paths:
            if not p.exists():
                raise FileNotFoundError(f"student_ckpt_files 항목이 가리키는 파일이 없습니다: {p}")
            mdl = RevStudent(hidden_dim=STUD_HID, num_heads=STUD_HEADS,
                             num_layers=STUD_LAYERS, dropout=STUD_DROPOUT).to(DEV)
            mdl.load_state_dict(torch.load(p, map_location=DEV))
            mdl.eval()
            models.append(mdl)

    # ---- B) seed 디렉터리 구조 사용 ----
    else:
        ckpt_tokens = [c.strip().lower() for c in args.ckpts.split(",")]
        if "auto" in ckpt_tokens and len(ckpt_tokens) > 1:
            raise ValueError("--ckpts auto 는 단독으로만 사용하세요.")

        # student_root 있으면 그걸 쓰고, 없으면 out 폴더 사용
        seed_base = getattr(args, "student_root", None) or args.out
        print(f"[CKPT] seed_base = {seed_base}")

        for sid in [s.strip() for s in args.seeds.split(",") if s.strip()]:
            seed_dir = seed_base / f"seed{sid}"
            if not seed_dir.exists():
                raise FileNotFoundError(f"seed 디렉터리가 없습니다: {seed_dir}")

            if ckpt_tokens == ["auto"]:
                metrics_path = seed_dir / "metrics.json"
                if not metrics_path.exists():
                    raise FileNotFoundError(f"auto 모드에 필요한 metrics.json이 없습니다: {metrics_path}")
                with open(metrics_path) as fp:
                    metr = json.load(fp)
                load_list = [max(("best", "swa", "last"), key=lambda t: metr[t]["f1"])]
            else:
                load_list = ckpt_tokens

            for tag in load_list:
                ckpt_path = seed_dir / f"{tag}.pt"
                if not ckpt_path.exists():
                    raise FileNotFoundError(f"체크포인트가 없습니다: {ckpt_path}")
                print(f"[CKPT] load: {ckpt_path}")
                mdl = RevStudent(hidden_dim=STUD_HID, num_heads=STUD_HEADS,
                                 num_layers=STUD_LAYERS, dropout=STUD_DROPOUT).to(DEV)
                mdl.load_state_dict(torch.load(ckpt_path, map_location=DEV))
                mdl.eval()
                models.append(mdl)

    if not models:
        raise RuntimeError("학생 모델을 하나도 로드하지 못했습니다.")

    print(f"[INFO] #Models loaded = {len(models)}")

    # 3) 윈도 점수 계산
    window_scores = []
    for x, ids in dl:
        x = x.to(DEV); ids = ids.to(DEV)
        x_cor, msk = mask_batch_per_seq(x, ids, rand_ratio=MASK_RAND, lowf_ratio=LOW_FREQ_RATIO)
        s_batch = torch.stack([kd_seq(m(teacher(x_cor)), x, msk) for m in models]).median(0)[0]
        window_scores.append(s_batch.cpu())
    window_scores = torch.cat(window_scores).numpy()  # (W,)

    np.save(out / "window_scores.npy", window_scores)

    # 4) 임계값 & 라벨 (고정값 사용)
    thr = 0.3
    window_preds = (window_scores >= thr).astype(int)
    np.save(out / "window_preds.npy", window_preds)
    with open(out / "threshold.json", "w") as f:
        json.dump({"method": "fixed", "threshold": float(thr)}, f, indent=2)

    # (비율 출력 대신 단순 개수만 출력)
    num_anom = int(window_preds.sum())
    print(f"[THR-FIXED] using fixed threshold={thr:.6f}; flagged_windows={num_anom}")


    # 5) dedup 포지션별 점수/라벨 역매핑 (윈도 내 최대 점수)
    WSIZE, STRIDE = args.window_size, args.stride
    W = len(window_scores)
    N_check = (W - 1) * STRIDE + WSIZE if W > 0 else 0
    if N_check != N_dedup:
        print(f"[WARN] dedup length mismatch: computed {N_check} vs meta {N_dedup} (계산은 진행)")

    pos_scores = np.full((N_dedup,), -np.inf, dtype=np.float32)
    for i, st in enumerate(win_starts):
        ed = st + WSIZE
        score = window_scores[i]
        pos_scores[st:ed] = np.maximum(pos_scores[st:ed], score)

    pos_scores[np.isneginf(pos_scores)] = -1e9
    pos_labels = (pos_scores >= thr).astype(int)
    np.save(out / "dedup_pos_scores.npy", pos_scores)
    np.save(out / "dedup_pos_labels.npy", pos_labels)

    # 6) 원본 행 라벨 브로드캐스트
    row_labels = pos_labels[orig_to_dedup_idx]
    assert row_labels.shape[0] == len(orig_to_dedup_idx)

    df_orig = pd.read_csv(args.struct_csv, dtype={"EventId": str})
    df_orig = df_orig.reset_index(drop=True)
    df_orig["label"] = row_labels.astype(np.int8)
    out_csv = out / "Android.log_structured_clean.labeled.csv"
    df_orig.to_csv(out_csv, index=False)
    print(f"[SAVE] labeled csv: {out_csv} (rows={len(df_orig)})")

    # 7) 시각화
    sns.set_style("whitegrid")

    plt.figure(figsize=(9, 5))
    sns.kdeplot(window_scores, fill=True, alpha=0.4)
    plt.axvline(thr, ls="--", lw=1.5, c="k",
                label=f"Fixed threshold = {thr:.6f}")
    plt.title("Window Anomaly-Score Distribution (Fixed)")
    plt.xlabel("Window Score"); plt.legend()
    plt.tight_layout()
    dist_path = out / f"win_dist_fixed_{str(thr).replace('.','_')}.png"
    plt.savefig(dist_path, dpi=300); plt.close()
    print(f"[PLOT] {dist_path}")

    plt.figure(figsize=(11, 4))
    cmap = np.where(pos_labels == 1, "r", "b")
    plt.scatter(range(len(pos_scores)), pos_scores, c=cmap, s=6, alpha=0.55)
    plt.axhline(thr, ls="--", lw=1, c="k")
    plt.title("Dedup Position Scores over Index (Fixed)")
    plt.xlabel("Dedup index"); plt.ylabel("Max Window Score")
    plt.tight_layout()
    scat_path = out / f"dedup_pos_scores_fixed_{str(thr).replace('.','_')}.png"
    plt.savefig(scat_path, dpi=300); plt.close()
    print(f"[PLOT] {scat_path}")



# =========================
#  CLI
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Build dual-branch inputs + Unlabeled Ensemble Inference (EVT) + Row labeling (0=normal,1=anomaly)"
    )
    # 입력 CSV
    parser.add_argument(
        "--struct_csv",
        type=Path,
        default=Path("/home/irv4/HybridTransformer/GUI/hhhhaha/data1111/result/안드로이드 핵심 시스템 서버가 크래시 발생.log_structured.csv"),
    )
    parser.add_argument(
        "--template_csv",
        type=Path,
        default=Path("/home/irv4/HybridTransformer/GUI/hhhhaha/data1111/result/안드로이드 핵심 시스템 서버가 크래시 발생.log_templates.csv"),
    )

    # 출력 폴더 (임베딩/윈도, 결과 저장 경로)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/irv4/HybridTransformer/output_test"),
    )

    # SBERT & 윈도
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--sbert_batch", type=int, default=64)
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--stride", type=int, default=50)

    # 추론 관련
    parser.add_argument(
        "--teacher_ckpt",
        type=Path,
        default=Path("/home/irv4/HybridTransformer/GUI/hhhhaha/teacher_pretrained_aosp.pt"),
    )
    parser.add_argument("--batch", type=int, default=192)

    # 학생 ckpt 위치/지정 방식
    parser.add_argument(
        "--student_root",
        type=Path,
        default=Path("/home/irv4/HybridTransformer/GUI/hhhhaha/revkd_out_v5_aosp"),
        help="훈련된 학생 ckpt들이 들어있는 루트( seed{sid}/best.pt ... )",
    )
    parser.add_argument("--student_ckpt_files", type=str, default="",
                        help="쉼표로 구분된 .pt 경로들(지정 시 --seeds/--ckpts 무시)")

    parser.add_argument("--seeds", type=str, default="42,77,99")
    parser.add_argument("--ckpts", type=str, default="best")

    args = parser.parse_args()

    # 1) 빌드 & 저장
    N_dedup, orig_to_dedup_idx, win_starts = build_dual_branch_and_save(args)

    # 2) 추론 & 라벨링 & 저장 (EVT)
    infer_ensemble_unlabeled_and_label_rows(args, N_dedup, orig_to_dedup_idx, win_starts)


if __name__ == "__main__":
    main()
