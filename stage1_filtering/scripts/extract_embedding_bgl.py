#!/usr/bin/env python3
# ============================================================================  
#  build_dual_branch_inputs.py   ―  Dual-Branch 입력 생성 (dedup + Timer 포함)
# ============================================================================  
#  • EO 브랜치 : SBERT(all-mpnet-base-v2) 임베딩 시퀀스
#  • AE 브랜치 : 템플릿-ID 시퀀스(LogSD 스타일)
#  • Attention Mask : Longformer용 토큰 마스크 (패딩 없으면 모두 1)
#  • Timer : 단계별 소요 시간 출력
#  • dedup  : 연속 EventId 중복 제거 (구간에 1이 있으면 라벨=1)
# ============================================================================

import os, json, time, contextlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ── Timer 컨텍스트 ──────────────────────────────────────────
@contextlib.contextmanager
def Timer(msg: str):
    t0 = time.perf_counter()
    print(f"[TIMER] {msg} …", flush=True)
    yield
    print(f"[TIMER] {msg} done  ➜  {time.perf_counter() - t0:,.2f} s", flush=True)
# ───────────────────────────────────────────────────────────

# ── (1) 고정 파라미터 ──────────────────────────────────────
EMBED_MODEL   = "sentence-transformers/all-mpnet-base-v2"

STRUCT_CSV    = "/home/bigdatanai/hhhhaha/data/parse_result/BGL.log_structured.csv"
TEMPLATE_CSV  = "/home/bigdatanai/hhhhaha/data/parse_result/BGL.log_templates.csv"

OUT_DIR       = "/home/bigdatanai/hhhhaha/data/embeddings_splits/"

WINDOW_SIZE   = 50      # 한 윈도 길이
STRIDE        = 40      # 슬라이딩 간격
BATCH_SIZE    = 64      # SBERT encode 배치
TRAIN_RATIO   = 0.85    # 정상 윈도 중 train 비율
# ───────────────────────────────────────────────────────────

def sbert_encode(texts, model, bs):
    """문장 리스트 → SBERT float16 임베딩"""
    return model.encode(
        texts, batch_size=bs, convert_to_numpy=True,
        show_progress_bar=True, device="cuda"
    ).astype(np.float16)

def dedup_consecutive(df: pd.DataFrame) -> pd.DataFrame:
    """
    EventId 가 연속으로 반복될 때 첫 행만 남기고,
    그 구간에 label 이 1 이라도 있으면 label=1 로 합산.
    """
    df = df.copy()
    df["_row"] = np.arange(len(df))
    groups = (df["EventId"] != df["EventId"].shift()).cumsum()

    return (df.assign(group_id=groups)
              .groupby("group_id", sort=False)
              .agg({
                  "_row": "first",
                  "EventId": "first",
                  "Label":  lambda s: int((s != 0).any())
              })
              .sort_values("_row")
              .drop(columns="_row")
              .reset_index(drop=True))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) SBERT 모델 로드 --------------------------------------------------
    with Timer("Load SBERT model (GPU)"):
        sbert = SentenceTransformer(EMBED_MODEL, device="cuda")

    # 2) 템플릿 CSV 읽기 + 임베딩 -----------------------------------------
    with Timer("Read TEMPLATE_CSV"):
        tmpl_df = pd.read_csv(TEMPLATE_CSV, dtype={"EventId": str})

    with Timer("SBERT encode templates"):
        tmpl_emb = sbert_encode(tmpl_df["EventTemplate"].tolist(),
                                sbert, BATCH_SIZE)
    id2idx = {eid: i for i, eid in enumerate(tmpl_df["EventId"].tolist())}

    # 3) 구조화 로그 읽기 + 라벨 변환 + dedup ------------------------------
    with Timer("Read STRUCT_CSV"):
        log_df = pd.read_csv(STRUCT_CSV, dtype={"EventId": str})
    log_df = log_df.reset_index(drop=True)

    # '-' → 0, 그 외 → 1
    log_df["Label"] = (log_df["Label"] != "-").astype(int)

    with Timer("Dedup consecutive EventId"):
        log_df = dedup_consecutive(log_df)
    print(f"[INFO] total log lines after dedup = {len(log_df):,}")

    # 4) 슬라이딩 윈도 -----------------------------------------------------
    event_idx = np.array([id2idx[e] for e in log_df["EventId"]], dtype=np.int32)
    flags     = log_df["Label"].astype(np.int8).values

    ctx_win, ae_win, mask_win, y_all, label_win = [], [], [], [], []
    with Timer("Build sliding windows"):
        for st in tqdm(range(0, len(event_idx) - WINDOW_SIZE + 1, STRIDE),
                       desc="windows", miniters=1000):
            ed        = st + WINDOW_SIZE
            win_ids   = event_idx[st:ed]
            win_flags = flags[st:ed]

            ctx_win.append(tmpl_emb[win_ids])
            ae_win.append(win_ids)
            mask_win.append(np.ones(WINDOW_SIZE, dtype=np.int8))
            y_all.append(int(win_flags.any()))
            label_win.append(win_flags)

    ctx_win   = np.asarray(ctx_win,   dtype=np.float16)
    ae_win    = np.asarray(ae_win,    dtype=np.int32)
    mask_win  = np.asarray(mask_win,  dtype=np.int8)
    y_all     = np.asarray(y_all,     dtype=np.int8)
    label_win = np.asarray(label_win, dtype=np.int8)

    # 5) 학습/테스트 분할 ---------------------------------------------------
    normal_idx = np.where(y_all == 0)[0]
    np.random.shuffle(normal_idx)
    n_train = int(len(normal_idx) * TRAIN_RATIO)
    train_norm_idx = normal_idx[:n_train]
    test_idx = np.concatenate([normal_idx[n_train:], np.where(y_all == 1)[0]])

    # 6) freq_map 저장 ------------------------------------------------------
    uniq, cnts = np.unique(ae_win[train_norm_idx].ravel(), return_counts=True)
    freq_map = dict(zip(uniq.tolist(), cnts.tolist()))
    with Timer("Save freq_map.json"):
        with open(os.path.join(OUT_DIR, "freq_map.json"), "w") as f:
            json.dump({str(k): int(v) for k, v in freq_map.items()}, f)

    # 7) numpy 파일 저장 ----------------------------------------------------
    with Timer("Save numpy arrays"):
        np.save(os.path.join(OUT_DIR, "X_train_ctx.npy"),  ctx_win[train_norm_idx])
        np.save(os.path.join(OUT_DIR, "X_test_ctx.npy"),   ctx_win[test_idx])
        np.save(os.path.join(OUT_DIR, "X_train_ae.npy"),   ae_win[train_norm_idx])
        np.save(os.path.join(OUT_DIR, "X_test_ae.npy"),    ae_win[test_idx])
        np.save(os.path.join(OUT_DIR, "X_train_mask.npy"), mask_win[train_norm_idx])
        np.save(os.path.join(OUT_DIR, "X_test_mask.npy"),  mask_win[test_idx])
        np.save(os.path.join(OUT_DIR, "y_test.npy"),       y_all[test_idx])
        np.save(os.path.join(OUT_DIR, "y_test2.npy"),      label_win[test_idx])

    print(f"[DONE] y_test:  {y_all[test_idx].shape} | "
          f"y_test2: {label_win[test_idx].shape} | "
          f"x_train_ctx: {ctx_win[train_norm_idx].shape}")

if __name__ == "__main__":
    main()
