# gui/upload_file.py
import streamlit as st

def upload_file_page():
    uploaded_file = st.file_uploader(
        "Android 로그 파일을 업로드하세요 (.log, .txt, .csv 등)",
        type=["log", "txt", "csv"],
    )

    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        try:
            text = bytes_data.decode("utf-8")
        except UnicodeDecodeError:
            text = bytes_data.decode("utf-8", errors="ignore")

        st.session_state["raw_text"] = text
        st.session_state["filename"] = uploaded_file.name

        st.success(f"업로드 완료: {uploaded_file.name}")
        lines = text.splitlines()
        st.write(f"총 라인 수: {len(lines)}")

        with st.expander("원본 로그 미리보기 (상위 50줄)"):
            st.text("\n".join(lines[:50]))
    else:
        st.info("로그 파일을 업로드하면 이후 단계에서 그대로 사용됩니다.")
