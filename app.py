# app.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="삼성전자 Dashboard", layout="wide")

DATA_PATH = Path("data") / "삼성전자_test.xlsx"  # 리포에 올린 그대로

@st.cache_data(show_spinner=False)
def load_excel(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"엑셀 파일이 없습니다: {path}")
    # 시트 목록
    xl = pd.ExcelFile(path, engine="openpyxl")
    sheets = xl.sheet_names

    # 각 시트 로드 (첫 행을 헤더로 가정)
    frames = {}
    for sh in sheets:
        df = xl.parse(sh, engine="openpyxl")
        # 공백 컬럼명 정리
        df.columns = [str(c).strip() for c in df.columns]
        # 날짜형 자동 추정
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    parsed = pd.to_datetime(df[col], errors="raise", infer_datetime_format=True)
                    # 날짜로 해석된 비율이 70% 이상일 때만 치환
                    mask = parsed.notna()
                    if mask.mean() >= 0.7:
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    pass
        frames[sh] = df
    return sheets, frames

def numeric_cols(df: pd.DataFrame):
    return [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]

def guess_date_col(df: pd.DataFrame):
    candidates = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
    # 흔한 한/영 컬럼명 우선
    pri = ["일자","날짜","Date","date","기준일","기준일자"]
    for p in pri:
        if p in df.columns and (np.issubdtype(df[p].dtype, np.datetime64)):
            return p
    return candidates[0] if candidates else None

st.title("📊 삼성전자 대시보드")

# 파일 경로 표기
st.caption(f"소스: `{DATA_PATH.as_posix()}`")

try:
    sheets, frames = load_excel(DATA_PATH)
except Exception as e:
    st.error(f"엑셀 로드 실패: {e}")
    st.stop()

# --- 사이드바 ---
st.sidebar.header("설정")
sheet = st.sidebar.selectbox("시트 선택", sheets)
df = frames[sheet].copy()

st.subheader(f"시트: {sheet}")
st.dataframe(df, use_container_width=True, hide_index=True)

# --- 차트 영역 ---
st.markdown("---")
st.subheader("차트")

num_cols = numeric_cols(df)
if not num_cols:
    st.info("숫자형 컬럼이 없어 차트를 그릴 수 없습니다. (엑셀 숫자 서식/텍스트 여부 확인)")
else:
    date_col = guess_date_col(df)
    mode = st.sidebar.radio("차트 x축 모드", ["인덱스", "날짜(추정)"], index=1 if date_col else 0)

    y_cols = st.multiselect("Y축(복수 선택 가능)", num_cols, default=num_cols[: min(3, len(num_cols))])

    if y_cols:
        plot_df = df[y_cols].copy()
        if mode == "날짜(추정)" and date_col:
            # 날짜 정렬
            sdf = df[[date_col] + y_cols].dropna(subset=[date_col]).sort_values(date_col)
            sdf = sdf.set_index(date_col)
            st.line_chart(sdf, use_container_width=True)
            st.caption(f"X축: {date_col} (자동 인식)")
        else:
            st.line_chart(plot_df, use_container_width=True)
            st.caption("X축: 행 인덱스")
    else:
        st.info("Y축에 표시할 숫자 컬럼을 선택해 주세요.")

# 다운로드(필요 시)
with st.expander("원본 데이터 내보내기"):
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("현재 시트 CSV 다운로드", data=csv, file_name=f"{sheet}.csv", mime="text/csv")

st.markdown("---")
st.caption("© risk-board · Streamlit")
