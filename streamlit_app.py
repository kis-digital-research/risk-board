import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import altair as alt
import time
from hmmlearn.hmm import GaussianHMM
import warnings
warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="우체국보험 리스크 스코어보드",   # 브라우저 탭 이름
    page_icon="📊"
)
st.markdown("""
<style>
    /* Fork 버튼 숨기기 */
    div[data-testid="stActionButton"] button[kind="header"]:nth-child(1) {
        display: none;
    }
    /* GitHub 아이콘 버튼 숨기기 */
    div[data-testid="stActionButton"] button[kind="header"]:nth-child(2) {
        display: none;
    }
    /* 메뉴 버튼 유지 */
    span[data-testid="stMainMenu"] {
        display: block;
    }
    .badge-low { background: #E8F5E9; color: #2E7D32; }     /* 안정 */
    .badge-mid { background: #FFFDE7; color: #F9A825; }     /* 중립 */
    .badge-high{ background: #FFEBEE; color: #C62828; }     /* 위험 */
}
</style>
""", unsafe_allow_html=True)

rename_dict = {
    'Date.1': 'Date',
    '시티 매크로 리스크 지수': 'Citi 매크로 리스크',
    '국고채장단기금리차(10Y-3Y).1': '국고채 장단기금리차(10Y-3Y)',
    'VKOSPI.1': 'KOSPI200 기대변동성(VKOSPI)',
    '외국인 순매수 혹은 지분율': '외국인 순매수(60일 누계)',
    '美주간경제인덱스(WEI).1': '美주간경제인덱스(WEI)',
    'ISM제조업지수.1': '미국 ISM제조업지수',
    'VIX Index.1': 'S&P500 기대변동성(VIX)',
    'CBOE Put/Call Ratio(센티멘트).1': '글로벌경기선행지수',
    '시티 경기서프라이즈 지수.1': '미국 경기서프라이즈 지수',
    '미국 정책불확실성지수.1': '미국 정책불확실성 지수',
    '글로벌주식모멘텀.1': '글로벌주식모멘텀',
    '국내주식': '국내주식리스크',
    '해외주식': '글로벌주식리스크',
    '국내 BEI Rate.1': '국내 BEI Rate',  # 필요시 '국내 기대인플레이션' 등으로 수정
    'JPM EMBI Global Spread.1': '신흥국채권 스프레드(JPM EMBI)',
    'MOVE.1': '채권 기대변동성(MOVE)',
    '미국 기대물가(BEI 5년).1': '미국 기대인플레이션(5년)',
    '미국채 10년물 금리.1': '미국채 10년물 금리',
    '채권지수 ': '채권리스크',
    '수출증가율 .1': '국내 수출증가율',
    '한국 CDS 프리미엄.2': '국내 CDS 프리미엄',
    '한국1Y-미국1Y 금리차.1': '한국1Y-미국1Y 금리차',
    '외환변동성지수(JPM).1': '외환변동성지수(JPM)',
    '달러인덱스(DXY).1': '달러인덱스(DXY)',
    'FX 지수 ': '외환리스크',
    'CD(91일).1': '국내 기준금리',
    '한국 CDS 프리미엄.3': '국내 CDS 프리미엄',
    '미국 하이일드 스프레드(OAS).1': '미국 하이일드 스프레드',
    '장단기금리차(미국채 10년물-2년물).1': '미 장단기금리차(10년-2년)',
    'BofA 메릴린치 글로벌 금융 스트레스.1': '글로벌 금융 스트레스(BofA)',
    '크레딧 지수 ': '크레딧/유동성리스크',
    '한국 기준금리.1': '국내 기준금리',
    '유가(WTI 최근월물 CL1).1': '유가(WTI 최근월물)',
    '글로벌 운임지수.1': '건화물 운임지수(BDI)',
    '상업용 부동산 공실률(CRE Vacancy rate)-cbre.1': '미국 상업용 부동산 공실률',
    '공실률 LA': '미국 상업용 부동산 공실률 (LA)',
    '공실률 보스턴': '미국 상업용 부동산 공실률 (보스턴)',
    '공실률 시카고': '미국 상업용 부동산 공실률 (시카고)',
    '애틀랜타 공실률.1': '미국 상업용 부동산 공실률 (애틀랜타)',
    '뉴욕 공실률.1': '오피스 공실률 (뉴욕)',
    '샌프란시스코 공실률.1': '오피스 공실률 (샌프란시스코)',
    '파리 오피스 공실률': '오피스 공실률 (파리)',
    '런던 오피스 공실률.1': '오피스 공실률 (런던)',
    '베를린 오피스 공실률.1': '오피스 공실률 (베를린)',
    '마드리드 오피스 공실률.1': '오피스 공실률 (마드리드)',
    '멜버른 오피스 공실률.1': '오피스 공실률 (멜버른)',
    'Fed Delinquency rate on loians secured by RE all commercial banks': '미국 부동산담보대출 연체율',
    '미국 모기지 금리(30년).1': '미국 모기지 금리(30년)',
    'S&P Case-Shiller 주택가격 지수.1': '미국 주택가격 지수',
    '국내 리스크종합지수.1': '국내 리스크종합지수',
    '글로벌 리스크종합지수.1': '글로벌 리스크종합지수',
    '원달러환율.1': '원달러 환율',
    '국내 경기선행지수 순환변동치': '국내 경기선행지수 순환변동치_절대수치',
    '국내 경기선행지수 순환변동치.1': '국내 경기선행지수 순환변동치'
}

def fit_hmm_posterior(series: pd.Series, n_states: int = 3, random_state: int = 100):

    s = series.dropna()
    X = (s.values.reshape(-1, 1).astype(float) * 100.0)
    idx = s.index

    model = GaussianHMM(n_components=n_states, covariance_type="diag", random_state=random_state, n_iter=200, tol=1e-6, init_params="stmcw")
    model.fit(X)

    # posterior(gamma)
    _, post = model.score_samples(X)  # shape: (T, n_states)

    # 상태 평균으로 라벨링(오름차순: 낮음=Low, 중간=Mid, 높음=High)
    means = model.means_.flatten()
    order = np.argsort(means)  # 작은→큰
    label_map = {order[0]: "Low", order[1]: "Mid", order[2]: "High"}

    post_df = pd.DataFrame(post, index=idx, columns=[f"st{k}" for k in range(n_states)])
    post_df = post_df.rename(columns={f"st{k}": label_map[k] for k in range(n_states)})
    post_df = post_df[["Low","Mid","High"]]

    # Viterbi 경로 (상태 인덱스 → 라벨)
    states = model.predict(X)
    state_labels = pd.Series([label_map[s] for s in states], index=idx, name="state")

    # means_dict = {lab: float(means[k]) for k, lab in zip(order, ["Low","Mid","High"])}

    return post_df, state_labels

def run_and_export(risk_df: pd.DataFrame, target_col: str, random_state: int = 100):
    """
    risk_df: [Date index] + 표준 컬럼(NEEDED)
    target_col: 'GRCI' 또는 'KRCI'
    out_csv: 출력 파일명
    """
    s = risk_df[target_col].astype(float)
    # R과 동일 개념
    post, state_labels = fit_hmm_posterior(s, n_states=3, random_state=random_state)

    # R: probs$state와 동일 개념 (Viterbi 경로의 라벨)
    # R은 숫자 state를 쓰지만, 여기서는 해석을 위해 라벨 문자열 사용.
    # 완전 동일하게 '숫자 state'가 필요하면 아래 줄을 바꾸면 됨.
    # -> numeric_state = model.predict((s.values.reshape(-1,1)*100.0)).astype(int) + 1
    #    out.insert(0, "state", numeric_state)
    out = risk_df.copy()
    out = out.join(post, how="left")
    out.insert(0, "state", state_labels)  # 첫 컬럼으로 state
    out = out.reset_index().rename(columns={"Date": "Date"})  # Date 컬럼화
    out = out[["Date", target_col, "state", "Low", "Mid", "High"]]
    out.columns = ["Date", target_col, target_col+"_state", target_col+"_Low", target_col+"_Mid", target_col+"_High"]
    return out


# 데이터 로드 함수
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('data/리스크보드New_v4_rawdata.csv', header=5, usecols=list(rename_dict.keys()))
    df = df[pd.to_datetime(df['Date.1']) < datetime.today() - timedelta(days=1)]
    
    
    composite = df[['Date.1', '국내주식', '해외주식', '채권지수 ', 'FX 지수 ', '크레딧 지수 ', '국내 리스크종합지수.1', '글로벌 리스크종합지수.1']]
    composite.columns = ['Date', 'K_EQUITY', 'G_EQUITY', 'FI', 'FX', 'CREDIT', 'KRCI', 'GRCI']
    
    for c in ["K_EQUITY", "G_EQUITY", "FI", "FX", "CREDIT", "KRCI", "GRCI"]:
        composite[c] = pd.to_numeric(composite[c], errors="coerce")
    
    grci_out = run_and_export(composite, "GRCI", random_state=100)
    krci_out = run_and_export(composite, "KRCI", random_state=100)
    
    rci = pd.merge(grci_out, krci_out, on="Date", how="left")
    risk_df = rci.dropna(subset=['KRCI'])
    risk_df['Date'] = pd.to_datetime(risk_df['Date'])
    risk_df = risk_df.sort_values('Date', ascending=False).reset_index(drop=True)
    
    df.rename(columns=rename_dict, inplace=True)

    econ_df = df.dropna(subset=['국내 리스크종합지수'])
    econ_df['Date'] = pd.to_datetime(econ_df['Date'])
    econ_df = econ_df.sort_values('Date', ascending=False).reset_index(drop=True)
    econ_df = econ_df.loc[:, ~econ_df.columns.duplicated()]
    
    for col in risk_df.columns:
        if col not in ['Date', 'GRCI_state', 'KRCI_state']:
            risk_df[col] = pd.to_numeric(risk_df[col].astype(str).str.strip(), errors='coerce')
    
    for col in econ_df.columns:
        if col != 'Date':
            econ_df[col] = pd.to_numeric(econ_df[col].astype(str).str.strip(), errors='coerce')
    
    return risk_df, econ_df

# 메인 함수에서 호출


# 변화량 계산 및 표시 함수
def get_change_symbol(change):
    #숫자형으로 변환 
    change = float(change)
    if np.isnan(change):
        return "-"
    if abs(round(change,2)) < 0.01:
        return "-"
    if change > 0:
        return f"▲ {abs(change):.2f}"
    elif change < 0:
        return f"▽ {abs(change):.2f}"
    else:
        return "-"

# 색상 적용 함수
def color_change(val):
    if '▲' in str(val):
        return 'color: red'
    elif '▽' in str(val):
        return 'color: blue'
    else:
        return 'color: gray'

def bytes_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")
        # 배지 CSS

state_color = {'안정': '#2E7D32', '중립': '#F9A825', '위험': '#C62828'}

k_indicators = [
        "Citi 매크로 리스크",
        "국고채 장단기금리차(10Y-3Y)",
        "KOSPI200 기대변동성(VKOSPI)",
        "외국인 순매수(60일 누계)",
        "미국 정책불확실성 지수",
        "국내 CDS 프리미엄",
        "원달러 환율",
    ]
k_categories = [
        "국내·외 주식 등",
        "국내·외 주식 등",
        "국내주식",
        "국내주식",
        "국내·외 주식 등",
        "외환",
        "외환"
    ]
g_indicators = [
        "Citi 매크로 리스크",
        "美주간경제인덱스(WEI)",
        "S&P500 기대변동성(VIX)",
        "미국 정책불확실성 지수",
        "글로벌주식모멘텀",
        "신흥국채권 스프레드(JPM EMBI)",
        "외환변동성지수(JPM)",
        "달러인덱스(DXY)",
        "미 장단기금리차(10년-2년)"
    ]
g_categories = [
        "국내·외 주식 등",
        "국내·외 주식 등",
        "해외주식",
        "국내·외 주식 등",
        "국내·외 주식 등",
        "해외채권",
        "외환",
        "외환",
        "크레딧"
    ]

k_equity_indicators = [
        "Citi 매크로 리스크",
        "美주간경제인덱스(WEI)",
        "미국 정책불확실성 지수",
        "글로벌경기선행지수",

        "KOSPI200 기대변동성(VKOSPI)",
        "국고채 장단기금리차(10Y-3Y)",
        "외국인 순매수(60일 누계)",
        "국내 경기선행지수 순환변동치"
    ]
k_equity_categories = [
        "공통요인",
        "공통요인",
        "공통요인",
        "공통요인",
        "국내 주식R 요인",
        "국내 주식R 요인",
        "국내 주식R 요인",
        "국내 주식R 요인"
    ]

g_equity_indicators = [
        "Citi 매크로 리스크",
        "美주간경제인덱스(WEI)",
        "미국 정책불확실성 지수",
        "글로벌경기선행지수",

        "미국 ISM제조업지수",
        "S&P500 기대변동성(VIX)",
        "미국 경기서프라이즈 지수",
        "글로벌주식모멘텀"
    ]
g_equity_categories = [
        "공통요인",
        "공통요인",
        "공통요인",
        "공통요인",
        "글로벌 주식R 요인",
        "글로벌 주식R 요인",
        "글로벌 주식R 요인",
        "글로벌 주식R 요인"
    ]

fi_indicators = [
    '신흥국채권 스프레드(JPM EMBI)',
    '채권 기대변동성(MOVE)',
    '미국 기대인플레이션(5년)',
    '미국채 10년물 금리'
    ]
fi_categories = [
        "채권R 요인",
        "채권R 요인",
        "채권R 요인",
        "채권R 요인"
        ]

fx_indicators = [
    '국내 수출증가율',
    '국내 CDS 프리미엄',
    '외환변동성지수(JPM)',
    '달러인덱스(DXY)'
    ]
fx_categories = [
        "외환R 요인",
        "외환R 요인",
        "외환R 요인",
        "외환R 요인"
        ]

cr_indicators = [
    '국내 CDS 프리미엄',
    '미국 하이일드 스프레드',
    '미 장단기금리차(10년-2년)',
    '글로벌 금융 스트레스(BofA)',
            ]
cr_categories = [
        "크레딧/유동성R 요인",
        "크레딧/유동성R 요인",
        "크레딧/유동성R 요인",
        "크레딧/유동성R 요인"
        ]

ai_indicators = [
    '국내 기준금리',
    '유가(WTI 최근월물)',
    '건화물 운임지수(BDI)',
    '미국 상업용 부동산 공실률',
    '미국 상업용 부동산 공실률 (LA)',
    '미국 상업용 부동산 공실률 (보스턴)',
    '미국 상업용 부동산 공실률 (시카고)',
    '미국 상업용 부동산 공실률 (애틀랜타)',
    '오피스 공실률 (뉴욕)',
    '오피스 공실률 (샌프란시스코)',
    '오피스 공실률 (파리)',
    '오피스 공실률 (런던)',
    '오피스 공실률 (베를린)',
    '오피스 공실률 (마드리드)',
    '오피스 공실률 (멜버른)',
    '미국 부동산담보대출 연체율',
    '미국 모기지 금리(30년)',
    '미국 주택가격 지수'
            ]
ai_categories = [
        "국내 대체투자R 요인",
        "공통 요인",
        "공통 요인",
        "해외 대체투자R 요인",
        "해외 대체투자R 요인",
        "해외 대체투자R 요인",
        "해외 대체투자R 요인",
        "해외 대체투자R 요인",
        "해외 대체투자R 요인",
        "해외 대체투자R 요인",
        "해외 대체투자R 요인",
        "해외 대체투자R 요인",
        "해외 대체투자R 요인",
        "해외 대체투자R 요인",
        "해외 대체투자R 요인",
        "해외 대체투자R 요인",
        "해외 대체투자R 요인",
        "해외 대체투자R 요인",
        ]

state_map = {"Low": ("안정", "badge-low"), "Mid": ("중립", "badge-mid"), "High": ("위험", "badge-high")}
RCI_map = {"KRCI": "국내 리스크 종합지수", "GRCI": "글로벌 리스크 종합지수"}
RCI_IMJ_map = {"KRCI": 'https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/1f1f0-1f1f7.svg',
                "GRCI": 'https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/1f30f.svg'}
# 메인 앱
def main():
    st.title("우체국보험 리스크 스코어보드")
    st.markdown("<div style='text-align: right; color: #909090;'>한국투자증권 리서치본부</div>", unsafe_allow_html=True)
    
    risk_df, econ_df = load_and_preprocess_data()
    
 # 사용 가능한 날짜 목록 (포맷팅)
    available_dates = risk_df['Date'].dt.strftime('%Y-%m-%d').unique()
    
    st.divider()
    st.write("")
  
    with st.sidebar:
        st.markdown("### 기준일자 선택")
        selected_date_str = st.selectbox("", available_dates)
        selected_date = datetime.strptime(selected_date_str, '%Y-%m-%d')
        selected_date_3y_ago = selected_date - pd.DateOffset(years=3)
        st.divider()
        
        # 섹션 이동 링크 추가
        st.markdown("### 리스크 섹션 바로가기")
        sections = [
            ("국내 리스크 종합지수 (KRCI)", "krci-section"),
            ("글로벌 리스크 종합지수 (GRCI)", "grci-section"),
            ("국내주식리스크", "k-equity-section"),
            ("글로벌주식리스크", "g-equity-section"),
            ("채권리스크", "fi-section"),
            ("외환리스크", "fx-section"),
            ("크레딧/유동성리스크", "cr-section"),
            ("대체투자리스크", "ai-section")
        ]
        for section_name, section_id in sections:
            st.markdown(
                f"""
                <a href="#{section_id}" onclick="document.getElementById('{section_id}').scrollIntoView({{behavior: 'smooth'}}); return false;">
                    {section_name}
                </a>
                """,
                unsafe_allow_html=True
            )
        st.divider()

        st.markdown("### 데이터(.CSV) 다운로드")
        st.download_button(
            "리스크 지표 DATA",
            data=bytes_csv(risk_df),
            file_name=f"risk_index_data_{selected_date.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        st.download_button(
            "경제 지표 DATA",
            data=bytes_csv(econ_df),
            file_name=f"economic_index_data_{selected_date.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        st.divider()
        st.markdown("<p style='font-size: 10px; color:grey'> · 본 자료는 고객의 증권투자를 돕기 위하여 작성된 당사의 저작물로서 모든 저작권은 당사에게 있으며, 당사의 동의 없이 어떤 형태로든 복제, 배포, 전송, 변형할 수 없습니다.</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 10px; color:grey'> · 본 자료는 리서치센터에서 수집한 자료 및 정보를 기초로 작성된 것이나 당사가 그 자료 및 정보의 정확성이나 완전성을 보장할 수는 없으므로 당사는 본 자료로써 고객의 투자 결과에 대한 어떠한 보장도 행하는 것이 아닙니다. 최종적 투자 결정은 고객의 판단에 기초한 것이며 본 자료는 투자 결과와 관련한 법적 분쟁에서 증거로 사용될 수 없습니다.</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 10px; color:grey'> · 본 자료에 제시된 종목들은 리서치센터에서 수집한 자료 및 정보 또는 계량화된 모델을 기초로 작성된 것이나, 당사의 공식적인 의견과는 다를 수 있습니다.</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 10px; color:grey'> · 이 자료에 게재된 내용들은 작성자의 의견을 정확하게 반영하고 있으며, 외부의 부당한 압력이나 간섭 없이 작성되었음을 확인합니다.</p>", unsafe_allow_html=True)
                                  

    # 이전 날짜 찾기 (다음 행, 전주, 전월, 전년도)
    idx = risk_df[risk_df['Date'] == selected_date].index[0]
    previous_data = {}

    # 다음 행 (이전 날짜)
    if idx + 1 < len(risk_df):
        previous_data['next'] = {
            'risk': risk_df.iloc[idx + 1],
            'econ': econ_df.iloc[idx + 1]
        }
    else:
        previous_data['next'] = None

    # 전월 (4주 전)
    if idx + 1 < len(risk_df):
        previous_data['month'] = {
            'risk': risk_df.iloc[idx + 4],
            'econ': econ_df.iloc[idx + 4]
        }
    else:
        previous_data['month'] = None

    # 전년도 (52주 전)
    if idx + 1 < len(risk_df):
        previous_data['year'] = {
            'risk': risk_df.iloc[idx + 52],
            'econ': econ_df.iloc[idx + 52]
        }
    else:
        previous_data['year'] = None

    # 데이터 유효성 체크 및 할당
    current_risk = risk_df[risk_df['Date'] == selected_date].iloc[0]
    current_econ = econ_df[econ_df['Date'] == selected_date].iloc[0]
    previous_risk = previous_data['next']['risk'] if previous_data['next'] else None
    previous_econ = previous_data['next']['econ'] if previous_data['next'] else None
    previous_month_risk = previous_data['month']['risk'] if previous_data['month'] else None
    previous_month_econ = previous_data['month']['econ'] if previous_data['month'] else None
    previous_year_risk = previous_data['year']['risk'] if previous_data['year'] else None
    previous_year_econ = previous_data['year']['econ'] if previous_data['year'] else None

    if not previous_data['next']:
        st.warning("이전 데이터가 없습니다.")
        return

    # 종합 지수 표시
    # st.subheader("종합 지수")
    
    ###########함수화
    def RCI_SECTION(RCI_name,indicators,categories):
        state_label, badge_class = state_map.get(current_risk[f'{RCI_name}_state'], ("Unknown", "badge-mid"))
        wow_change = get_change_symbol(current_risk[RCI_name]- previous_risk[RCI_name])
        wow_change_color = color_change(wow_change)
        mom_chage = get_change_symbol(current_risk[RCI_name]- previous_month_risk[RCI_name])
        mom_chage_color = color_change(mom_chage)
        yoy_chage = get_change_symbol(current_risk[RCI_name]- previous_year_risk[RCI_name])
        yoy_chage_color = color_change(yoy_chage)
        
        current = [current_econ.get(col, np.nan) for col in indicators]
        previous = [previous_econ.get(col, np.nan) for col in indicators]
        data = {
            '지표': indicators,
            '세부 리스크' : categories,

            '이전': [f"{p:.2f}" if not np.isnan(p) else '-' for p in previous],
            '현재': [f"{c:.2f}" if not np.isnan(c) else '-' for c in current],

            '변화': [get_change_symbol(c - p if not np.isnan(c) and not np.isnan(p) else np.nan) for c, p in zip(current, previous)]
        }
        df = pd.DataFrame(data)
        styled_df = df.style.applymap(color_change, subset=['변화'])

        chart_df = risk_df[(risk_df['Date'] >= selected_date_3y_ago) & (risk_df['Date'] <= selected_date)]
        chart_df = chart_df[['Date',RCI_name]]
        chart = (
            alt.Chart(chart_df)
            # .mark_line(color= state_color[state_label]
            .mark_line(color= "grey"
            #         #    , point=alt.OverlayMarkDef(color=state_color[state_label])
                       )           
            .encode(
                x=alt.X("Date:T", title=None, axis=alt.Axis(format="%Y-%m")),
                y=alt.Y(f"{RCI_name}:Q", title=f"{RCI_name}_last 3 years", scale=alt.Scale(domain=[chart_df[RCI_name].min()*0.95, chart_df[RCI_name].max()*1.05]))  # ★ Y축 범위 지정
            )
            .properties(height=200)
        )

        with st.container(border=True):
            st.markdown(f'<div id="{RCI_name.lower()}-section"></div>', unsafe_allow_html=True)
            for name, section_id in sections:
                    if name == RCI_name:
                        st.markdown(f'<div id="{section_id}"></div>', unsafe_allow_html=True)
            # st.write("")
            st.markdown(f"<div style='display: flex; align-items: center;'>\
                            <img src='{RCI_IMJ_map[RCI_name]}' style='height: 32px; margin-right: 6px;'>\
                            <h3 style=' align-items: center;'> {RCI_map[RCI_name]} : {current_risk[RCI_name]:.2f}</h3>\
                            <span class='{badge_class}' style='font-weight: bold;'>{state_label}</span>\
                        </div>", unsafe_allow_html=True)
            st.write(f"· 전주대비 : <span style='{wow_change_color}'>{wow_change} </span> \
                    전월대비 : <span style='{mom_chage_color}'>{mom_chage}</span>\
                    전년대비 : <span style='{yoy_chage_color}'>{yoy_chage}</span> "
                    , unsafe_allow_html=True)
            st.markdown(f"<div style='display: flex; align-items: center;'>\
                        <span style=white-space: pre;'>· 현재 &nbsp </span>\
                        <span class='{badge_class}' style='font-weight: bold;'>{state_label}</span>\
                        <span style= white-space: pre;'>국면 </span>\
                        <span style='color: gray; white-space: pre;'>   확률 : 안정({(current_risk[f'{RCI_name}_Low']*100):.2f}%), 중립({(current_risk[f'{RCI_name}_Mid']*100):.2f}%), 위험({(current_risk[f'{RCI_name}_High']*100):.2f}%) </span>\
                        </div>", unsafe_allow_html=True)
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(styled_df,hide_index=True, width=1000)
            # st.write("")

    def RISK_INDEX_SECTION(RI_name,indicators,categories):
            wow_change = "-"
            wow_change_color = "color: black;"
            mom_change = "-"
            mom_change_color = "color: black;"
            yoy_change = "-"
            yoy_change_color = "color: black;"
            if RI_name != "대체투자리스크":
                # state_label, badge_class = state_map.get(current_risk[f'{RI_name}_state'], ("Unknown", "badge-mid"))
                wow_change = get_change_symbol(current_econ[RI_name]- previous_econ[RI_name])
                wow_change_color = color_change(wow_change)
                mom_change = get_change_symbol(current_econ[RI_name]- previous_month_econ[RI_name])
                mom_change_color = color_change(mom_change)
                yoy_change = get_change_symbol(current_econ[RI_name]- previous_year_econ[RI_name])
                yoy_change_color = color_change(yoy_change)
            
            current = [current_econ.get(col, np.nan) for col in indicators]
            previous = [previous_econ.get(col, np.nan) for col in indicators]
            data = {
                '지표': indicators,
                '세부 리스크' : categories,
                
                '이전': [f"{p:.2f}" if not np.isnan(p) else '-' for p in previous],
                '현재': [f"{c:.2f}" if not np.isnan(c) else '-' for c in current],

                '변화': [get_change_symbol(c - p if not np.isnan(c) and not np.isnan(p) else np.nan) for c, p in zip(current, previous)]
            }
            df = pd.DataFrame(data)
            styled_df = df.style.applymap(color_change, subset=['변화'])

            with st.container(border=True):
                for name, section_id in sections:
                    if name == RI_name:
                        st.markdown(f'<div id="{section_id}"></div>', unsafe_allow_html=True)
                # st.write("")

                if RI_name != "대체투자리스크":
                    st.markdown(f"<div style='display: flex; align-items: center;'>\
                            <h3 style=' align-items: center;'> {RI_name} : {current_econ[RI_name]:.2f}</h3>\
                            </div>", unsafe_allow_html=True)
                    st.write(f"· 전주대비 : <span style='{wow_change_color}'>{wow_change} </span> \
                            전월대비 : <span style='{mom_change_color}'>{mom_change}</span>\
                            전년대비 : <span style='{yoy_change_color}'>{yoy_change}</span> "
                            , unsafe_allow_html=True)
                else : 
                    time.sleep(.5)
                    st.markdown(f"<div style='display: flex; align-items: center;'>\
                            <h3 style=' align-items: center;'> {RI_name}</h3>\
                            </div>", unsafe_allow_html=True)
                if RI_name =="대체투자리스크":
                    st.dataframe(styled_df,hide_index=True, width=1000, height=667)
                else :
                    st.dataframe(styled_df,hide_index=True, width=1000)
                # st.write("")
    st.subheader("종합 리스크지표", divider="grey")
    RCI_SECTION("KRCI",k_indicators,k_categories)    
    RCI_SECTION("GRCI",g_indicators,g_categories)
    st.write("")
    st.write("")
    st.subheader("세부 리스크지표", divider="grey")
    RISK_INDEX_SECTION("국내주식리스크",k_equity_indicators,k_equity_categories)
    RISK_INDEX_SECTION("글로벌주식리스크",g_equity_indicators,g_equity_categories)
    RISK_INDEX_SECTION("채권리스크",fi_indicators,fi_categories)
    RISK_INDEX_SECTION("외환리스크",fx_indicators,fx_categories)
    RISK_INDEX_SECTION("크레딧/유동성리스크",cr_indicators,cr_categories)
    RISK_INDEX_SECTION("대체투자리스크",ai_indicators,ai_categories)
    
if __name__ == "__main__":
    main()
