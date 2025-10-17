import pandas as pd 
from datetime import datetime, timedelta
import numpy as np
from hmmlearn.hmm import GaussianHMM
import warnings
warnings.filterwarnings("ignore")
pd.options.display.float_format = lambda x: f"{x:.6f}"

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
# df = pd.read_excel('data/리스크보드New_v3_rawdata.xlsx',sheet_name='종합_',header=5,usecols="BT:FR")
df = pd.read_csv('data/리스크보드New_v4_rawdata.csv',header=5,usecols=list(rename_dict.keys()))
df = df[pd.to_datetime(df['Date.1']) < datetime.today() - timedelta(days=1)]

composite = df[['Date.1', '국내주식', '해외주식', '채권지수 ', 'FX 지수 ', '크레딧 지수 ', '국내 리스크종합지수.1', '글로벌 리스크종합지수.1']]
composite.columns = ['Date','K_EQUITY','G_EQUITY','FI','FX','CREDIT','KRCI','GRCI']

for c in ["K_EQUITY","G_EQUITY","FI","FX","CREDIT","KRCI","GRCI"]:
    composite[c] = pd.to_numeric(composite[c], errors="coerce")

def fit_hmm_posterior(series: pd.Series, n_states: int = 3, random_state: int = 100):
    """
    R(depmixS4)와 동일 컨셉:
      - 3상태 가우시안 HMM
      - 입력 series * 100 스케일
      - posterior 확률 반환 (T x 3), 상태 평균으로 Low/Mid/High 라벨링
    반환:
      post_df: DataFrame [Low, Mid, High]
      state_labels: Series('Low'/'Mid'/'High') - Viterbi 경로 라벨
      means_dict: {'Low': µ_low, 'Mid': µ_mid, 'High': µ_high}
      model: 학습된 GaussianHMM
    """
    s = series.dropna()
    X = (s.values.reshape(-1, 1).astype(float) * 100.0)
    idx = s.index

    model = GaussianHMM(n_components=n_states, covariance_type="full", random_state=random_state, n_iter=1000, tol=1e-6, init_params="stmcw")
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

    means_dict = {lab: float(means[k]) for k, lab in zip(order, ["Low","Mid","High"])}

    return post_df, state_labels, means_dict, model

def run_and_export(risk_df: pd.DataFrame, target_col: str, random_state: int = 100):
    """
    risk_df: [Date index] + 표준 컬럼(NEEDED)
    target_col: 'GRCI' 또는 'KRCI'
    out_csv: 출력 파일명
    """
    s = risk_df[target_col].astype(float)
    # R과 동일 개념
    post, state_labels, means_dict, model = fit_hmm_posterior(s, n_states=3, random_state=random_state)

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
    return out, means_dict, model

# def save_data():
# 3-1) GRCI
grci_out, grci_means, grci_model = run_and_export(composite, "GRCI", random_state=100)

# 3-2) KRCI
krci_out, krci_means, krci_model = run_and_export(composite, "KRCI", random_state=100)

rci = pd.merge(grci_out, krci_out, on="Date", how="left")
# df_3y = df[df['Date.1'] >= (df['Date.1'].max() - pd.DateOffset(years=3))]
# rci_3y = rci[rci['Date'] >= (df['Date.1'].max() - pd.DateOffset(years=3))]


df.rename(columns=rename_dict,inplace=True)
df = df.dropna(subset=['국내 리스크종합지수'])
rci = rci.dropna(subset=['KRCI'])
df.to_excel('data/economic_index_data.xlsx',index=False)
rci.to_excel('data/risk_composite_index_data.xlsx',index=False)