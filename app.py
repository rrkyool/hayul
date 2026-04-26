import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

# -----------------------------
# 기본 설정 및 헤더
# -----------------------------
st.set_page_config(layout="wide")

header_left, header_right = st.columns([4, 1])
with header_left:
    st.title("📈 시계열 분석 Project1 수요 예측")
with header_right:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("C321032 박하율")

# 세션 상태 관리
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=["모델 종류", "평가 방법", "MAE", "MdRAE", "TS", "예측 평균"])
if "eval_preds" not in st.session_state:
    st.session_state.eval_preds = {}

# -----------------------------
# 분석 함수
# -----------------------------
def load_data(file):
    for enc in ["utf-8", "cp949", "euc-kr"]:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except: continue
    return None

def hampel_filter(series, window=5, n=3):
    series = series.astype(float)
    new = series.copy()
    for i in range(window, len(series)-window):
        win = series.iloc[i-window:i+window]
        med = np.median(win)
        mad = np.median(np.abs(win-med))
        if mad == 0: continue
        if abs(series.iloc[i]-med) > n*mad:
            new.iloc[i] = med
    return new

def fft_denoise(signal, keep_ratio=0.1):
    fft = np.fft.fft(signal)
    n = len(fft)
    cutoff = int(n * keep_ratio)
    fft[cutoff:n-cutoff] = 0
    return np.fft.ifft(fft).real

def mae(y, yhat): return np.mean(np.abs(np.array(y) - np.array(yhat)))

def mdrae(y, yhat):
    y, yhat = np.array(y), np.array(yhat)
    naive = y[1:]; y_prev = y[:-1]
    denom = np.abs(naive - y_prev)
    num = np.abs(naive - yhat[1:])
    return np.median(num / (denom + 1e-8))

def tracking_signal(y, yhat):
    err = np.array(y) - np.array(yhat)
    mad = np.mean(np.abs(err))
    return np.sum(err) / (mad + 1e-8)

def get_best_forecast(train, horizon, model_type):
    train = pd.Series(train).astype(float)
    if model_type == "이동평균":
        val = train.rolling(window=12, min_periods=1).mean().iloc[-1]
        return np.repeat(val, horizon)
    elif model_type == "지수평활":
        model = ExponentialSmoothing(train).fit()
        return model.forecast(horizon).values
    elif model_type == "Holt-Winters":
        try: model = ExponentialSmoothing(train, trend="add", seasonal="mul", seasonal_periods=12).fit()
        except: model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
        return model.forecast(horizon).values
    elif model_type in ["ARIMA", "SARIMA"]:
        is_seasonal = (model_type == "SARIMA")
        step_m = auto_arima(train, seasonal=is_seasonal, m=12 if is_seasonal else 1, stepwise=True)
        return step_m.predict(n_periods=horizon).values
    return np.repeat(train.iloc[-1], horizon)

def run_eval_simulation(train, test, model_type):
    history = list(train)
    preds = []
    for i in range(len(test)):
        yhat = get_best_forecast(history, 1, model_type)[0]
        preds.append(yhat)
        history.append(test.iloc[i])
    return np.array(preds)

# -----------------------------
# 상단 레이아웃
# -----------------------------
top_left, top_right = st.columns(2)

with top_left:
    with st.container(border=True):
        st.subheader("📂 데이터 업로드")
        file = st.file_uploader("CSV 파일을 선택하세요", label_visibility="collapsed")
        if file:
            df_raw_data = load_data(file)
            if df_raw_data is not None:
                date_col = df_raw_data.columns[0]
                value_col = df_raw_data.select_dtypes(include=np.number).columns[0]
                df_raw_data[date_col] = pd.to_datetime(df_raw_data[date_col])
                df_raw_data = df_raw_data.sort_values(date_col).set_index(date_col)
                
                raw_values = df_raw_data[value_col].copy()
                proc_values = raw_values.interpolate().pipe(hampel_filter).pipe(fft_denoise)
                df_raw_data[value_col] = proc_values
                
                fig_prep = go.Figure()
                fig_prep.add_trace(go.Scatter(x=df_raw_data.index, y=raw_values, name="원본", line=dict(color="gray", width=1), opacity=0.4))
                fig_prep.add_trace(go.Scatter(x=df_raw_data.index, y=proc_values, name="전처리", line=dict(color="#00CC96")))
                fig_prep.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_prep, use_container_width=True)

with top_right:
    with st.container(border=True):
        st.subheader("⚙️ 모델 선택 및 설정")
        c1, c2 = st.columns(2)
        with c1:
            m_type = st.selectbox("예측 모델", ["이동평균", "지수평활", "Holt-Winters", "ARIMA", "SARIMA"])
            h_len = st.number_input("예측 길이(시평)", 5, 100, 12)
            
        with c2:
            e_type = st.selectbox("평가 방식", ["Rolling", "Expanding"])
            u_type = st.selectbox("시간 단위", ["일", "주", "월", "년"], index=2)
            
        
        st.write("") # 간격
        btn_run = st.button("🚀 예측 실행", use_container_width=True, type="primary")
        if st.button("🗑️ 로그 초기화", use_container_width=True):
            st.session_state.results_df = pd.DataFrame(columns=["모델 종류", "평가 방법", "MAE", "MdRAE", "TS", "예측 평균"])
            st.session_state.eval_preds = {}
            st.rerun()

# -----------------------------
# 분석 실행 및 결과 레이아웃
# -----------------------------
if file and btn_run:
    split_idx = int(len(df) * 0.8)
    train_set, test_set = df[val_col][:split_idx], df[val_col][split_idx:]
    
    # 1. 시뮬레이션 (평가용)
    test_preds = run_eval_simulation(train_set, test_set, m_type)
    st.session_state.eval_preds[m_type] = test_preds
    
    # 2. 미래 예측 (전체 데이터 기반)
    forecast_vals = get_best_forecast(df[val_col], h_len, m_type)
    avg_f = round(float(np.mean(forecast_vals)), 1)
    
    # 3. 지표 계산 및 로그 저장
    m_val = round(mae(test_set, test_preds), 2)
    r_val = round(mdrae(test_set, test_preds), 2)
    ts_val = round(tracking_signal(test_set, test_preds), 2)
    
    new_entry = pd.DataFrame([{"모델 종류": m_type, "평가 방법": e_type, "MAE": m_val, "MdRAE": r_val, "TS": ts_val, "예측 평균": avg_f}])
    st.session_state.results_df = pd.concat([st.session_state.results_df, new_entry], ignore_index=True)

if not st.session_state.results_df.empty:
    bot_left, bot_right = st.columns([1, 1.2])
    
    with bot_left:
        with st.container(border=True):
            st.subheader("📏 평가 결과 및 로그")
            st.dataframe(st.session_state.results_df, use_container_width=True, hide_index=True)
            
            st.markdown("**💡 지표 참고 사항**: MAE(낮을수록 우수), MdRAE(<1 우수), TS(±4 이내 정상)")
            
            # [복구] 누적 시뮬레이션 비교 차트
            fig_eval = go.Figure()
            fig_eval.add_trace(go.Scatter(x=test_set.index, y=test_set, name="Actual", mode='lines+markers', line=dict(color="black")))
            for name, preds in st.session_state.eval_preds.items():
                fig_eval.add_trace(go.Scatter(x=test_set.index, y=preds, name=f"Pred({name})", line=dict(dash='dot')))
            fig_eval.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_eval, use_container_width=True)

    with bot_right:
        with st.container(border=True):
            st.subheader("📊 수요 예측 결과")
            
            freq_dict = {"일": "D", "주": "W", "월": "MS", "년": "YS"}
            future_dates = pd.date_range(df.index[-1], periods=h_len+1, freq=freq_dict[u_type])[1:]
            
            # [수정] Train + Test 전체를 포함한 시각화
            fig_all = go.Figure()
            # 과거 데이터 전체 (Train + Test)
            fig_all.add_trace(go.Scatter(x=df.index, y=df[val_col], name="과거 데이터", line=dict(color="#1f77b4")))
            # 미래 예측 데이터 (과거의 마지막 시점과 연결)
            fig_all.add_trace(go.Scatter(x=future_dates, y=forecast_vals, name="미래 예측", line=dict(color="#ef553b", width=3)))
            
            fig_all.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_all, use_container_width=True)
            
            st.info(f"✨ **결과 요약**: 향후 {h_len}{u_type}간 평균 예상 수요는 **{st.session_state.results_df['예측 평균'].iloc[-1]}**입니다.")
