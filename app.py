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

# 우측 상단 이름 표시 (Sub-title 느낌)
header_left, header_right = st.columns([4, 1])
with header_left:
    st.title("📈 시계열 분석 Project1 수요 예측")
with header_right:
    st.markdown("<br>", unsafe_allow_html=True) # 간격 조정
    st.subheader("C321032 박하율")

# 세션 상태 초기화
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame(
        columns=["모델 종류", "평가 방법", "상태", "MAE", "MdRAE", "TS", "예측 평균"]
    )

if "eval_preds" not in st.session_state:
    st.session_state.eval_preds = {}

# -----------------------------
# 분석 함수 (기존 로직 유지)
# -----------------------------
def load_data(file):
    for enc in ["utf-8", "cp949", "euc-kr"]:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except:
            continue
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
    naive = y[1:]
    y_t_minus_1 = y[:-1]
    denom = np.abs(naive - y_t_minus_1)
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
        try:
            model = ExponentialSmoothing(train, trend="add", seasonal="mul", seasonal_periods=12).fit()
        except:
            model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
        return model.forecast(horizon).values
    elif model_type in ["ARIMA", "SARIMA"]:
        is_seasonal = (model_type == "SARIMA")
        stepwise_model = auto_arima(train, seasonal=is_seasonal, m=12 if is_seasonal else 1, stepwise=True)
        return stepwise_model.predict(n_periods=horizon).values
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
# [상단 레이아웃] 데이터 업로드 & 모델 선택
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
            u_type = st.selectbox("시간 단위", ["일", "주", "월", "년"], index=2)
            e_type = st.selectbox("평가 방식", ["Rolling", "Expanding"])
        
        st.write("") # 간격
        btn_run = st.button("🚀 예측 실행", use_container_width=True, type="primary")
        if st.button("🗑️ 로그 초기화", use_container_width=True):
            st.session_state.results_df = pd.DataFrame(columns=["모델 종류", "평가 방법", "상태", "MAE", "MdRAE", "TS", "예측 평균"])
            st.session_state.eval_preds = {}
            st.rerun()

# -----------------------------
# [하단 레이아웃] 평가 결과 & 수요 예측 결과
# -----------------------------
if file and btn_run:
    # 데이터 분할
    split_idx = int(len(df_raw_data) * 0.8)
    train_set, test_set = df_raw_data[value_col][:split_idx], df_raw_data[value_col][split_idx:]
    
    # 1. 미래 예측 수행
    forecast_vals = get_best_forecast(train_set, h_len, m_type)
    avg_forecast = round(float(np.mean(forecast_vals)), 2)
    
    # 2. 성능 평가 시뮬레이션
    test_preds = run_eval_simulation(train_set, test_set, m_type)
    st.session_state.eval_preds[m_type] = test_preds
    
    # 지표 계산
    m_val = mae(test_set, test_preds)
    r_val = mdrae(test_set, test_preds)
    ts_val = tracking_signal(test_set, test_preds)
    status = "PASS" if abs(ts_val) < 4 else "FAIL"
    
    # 로그 업데이트 (예측 평균 포함)
    new_entry = pd.DataFrame([{
        "모델 종류": m_type, "평가 방법": e_type, "상태": status, 
        "MAE": round(m_val, 2), "MdRAE": round(r_val, 2), "TS": round(ts_val, 2),
        "예측 평균": avg_forecast
    }])
    st.session_state.results_df = pd.concat([st.session_state.results_df, new_entry], ignore_index=True)

# 결과 표시 구역
if not st.session_state.results_df.empty:
    bottom_left, bottom_right = st.columns(2)
    
    with bottom_left:
        with st.container(border=True):
            st.subheader("📏 평가 결과 및 로그")
            
            # 누적 로그 테이블
            st.dataframe(st.session_state.results_df, use_container_width=True, hide_index=True)
            
            # 성능 지표 시각화 (최근 모델 기준)
            fig_eval = go.Figure()
            fig_eval.add_trace(go.Scatter(x=test_set.index, y=test_set, name="Actual", mode='lines+markers'))
            for name, p_v in st.session_state.eval_preds.items():
                fig_eval.add_trace(go.Scatter(x=test_set.index, y=p_v, name=f"Pred({name})", line=dict(dash='dot')))
            fig_eval.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_eval, use_container_width=True)

    with bottom_right:
        with st.container(border=True):
            st.subheader(f"📊 수요 예측 결과 (시평={h_len}{u_type})")
            
            # 미래 날짜 생성
            freq_dict = {"일": "D", "주": "W", "월": "MS", "년": "YS"}
            future_dates = pd.date_range(df_raw_data.index[-1], periods=h_len+1, freq=freq_dict[u_type])[1:]
            
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(x=df_raw_data.index[-24:], y=df_raw_data[value_col][-24:], name="History", mode='lines'))
            fig_res.add_trace(go.Scatter(x=future_dates, y=forecast_vals, name="Forecast", line=dict(color='red', width=3)))
            fig_res.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_res, use_container_width=True)
            
            st.metric("최종 예상 수요 평균", f"{avg_forecast} {u_type}당")
            st.caption(f"범위: {round(min(forecast_vals),2)} ~ {round(max(forecast_vals),2)}")

elif file:
    st.info("💡 설정을 확인한 후 '예측 실행' 버튼을 눌러주세요.")
