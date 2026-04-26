import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(layout="wide")
st.title("📈 시계열 분석 Project1 수요 예측")

if "run_model" not in st.session_state:
    st.session_state.run_model = False

if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame(
        columns=["모델 종류", "평가 방법", "상태", "MAE", "MdRAE", "TS"]
    )

if "eval_preds" not in st.session_state:
    st.session_state.eval_preds = {}

# -----------------------------
# 분석 함수 (강의 자료 가이드 반영)
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

def mae(y, yhat):
    return np.mean(np.abs(y - yhat))

def mdrae(y, yhat):
    naive = y.shift(1)
    return np.median(np.abs((y - yhat) / (y - naive)).dropna())

def tracking_signal(y, yhat):
    err = y - yhat
    return err.sum() / (np.mean(np.abs(err)) + 1e-8)

def rolling_forecast(train, test, model_type):
    history = list(train)
    preds = []
    for t in range(len(test)):
        if model_type == "ARIMA":
            model = ARIMA(history, order=(1,1,1)).fit()
            yhat = model.forecast()[0]
        elif model_type == "SARIMA":
            model = SARIMAX(history, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
            yhat = model.forecast()[0]
        else:
            yhat = history[-1]
        preds.append(yhat)
        history.append(test.iloc[t])
    return np.array(preds)

def expanding_forecast(train, test, model_type):
    preds = []
    for i in range(len(test)):
        hist = pd.concat([train, test[:i]])
        if model_type == "ARIMA":
            model = ARIMA(hist, order=(1,1,1)).fit()
            yhat = model.forecast()[0]
        elif model_type == "SARIMA":
            model = SARIMAX(hist, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
            yhat = model.forecast()[0]
        else:
            yhat = hist.iloc[-1]
        preds.append(yhat)
    return np.array(preds)

# -----------------------------
# [중요] 모델 학습 및 예측 로직 수정
# -----------------------------
def get_best_forecast(train, horizon, model_type):
    train = train.astype(float)
    
    if model_type == "이동평균":
        # 강의자료의 window_length 반영 (임의로 12 설정)
        val = train.rolling(window=12, min_periods=1).mean().iloc[-1]
        return np.repeat(val, horizon)
    
    elif model_type == "지수평활":
        # Simple Exponential Smoothing (Trend/Seasonal None)
        model = ExponentialSmoothing(train, trend=None, seasonal=None).fit()
        return model.forecast(horizon)
    
    elif model_type == "Holt-Winters":
        # 강의자료 가이드: 추세는 선형 증가(add), 계절성은 증가하므로(mul)
        # 데이터에 0이나 음수가 있으면 mul이 에러나므로 처리
        try:
            model = ExponentialSmoothing(
                train, 
                trend="add", 
                seasonal="mul", 
                seasonal_periods=12
            ).fit()
        except:
            # mul 에러 시 add로 대체
            model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
        return model.forecast(horizon)
    
    elif model_type in ["ARIMA", "SARIMA"]:
        is_seasonal = (model_type == "SARIMA")
        stepwise_model = auto_arima(train, seasonal=is_seasonal, m=12 if is_seasonal else 1,
                                    stepwise=True, suppress_warnings=True, 
                                    error_action='ignore', max_p=3, max_q=3)
        return stepwise_model.predict(n_periods=horizon)
    
    return np.repeat(train.iloc[-1], horizon)

# 평가용 시뮬레이션 함수 (학습 로직과 동일하게 유지)
def run_eval_simulation(train, test, model_type, eval_method):
    history = list(train)
    preds = []
    
    for i in range(len(test)):
        current_train = pd.Series(history)
        # 각 시점마다 예측 수행
        yhat = get_best_forecast(current_train, 1, model_type)[0]
        preds.append(yhat)
        
        if eval_method == "rolling":
            # 한 칸씩 이동하며 윈도우 유지 (여기서는 최신 데이터 추가)
            history.append(test.iloc[i])
        else:
            # 전체 누적
            history.append(test.iloc[i])
            
    return np.array(preds)

# -----------------------------
# 레이아웃 및 실행
# -----------------------------
left, right = st.columns([1, 1.3])

with left:
    with st.container(border=True):
        st.subheader("📂 데이터 업로드")
        file = st.file_uploader("CSV 업로드")
        if file:
            df = load_data(file)
            date_col = df.columns[0]
            value_col = df.select_dtypes(include=np.number).columns[0]
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col).set_index(date_col)
            
            raw = df[value_col].copy()
            proc = raw.interpolate().pipe(hampel_filter).pipe(fft_denoise)
            df[value_col] = proc
            st.success("✔ 전처리 완료")

            fig_pre = go.Figure()
            fig_pre.add_trace(go.Scatter(x=df.index, y=raw, name="원본", line=dict(color="gray"), opacity=0.4))
            fig_pre.add_trace(go.Scatter(x=df.index, y=proc, name="전처리 후", line=dict(color="blue")))
            st.plotly_chart(fig_pre, use_container_width=True)

    if file:
        with st.container(border=True):
            st.subheader("⚙️ 예측 설정")
            model_type = st.selectbox("모델", ["이동평균", "지수평활", "Holt-Winters", "ARIMA", "SARIMA"])
            horizon = st.number_input("예측 길이", 5, 200, 30)
            unit = st.selectbox("단위", ["일", "주", "월", "년"])
            eval_type = st.selectbox("평가 방식", ["rolling", "expanding"])
            
            if st.button("🚀 예측 실행"):
                st.session_state.run_model = True

        if st.button("🗑️ 예측 수행 초기화"):
            st.session_state.results_df = pd.DataFrame(columns=["모델 종류", "평가 방법", "상태", "MAE", "MdRAE", "TS"])
            st.session_state.eval_preds = {}
            st.session_state.run_model = False
            st.rerun()

with right:
    if file and st.session_state.run_model:
        split = int(len(df) * 0.8)
        train, test = df[value_col][:split], df[value_col][split:]
        
        with st.container(border=True):
            chart_title = f"시평 = {horizon}{unit}"
            st.subheader(f"📊 수요 예측 결과 ({chart_title})")
            
            try:
                with st.spinner(f"{model_type} 최적화 학습 중..."):
                    forecast = get_best_forecast(train, horizon, model_type)

                freq_map = {"일": "D", "주": "W", "월": "M", "년": "Y"}
                future_idx = pd.date_range(df.index[-1], periods=horizon+1, freq=freq_map[unit])[1:]
                
                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(x=train.index, y=train, name="Train", mode='lines+markers', line=dict(color='#1f77b4')))
                fig_res.add_trace(go.Scatter(x=test.index, y=test, name="Test", mode='lines+markers', line=dict(color='#ff7f0e')))
                fig_res.add_trace(go.Scatter(x=future_idx, y=forecast, name="Predicted", mode='lines+markers', line=dict(color='#2ca02c', dash='dot')))
                st.plotly_chart(fig_res, use_container_width=True)
            except Exception as e:
                st.error(f"모델 실행 에러: {e}")

        with st.container(border=True):
            st.subheader("📏 평가 결과 및 로그")
            try:
                with st.spinner("성능 평가 중..."):
                    # 시뮬레이션 기반 평가 수행
                    current_preds = run_eval_simulation(train, test, model_type, eval_type)
                    st.session_state.eval_preds[model_type] = current_preds
                    
                    # 시각화 비교
                    fig_eval = go.Figure()
                    fig_eval.add_trace(go.Scatter(x=train.index[-24:], y=train.iloc[-24:], name="y_train (Recent)", mode='lines+markers'))
                    fig_eval.add_trace(go.Scatter(x=test.index, y=test, name="y_test", mode='lines+markers', line=dict(color='orange')))
                    
                    for m_name, p_val in st.session_state.eval_preds.items():
                        fig_eval.add_trace(go.Scatter(x=test.index, y=p_val, name=f"Predicted ({m_name})", mode='lines+markers', line=dict(dash='dash')))
                    
                    st.plotly_chart(fig_eval, use_container_width=True)

                    m, r, ts = mae(test, current_preds), mdrae(test, current_preds), tracking_signal(test, current_preds)
                    status = "PASS" if abs(ts) < 4 else "FAIL"
                    
                    new_data = pd.DataFrame([{
                        "모델 종류": model_type, "평가 방법": eval_type, "상태": status,
                        "MAE": round(m, 2), "MdRAE": round(r, 2), "TS": round(ts, 2)
                    }])
                    st.session_state.results_df = pd.concat([st.session_state.results_df, new_data], ignore_index=True)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("MAE", round(m, 2))
                    c2.metric("MdRAE", round(r, 2))
                    c3.metric("TS", round(ts, 2))

                    st.dataframe(st.session_state.results_df, use_container_width=True)
                    st.session_state.run_model = False
            except Exception as e:
                st.error(f"평가 에러: {e}")
