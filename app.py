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

# -----------------------------
# 분석 함수
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
# 메인 레이아웃
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
            model_type = st.selectbox("모델", ["이동평균", "지수평활", "Holt-Winters", "ARIMA", "SARIMA", "AutoARIMA"])
            horizon = st.number_input("예측 길이", 5, 200, 30)
            unit = st.selectbox("단위", ["일", "주", "월", "년"])
            eval_type = st.selectbox("평가 방식", ["rolling", "expanding"])
            if st.button("🚀 예측 실행"):
                st.session_state.run_model = True

with right:
    if file and st.session_state.run_model:
        split = int(len(df) * 0.8)
        train, test = df[value_col][:split], df[value_col][split:]
        
        with st.container(border=True):
            st.subheader("📊 예측 결과")
            try:
                if model_type == "이동평균":
                    forecast = np.repeat(train.rolling(5).mean().iloc[-1], horizon)
                elif model_type == "지수평활":
                    forecast = ExponentialSmoothing(train).fit().forecast(horizon)
                elif model_type == "Holt-Winters":
                    forecast = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit().forecast(horizon)
                elif model_type == "ARIMA":
                    forecast = ARIMA(train, order=(1,1,1)).fit().forecast(horizon)
                elif model_type == "SARIMA":
                    forecast = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False).forecast(horizon)
                else:
                    forecast = auto_arima(train, seasonal=True, m=12).predict(n_periods=horizon)

                freq_map = {"일": "D", "주": "W", "월": "M", "년": "Y"}
                future_idx = pd.date_range(df.index[-1], periods=horizon+1, freq=freq_map[unit])[1:]
                
                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(x=train.index, y=train, name="Train"))
                fig_res.add_trace(go.Scatter(x=test.index, y=test, name="Test"))
                fig_res.add_trace(go.Scatter(x=future_idx, y=forecast, name="Forecast"))
                st.plotly_chart(fig_res, use_container_width=True)
            except Exception as e:
                st.error(f"모델 에러: {e}")

        with st.container(border=True):
            st.subheader("📏 평가 지표")
            try:
                # 에러 발생 지점 정밀 수정
                if eval_type == "rolling":
                    preds = rolling_forecast(train, test, model_type)
                else:
                    preds = expanding_forecast(train, test, model_type)
                
                m, r, ts = mae(test, preds), mdrae(test, preds), tracking_signal(test, preds)
                c1, c2, c3 = st.columns(3)
                c1.metric("MAE", round(m, 2))
                c2.metric("MdRAE", round(r, 2))
                c3.metric("TS", round(ts, 2))
                
                if abs(ts) < 4: st.success("✅ 적절한 예측")
                else: st.warning("⚠️ 모델 재검토 필요")
            except Exception as e:
                st.error(f"평가 에러: {e}")
    elif file:
        st.info("👈 왼쪽에서 '예측 실행'을 눌러주세요.")
