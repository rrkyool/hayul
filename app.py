import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sktime.transformations.series.impute import Imputer
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(layout="wide")
st.title("📈 시계열 분석 & 예측")

# -----------------------------
# 유틸 함수
# -----------------------------
def hampel_filter(series, window=5, n=3):
    series = series.astype(float)
    new = series.copy()

    if len(series) < window * 2:
        return new

    for i in range(window, len(series) - window):
        win = series.iloc[i-window:i+window].dropna()
        if len(win) == 0:
            continue

        med = np.median(win)
        mad = np.median(np.abs(win - med))

        if mad == 0:
            continue

        if abs(series.iloc[i] - med) > n * mad:
            new.iloc[i] = med

    return new


def fft_denoise(signal, keep_ratio=0.1):
    signal = np.array(signal, dtype=float)
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


# -----------------------------
# 레이아웃
# -----------------------------
left, right = st.columns([1, 1.2])

# -----------------------------
# 📌 LEFT: 데이터 + 옵션
# -----------------------------
with left:

    st.subheader("1️⃣ 데이터 업로드")
    file = st.file_uploader("CSV 업로드")

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        date_col = st.selectbox("날짜 컬럼", df.columns)
        value_col = st.selectbox("값 컬럼", df.select_dtypes(include=np.number).columns)

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).set_index(date_col)

        # 🔥 dtype 안정화
        df[value_col] = df[value_col].astype(float)

        st.subheader("2️⃣ 전처리")

        # 결측치 (sktime)
        if st.checkbox("결측치 처리"):
            imputer = Imputer(method="drift")
            df[value_col] = imputer.fit_transform(df[value_col])

        # 이상치
        if st.checkbox("이상치 제거 (Hampel)"):
            df[value_col] = hampel_filter(df[value_col])

        # FFT
        if st.checkbox("노이즈 제거 (FFT)"):
            df[value_col] = fft_denoise(df[value_col])

        st.line_chart(df[value_col])

        st.subheader("3️⃣ 예측 옵션")

        model_type = st.selectbox("모델 선택", [
            "이동평균",
            "지수평활",
            "Holt",
            "Holt-Winters",
            "분해",
            "STL",
            "ARIMA"
        ])

        horizon = st.number_input("예측 기간", 5, 200, 30)

        freq = st.selectbox("단위", ["D", "M", "Y"])

        eval_method = st.selectbox("평가 방식", ["rolling", "expanding"])


# -----------------------------
# 📌 RIGHT: 결과
# -----------------------------
with right:

    if file:

        st.subheader("4️⃣ 예측 결과")

        # train/test
        split = int(len(df) * 0.8)
        train = df[value_col][:split]
        test = df[value_col][split:]

        forecast = None

        try:
            if model_type == "이동평균":
                forecast = train.rolling(5).mean().iloc[-1]
                forecast = np.repeat(forecast, horizon)

            elif model_type == "지수평활":
                model = ExponentialSmoothing(train).fit()
                forecast = model.forecast(horizon)

            elif model_type == "Holt":
                model = ExponentialSmoothing(train, trend="add").fit()
                forecast = model.forecast(horizon)

            elif model_type == "Holt-Winters":
                model = ExponentialSmoothing(
                    train,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=12
                ).fit()
                forecast = model.forecast(horizon)

            elif model_type == "분해":
                res = seasonal_decompose(train)
                st.line_chart(res.trend)
                forecast = np.repeat(train.iloc[-1], horizon)

            elif model_type == "STL":
                res = STL(train).fit()
                st.line_chart(res.trend)
                forecast = np.repeat(train.iloc[-1], horizon)

            elif model_type == "ARIMA":

                st.write("ADF p-value:", adfuller(train)[1])
                st.write("Ljung-Box p-value:", acorr_ljungbox(train, lags=[1])['lb_pvalue'].values[0])

                model = ARIMA(train, order=(1,1,1)).fit()
                forecast = model.forecast(horizon)

        except Exception as e:
            st.error(f"모델 오류 발생 → {e}")
            forecast = np.repeat(train.iloc[-1], horizon)

        # index 생성
        future_idx = pd.date_range(df.index[-1], periods=horizon, freq=freq)

        # 그래프
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=train.index, y=train, name="Train"))
        fig.add_trace(go.Scatter(x=test.index, y=test, name="Test"))
        fig.add_trace(go.Scatter(x=future_idx, y=forecast, name="Forecast"))

        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # 📏 평가
        # -----------------------------
        st.subheader("5️⃣ 평가 지표")

        try:
            model = ARIMA(train, order=(1,1,1)).fit()
            preds = model.forecast(len(test))

            m = mae(test, preds)
            r = mdrae(test, preds)
            ts = tracking_signal(test, preds)

            st.metric("MAE", round(m, 3))
            st.metric("MdRAE", round(r, 3))
            st.metric("TS", round(ts, 3))

            st.info("""
MAE: 평균 절대 오차 (작을수록 좋음)  
MdRAE: 상대 오차 (1보다 작으면 좋음)  
TS: -4 ~ 4 범위면 안정적  
""")

            if abs(ts) < 4:
                st.success("✅ 이 예측은 적절합니다.")
            else:
                st.error("❌ 편향된 예측입니다. 모델을 바꿔보세요.")

        except:
            st.warning("평가 실패")
