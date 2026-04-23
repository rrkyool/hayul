import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA

# -----------------------------
# 📌 페이지 설정 (가로 확장)
# -----------------------------
st.set_page_config(layout="wide")

st.title("📈 시계열 분석 & 예측 웹앱")

# -----------------------------
# 📌 데이터 로딩 (안정 버전)
# -----------------------------
def load_data(file):
    try:
        file.seek(0)
        return pd.read_csv(file, encoding='utf-8')
    except:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding='cp949')
        except:
            file.seek(0)
            return pd.read_csv(file, encoding='cp949', sep=';')

uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)

    st.write("데이터 미리보기", df.head())

    # -----------------------------
    # 컬럼 선택
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        date_col = st.selectbox("날짜 컬럼", df.columns)

    with col2:
        value_col = st.selectbox("값 컬럼", df.columns)

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)
    df.set_index(date_col, inplace=True)

    # -----------------------------
    # 📌 전처리
    # -----------------------------
    st.subheader("⚙️ 전처리")

    col1, col2, col3 = st.columns(3)

    with col1:
        missing_method = st.selectbox("결측치 처리", ["none", "interpolate", "ffill", "bfill"])

    with col2:
        outlier_flag = st.checkbox("이상치 처리 (Hampel)")

    with col3:
        fft_flag = st.checkbox("FFT 디노이징")

    # 결측치
    if missing_method == "interpolate":
        df[value_col] = df[value_col].interpolate()
    elif missing_method == "ffill":
        df[value_col] = df[value_col].fillna(method='ffill')
    elif missing_method == "bfill":
        df[value_col] = df[value_col].fillna(method='bfill')

    # Hampel
    def hampel_filter(series, window=5, n=3):
    new = series.copy()

    for i in range(window, len(series) - window):
        win = series.iloc[i-window:i+window]

        med = np.median(win)
        mad = np.median(np.abs(win - med))

        if abs(series.iloc[i] - med) > n * mad:
            new.iloc[i] = med

    return new

    if outlier_flag:
        df[value_col] = hampel_filter(df[value_col])

    # FFT
    def fft_denoise(signal, keep_ratio=0.1):
    fft = np.fft.fft(signal)

    n = len(fft)
    cutoff = int(n * keep_ratio)

    fft[cutoff:n-cutoff] = 0

    return np.fft.ifft(fft).real
    
    if fft_flag:
    signal = df[value_col].values

    denoised = fft_denoise(signal)

    if len(denoised) == len(df):
        df[value_col] = denoised
    else:
        st.error("FFT 결과 길이가 맞지 않습니다.")

    # -----------------------------
    # 📊 그래프
    # -----------------------------
    st.subheader("📊 시계열 데이터")

    fig = px.line(df, y=value_col)
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # 📌 모델 선택
    # -----------------------------
    st.subheader("📈 모델 선택")

    model_type = st.selectbox("모델", ["평활법", "분해", "ARIMA"])

    forecast = None

    # -----------------------------
    # 평활법
    # -----------------------------
    if model_type == "평활법":
        trend = st.selectbox("추세", [None, "add"])
        seasonal = st.selectbox("계절성", [None, "add", "mul"])
        period = st.number_input("계절 주기", value=12)

        model = ExponentialSmoothing(
            df[value_col],
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=period
        ).fit()

        forecast = model.forecast(10)

    # -----------------------------
    # 분해
    # -----------------------------
    elif model_type == "분해":
        method = st.selectbox("분해 방식", ["additive", "multiplicative", "STL"])

        if method == "STL":
            res = STL(df[value_col]).fit()
            st.line_chart(res.trend)
            st.line_chart(res.seasonal)
            st.line_chart(res.resid)
        else:
            res = seasonal_decompose(df[value_col], model=method)
            st.line_chart(res.trend)

    # -----------------------------
    # ARIMA
    # -----------------------------
    elif model_type == "ARIMA":

        st.subheader("📌 통계 검정")

        adf_p = adfuller(df[value_col])[1]
        st.write(f"ADF p-value: {adf_p:.4f}")

        lb_p = acorr_ljungbox(df[value_col], lags=[10])['lb_pvalue'].values[0]
        st.write(f"Ljung-Box p-value: {lb_p:.4f}")

        p = st.slider("p", 0, 5, 1)
        d = st.slider("d", 0, 2, 1)
        q = st.slider("q", 0, 5, 1)

        model = ARIMA(df[value_col], order=(p,d,q)).fit()
        forecast = model.forecast(10)

    # -----------------------------
    # 📊 결과 시각화
    # -----------------------------
    if forecast is not None:
        future_index = pd.date_range(df.index[-1], periods=10, freq='D')

        fig2 = px.line()
        fig2.add_scatter(x=df.index, y=df[value_col], name="Actual")
        fig2.add_scatter(x=future_index, y=forecast, name="Forecast")

        st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------
    # 📏 평가
    # -----------------------------
    st.subheader("📏 평가 지표")

    def mae(y, yhat):
        return np.mean(np.abs(y - yhat))

    st.info("MAE: 평균 절대 오차 (예측 정확도)")

    train_size = int(len(df) * 0.7)
    train = df[value_col][:train_size]
    test = df[value_col][train_size:]

    model = ARIMA(train, order=(1,1,1)).fit()
    preds = model.forecast(len(test))

    st.write("MAE:", mae(test, preds))
