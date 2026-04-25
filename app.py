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

# -----------------------------
# 세션 상태
# -----------------------------
if "results" not in st.session_state:
    st.session_state.results = []

if "run_model" not in st.session_state:
    st.session_state.run_model = False

if "forecast" not in st.session_state:
    st.session_state.forecast = None

# -----------------------------
# 함수
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
        if mad == 0:
            continue
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
            model = SARIMAX(history, order=(1,1,1),
                            seasonal_order=(1,1,1,12)).fit(disp=False)
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
            model = SARIMAX(hist, order=(1,1,1),
                            seasonal_order=(1,1,1,12)).fit(disp=False)
            yhat = model.forecast()[0]

        else:
            yhat = hist.iloc[-1]

        preds.append(yhat)

    return np.array(preds)


# -----------------------------
# 레이아웃
# -----------------------------
left, right = st.columns([1, 1.3])

# -----------------------------
# LEFT
# -----------------------------
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
            df[value_col] = df[value_col].astype(float)

            # 자동 전처리
            raw = df[value_col].copy()
            proc = raw.interpolate()
            proc = hampel_filter(proc)
            proc = fft_denoise(proc)
            df[value_col] = proc

            st.success("✔ 전처리 완료")

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df.index, y=raw,
                name="원본",
                line=dict(color="skyblue"),
                opacity=0.7
            ))
            
            fig.add_trace(go.Scatter(
                x=df.index, y=proc,
                name="전처리",
                line=dict(color="purple")
            ))

    if file:
        with st.container(border=True):
            st.subheader("⚙️ 예측 설정")

            model_type = st.selectbox("모델", [
                "이동평균",
                "지수평활",
                "Holt-Winters",
                "ARIMA",
                "SARIMA",
                "AutoARIMA"
            ])

            horizon = st.number_input("예측 길이", 5, 200, 30)

            unit = st.selectbox("단위", ["일", "주", "월", "년"])

            freq_map = {
                "일": "D",
                "주": "W",
                "월": "M",
                "년": "Y"
            }
            freq = freq_map[unit]

            eval_type = st.selectbox("평가 방식", ["rolling", "expanding"])

            if st.button("🚀 예측 실행"):
                st.session_state.run_model = True

            if st.button("🗑 평가 초기화"):
                st.session_state.results = []

# -----------------------------
# RIGHT
# -----------------------------
with right:
    if file:
        with st.container(border=True):
            st.subheader("📊 예측 결과")

            split = int(len(df)*0.8)
            train = df[value_col][:split]
            test = df[value_col][split:]

            forecast = None

            try:
                if model_type == "이동평균":
                    forecast = np.repeat(train.rolling(5).mean().iloc[-1], horizon)

                elif model_type == "지수평활":
                    model = ExponentialSmoothing(train).fit()
                    forecast = model.forecast(horizon)

                elif model_type == "Holt-Winters":
                    model = ExponentialSmoothing(
                        train, trend="add", seasonal="add", seasonal_periods=12
                    ).fit()
                    forecast = model.forecast(horizon)

                elif model_type == "ARIMA":
                
                    st.subheader("📊 통계 검정 결과")
                
                    adf_p = adfuller(train)[1]
                    lb_p = acorr_ljungbox(train, lags=[1])['lb_pvalue'].values[0]
                
                    st.write(f"ADF p-value: {adf_p:.4f}")
                    if adf_p < 0.05:
                        st.success("✔ 정상성 만족 → ARIMA 적용 가능")
                    else:
                        st.error("❌ 정상성 부족 → 차분 필요")
                
                    st.write(f"Ljung-Box p-value: {lb_p:.4f}")
                    if lb_p > 0.05:
                        st.success("✔ 잔차가 백색잡음에 가까움 → 모델 적절")
                    else:
                        st.warning("⚠ 자기상관 존재 → 모델 개선 필요")
                
                    model = ARIMA(train, order=(1,1,1)).fit()
                    forecast = model.forecast(horizon)

                elif model_type == "SARIMA":
                    model = SARIMAX(train,
                                    order=(1,1,1),
                                    seasonal_order=(1,1,1,12)).fit()
                    forecast = model.forecast(horizon)

                elif model_type == "AutoARIMA":
                    model = auto_arima(train, seasonal=True, m=12,
                                       suppress_warnings=True)
                    forecast = model.predict(n_periods=horizon)

            except Exception as e:
                st.error(e)
                forecast = np.repeat(train.iloc[-1], horizon)

            future_idx = pd.date_range(df.index[-1], periods=horizon, freq=freq)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train.index, y=train, name="Train"))
            fig.add_trace(go.Scatter(x=test.index, y=test, name="Test"))
            fig.add_trace(go.Scatter(x=future_idx, y=forecast, name="Forecast"))

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📌 수요 예측 결과")

            min_val = np.min(forecast)
            max_val = np.max(forecast)
            mean_val = np.mean(forecast)
            
            st.info(f"예측 수요는 약 {round(min_val,2)} ~ {round(max_val,2)} 범위이며, 평균은 {round(mean_val,2)} 입니다.")

        # -----------------------------
        # 평가
        # -----------------------------
        with st.container(border=True):
            st.subheader("📏 평가 지표")

            try:
                model = ARIMA(train, order=(1,1,1)).fit()
                preds = model.forecast(len(test))

                 if eval_type == "rolling":
                    preds = rolling_forecast(train, test, model_type)
                else:
                    preds = expanding_forecast(train, test, model_type)

                m = mae(test, preds)
                r = mdrae(test, preds)
                ts = tracking_signal(test, preds)

                fig2 = go.Figure()

                fig2.add_trace(go.Scatter(
                    x=train.index, y=train,
                    name="Train", line=dict(color="blue")
                ))
                
                fig2.add_trace(go.Scatter(
                    x=test.index, y=test,
                    name="Test", line=dict(color="orange")
                ))
                
                fig2.add_trace(go.Scatter(
                    x=test.index, y=preds,
                    name="Forecast", line=dict(color="green")
                ))
                
                # test 구간 강조
                fig2.add_vrect(
                    x0=test.index[0],
                    x1=test.index[-1],
                    fillcolor="orange",
                    opacity=0.1,
                    line_width=0
                )
                
                st.plotly_chart(fig2, use_container_width=True)

                result = {
                    "Model": model_type,
                    "MAE": round(m, 3),
                    "MdRAE": round(r, 3),
                    "TS": round(ts, 3)
                }

                st.session_state.results.append(result)

                df_res = pd.DataFrame(st.session_state.results)
                st.dataframe(df_res, use_container_width=True)

            except:
                st.warning("평가 실패")

