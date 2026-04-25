import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from pmdarima import auto_arima

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(layout="wide")
st.title("📈 시계열 수요 예측 시스템")

if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame(
        columns=["모델", "평가방식", "MAE", "MdRAE", "TS"]
    )

if "plot_data" not in st.session_state:
    st.session_state.plot_data = []

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
    fft[int(n*keep_ratio):int(n*(1-keep_ratio))] = 0
    return np.fft.ifft(fft).real

def stationarity_tests(series):
    adf_stat, adf_p = adfuller(series.dropna())
    lb_p = acorr_ljungbox(series.dropna(), lags=[10], return_df=True)["lb_pvalue"].values[0]
    return adf_stat, adf_p, lb_p

def mae(y, yhat):
    return np.mean(np.abs(y - yhat))

def mdrae(y, yhat):
    naive = y.shift(1)
    return np.median(np.abs((y - yhat)/(y - naive)).dropna())

def tracking_signal(y, yhat):
    err = y - yhat
    return err.sum()/(np.mean(np.abs(err))+1e-8)

# -----------------------------
# 모델 함수
# -----------------------------
def run_model(train, test, model_type, unit, horizon):

    freq_map = {"일":7, "주":52, "월":12, "년":1}
    m_val = freq_map[unit]

    if model_type == "이동평균":
        preds = np.repeat(train.rolling(5).mean().iloc[-1], len(test))
        forecast = np.repeat(preds[-1], horizon)

    elif model_type == "지수평활":
        model = ExponentialSmoothing(train).fit()
        preds = model.forecast(len(test))
        forecast = model.forecast(horizon)

    elif model_type == "Holt-Winters":
        model = ExponentialSmoothing(
            train, trend="add", seasonal="add", seasonal_periods=m_val
        ).fit()
        preds = model.forecast(len(test))
        forecast = model.forecast(horizon)

    elif model_type in ["ARIMA", "SARIMA"]:
        train_log = np.log1p(train)

        model = auto_arima(
            train_log,
            seasonal=(model_type=="SARIMA"),
            m=m_val,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            max_p=3, max_q=3,
            max_P=2, max_Q=2
        )

        preds = np.expm1(model.predict(n_periods=len(test)))
        forecast = np.expm1(model.predict(n_periods=horizon))

    return preds, forecast

# -----------------------------
# 평가 함수
# -----------------------------
def evaluate(series, model_type, unit, eval_type):

    split = int(len(series)*0.8)

    if eval_type == "단일":
        train = series[:split]
        test = series[split:]
        preds, _ = run_model(train, test, model_type, unit, len(test))
        return test, preds

    preds_list = []
    actual_list = []

    for i in range(split, len(series)-1):

        if eval_type == "rolling":
            train = series[i-split:i]
        else:
            train = series[:i]

        test = series[i:i+1]

        p, _ = run_model(train, test, model_type, unit, 1)

        preds_list.append(p[0])
        actual_list.append(test.iloc[0])

    return pd.Series(actual_list), pd.Series(preds_list)

# -----------------------------
# UI
# -----------------------------
left, right = st.columns([1,1.4])

with left:
    file = st.file_uploader("CSV 업로드")

    if file:
        df = load_data(file)

        date_col = df.columns[0]
        value_col = df.select_dtypes(include=np.number).columns[0]

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).set_index(date_col)

        raw = df[value_col]
        proc = raw.interpolate().pipe(hampel_filter).pipe(fft_denoise)
        df[value_col] = proc

        st.line_chart(df[value_col])

        # 정상성 검정
        stat, p, lb = stationarity_tests(df[value_col])
        st.write(f"ADF p-value: {p:.4f}")
        st.write(f"Ljung-Box p-value: {lb:.4f}")

        model_type = st.selectbox(
            "모델",
            ["이동평균", "지수평활", "Holt-Winters", "ARIMA", "SARIMA"]
        )

        eval_type = st.selectbox(
            "평가 방식",
            ["단일", "rolling", "expanding"]
        )

        unit = st.selectbox("단위", ["일","주","월","년"])
        horizon = st.slider("예측 길이", 5, 100, 30)

        run = st.button("🚀 실행")

        if st.button("🗑️ 초기화"):
            st.session_state.results_df = st.session_state.results_df.iloc[0:0]
            st.session_state.plot_data = []
            st.rerun()

# -----------------------------
# RIGHT
# -----------------------------
with right:
    if file and run:

        actual, preds = evaluate(df[value_col], model_type, unit, eval_type)

        m = mae(actual, preds)
        r = mdrae(actual, preds)
        ts = tracking_signal(actual, preds)

        # 누적 저장
        st.session_state.plot_data.append({
            "actual": actual,
            "pred": preds,
            "label": f"{model_type}-{eval_type}"
        })

        st.session_state.results_df = pd.concat([
            st.session_state.results_df,
            pd.DataFrame([{
                "모델":model_type,
                "평가방식":eval_type,
                "MAE":round(m,2),
                "MdRAE":round(r,2),
                "TS":round(ts,2)
            }])
        ], ignore_index=True)

        # -----------------------------
        # 상단: 예측 결과
        # -----------------------------
        st.subheader("📈 예측 결과 (누적 비교)")

        fig = go.Figure()
        for item in st.session_state.plot_data:
            fig.add_trace(go.Scatter(y=item["actual"], name=f"Actual-{item['label']}"))
            fig.add_trace(go.Scatter(y=item["pred"], name=f"Pred-{item['label']}", line=dict(dash="dot")))

        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # 하단: 평가 결과
        # -----------------------------
        st.subheader("📊 평가 결과")

        c1,c2,c3 = st.columns(3)
        c1.metric("MAE", round(m,2))
        c2.metric("MdRAE", round(r,2))
        c3.metric("TS", round(ts,2))

        st.dataframe(st.session_state.results_df, use_container_width=True)

    elif file:
        st.info("👈 실행 버튼을 눌러주세요.")
