import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(layout="wide")
st.title("📈 시계열 수요 예측 시스템")

if "run_model" not in st.session_state:
    st.session_state.run_model = False

if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame(
        columns=["모델 종류", "평가 방법", "MAE", "MdRAE", "TS", "모델 적합 여부"]
    )

# -----------------------------
# 함수 정의
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

def adf_test(series):
    result = adfuller(series.dropna())
    return result[0], result[1]

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
left, right = st.columns([1, 1.4])

# =============================
# LEFT : 입력 / 전처리
# =============================
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

        with st.container(border=True):
            st.subheader("🔧 전처리 결과")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=raw, name="원본", opacity=0.4))
            fig.add_trace(go.Scatter(x=df.index, y=proc, name="전처리"))
            st.plotly_chart(fig, use_container_width=True)

            # ADF
            stat, p = adf_test(df[value_col])
            st.write("### 📉 정상성 검정 (ADF)")
            st.write(f"ADF Statistic: {stat:.4f}")
            st.write(f"p-value: {p:.4f}")

            if p < 0.05:
                st.success("✔ 정상성 만족")
            else:
                st.warning("⚠ 비정상 → 차분 필요")

        with st.container(border=True):
            st.subheader("⚙️ 모델 설정")

            model_type = st.selectbox(
                "모델",
                ["이동평균", "지수평활", "Holt-Winters", "ARIMA", "SARIMA", "AutoARIMA"]
            )

            horizon = st.number_input("예측 길이", 5, 200, 30)
            unit = st.selectbox("단위", ["일", "주", "월", "년"])
            eval_type = st.selectbox("평가 방식", ["단일", "rolling", "expanding"])

            if st.button("🚀 예측 실행"):
                st.session_state.run_model = True

        if st.button("🗑️ 초기화"):
            st.session_state.results_df = st.session_state.results_df.iloc[0:0]
            st.session_state.run_model = False
            st.rerun()

# =============================
# RIGHT : 결과
# =============================
with right:
    if file and st.session_state.run_model:

        split = int(len(df) * 0.8)
        train = df[value_col][:split]
        test = df[value_col][split:]

        # -----------------------------
        # 모델 학습 (한 번만!)
        # -----------------------------
        if model_type == "이동평균":
            preds = np.repeat(train.rolling(5).mean().iloc[-1], len(test))
            forecast = np.repeat(preds[-1], horizon)

        elif model_type == "지수평활":
            model = ExponentialSmoothing(train).fit()
            preds = model.predict(start=test.index[0], end=test.index[-1])
            forecast = model.forecast(horizon)

        elif model_type == "Holt-Winters":
            model = ExponentialSmoothing(
                train, trend="add", seasonal="add", seasonal_periods=12
            ).fit()
            preds = model.predict(start=test.index[0], end=test.index[-1])
            forecast = model.forecast(horizon)

        # -----------------------------
        # AutoARIMA 고성능 버전
        # -----------------------------
        from sklearn.preprocessing import MinMaxScaler
        
        # 1. 로그 변환
        train_log = np.log1p(train)
        
        # 2. 계절성 설정 (단위 기반)
        freq_map = {
            "일": 7,
            "주": 52,
            "월": 12,
            "년": 1
        }
        m_val = freq_map[unit]
        
        # 3. AutoARIMA 모델
        model = auto_arima(
            train_log,
            seasonal=(model_type == "SARIMA"),
            m=m_val,
        
            # 성능 + 속도 핵심 옵션
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        
            # 탐색 범위 제한 (속도 개선)
            max_p=3, max_q=3,
            max_P=2, max_Q=2,
        
            # 자동 차분
            d=None,
            D=None,
        
            trace=False
        )
        
        # 4. 테스트 예측
        preds_log = model.predict(n_periods=len(test))
        
        # 5. 미래 예측
        forecast_log = model.predict(n_periods=horizon)
        
        # 6. 역변환
        preds = np.expm1(preds_log)
        forecast = np.expm1(forecast_log)

        # -----------------------------
        # 평가
        # -----------------------------
        m = mae(test, preds)
        r = mdrae(test, preds)
        ts = tracking_signal(test, preds)
        status = "PASS" if abs(ts) < 4 else "FAIL"

        new_data = pd.DataFrame([{
            "모델 종류": model_type,
            "평가 방법": eval_type,
            "MAE": round(m, 2),
            "MdRAE": round(r, 2),
            "TS": round(ts, 2),
            "모델 적합 여부": status
        }])

        st.session_state.results_df = pd.concat(
            [st.session_state.results_df, new_data],
            ignore_index=True
        )

        # -----------------------------
        # 평가 시각화
        # -----------------------------
        with st.container(border=True):
            st.subheader("📊 모델 평가")

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[value_col],
                name="Actual",
                line=dict(color="green")
            ))

            fig.add_trace(go.Scatter(
                x=test.index,
                y=preds,
                name="Prediction",
                line=dict(color="red", dash="dot")
            ))

            st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # 미래 예측
        # -----------------------------
        with st.container(border=True):
            st.subheader("📈 미래 예측")

            freq_map = {"일":"D","주":"W","월":"M","년":"Y"}
            future_idx = pd.date_range(
                df.index[-1],
                periods=horizon+1,
                freq=freq_map[unit]
            )[1:]

            fig2 = go.Figure()

            fig2.add_trace(go.Scatter(
                x=df.index,
                y=df[value_col],
                name="Actual"
            ))

            fig2.add_trace(go.Scatter(
                x=future_idx,
                y=forecast,
                name="Forecast",
                line=dict(color="red")
            ))

            st.plotly_chart(fig2, use_container_width=True)

            st.success(
                f"👉 향후 {horizon}기간 수요: "
                f"{round(min(forecast),2)} ~ {round(max(forecast),2)}"
            )

        # -----------------------------
        # 지표
        # -----------------------------
        with st.container(border=True):
            st.subheader("📏 평가 지표")

            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", round(m, 2))
            c2.metric("MdRAE", round(r, 2))
            c3.metric("TS", round(ts, 2))

            if status == "PASS":
                st.success("✅ 적합한 모델")
            else:
                st.warning("⚠ 모델 재검토 필요")

        # -----------------------------
        # 로그
        # -----------------------------
        with st.container(border=True):
            st.subheader("📋 누적 결과")
            st.dataframe(st.session_state.results_df, use_container_width=True)

    elif file:
        st.info("👈 예측 실행 버튼을 눌러주세요.")
