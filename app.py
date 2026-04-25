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
    # -----------------------------
    # 세션 상태 초기화 (결과 저장용)
    # -----------------------------
    if "results_df" not in st.session_state:
        st.session_state.results_df = pd.DataFrame(
            columns=["모델 종류", "평가 방법", "상태", "MAE", "MdRAE", "TS"]
        )

    if file and st.session_state.run_model:
        split = int(len(df) * 0.8)
        train, test = df[value_col][:split], df[value_col][split:]
        
        # 1. 시각화 섹션 (이미지 스타일 반영)
        with st.container(border=True):
            st.subheader(f"📊 Test Data vs 예측 결과 (시평={horizon})")
            try:
                # 모델별 예측 로직
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

                # 날짜 인덱스 생성
                freq_map = {"일": "D", "주": "W", "월": "M", "년": "Y"}
                future_idx = pd.date_range(df.index[-1], periods=horizon+1, freq=freq_map[unit])[1:]
                
                # Plotly 시각화 (업로드 이미지 스타일)
                fig_res = go.Figure()
                # Train
                fig_res.add_trace(go.Scatter(x=train.index, y=train, name="Train", mode='lines+markers', line=dict(color='#1f77b4')))
                # Test
                fig_res.add_trace(go.Scatter(x=test.index, y=test, name="Test", mode='lines+markers', line=dict(color='#ff7f0e')))
                # Predicted (Test 구간과 겹치는 예측값 시각화를 위해 별도 계산 혹은 Horizon 표시)
                fig_res.add_trace(go.Scatter(x=future_idx, y=forecast, name="Predicted", mode='lines+markers', line=dict(color='#2ca02c', dash='dot')))
                
                fig_res.update_layout(title=f"Test Data vs 예측 결과 (시평={horizon})", hovermode="x unified")
                st.plotly_chart(fig_res, use_container_width=True)
                
            except Exception as e:
                st.error(f"모델 실행 에러: {e}")

        # 2. 평가 및 로그 누적 섹션
        with st.container(border=True):
            st.subheader("📏 평가 결과 및 로그")
            try:
                # 평가 수행 (Rolling/Expanding)
                if eval_type == "rolling":
                    preds = rolling_forecast(train, test, model_type)
                else:
                    preds = expanding_forecast(train, test, model_type)
                
                m, r, ts = mae(test, preds), mdrae(test, preds), tracking_signal(test, preds)
                status = "PASS" if abs(ts) < 4 else "FAIL"
                
                # 결과 누적
                new_data = pd.DataFrame([{
                    "모델 종류": model_type,
                    "평가 방법": eval_type,
                    "상태": status,
                    "MAE": round(m, 2),
                    "MdRAE": round(r, 2),
                    "TS": round(ts, 2)
                }])
                st.session_state.results_df = pd.concat([st.session_state.results_df, new_data], ignore_index=True)

                # 지표 요약 (Metric)
                c1, c2, c3 = st.columns(3)
                c1.metric("MAE", round(m, 2))
                c2.metric("MdRAE", round(r, 2))
                c3.metric("TS", round(ts, 2))

                if status == "PASS":
                    st.success("✅ 적절한 예측입니다.")
                else:
                    st.warning("⚠️ 모델 재검토가 필요합니다 (TS 임계치 초과).")

                # 누적 데이터프레임 출력
                st.write("---")
                st.write("📋 **누적 분석 로그**")
                st.dataframe(st.session_state.results_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"평가 에러: {e}")

    elif file:
        st.info("👈 왼쪽에서 '예측 실행' 버튼을 눌러주세요.")

# 왼쪽 사이드바 하단 혹은 버튼 영역에 '초기화' 추가 (Left 섹션 수정 포함)
with left:
    if file:
        if st.button("🗑️ 예측 수행 초기화"):
            st.session_state.results_df = pd.DataFrame(columns=["모델 종류", "평가 방법", "상태", "MAE", "MdRAE", "TS"])
            st.session_state.run_model = False
            st.rerun()
