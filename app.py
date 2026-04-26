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

# 모든 모델의 평가 결과를 저장할 딕셔너리
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

# 모델별 최적 학습 및 예측 함수 (성능 개선)
def get_best_forecast(train, horizon, model_type):
    if model_type == "이동평균":
        val = train.rolling(window=12, min_periods=1).mean().iloc[-1]
        return np.repeat(val, horizon)
    
    elif model_type == "지수평활":
        # 단순 지수 평활보다는 최적 alpha를 찾는 구조
        model = ExponentialSmoothing(train, trend=None, seasonal=None).fit(optimized=True)
        return model.forecast(horizon)
    
    elif model_type == "Holt-Winters":
        # 계절성 자동 탐색 및 최적화
        model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit(optimized=True)
        return model.forecast(horizon)
    
    elif model_type in ["ARIMA", "SARIMA"]:
        is_seasonal = (model_type == "SARIMA")
        stepwise_model = auto_arima(train, seasonal=is_seasonal, m=12 if is_seasonal else 1,
                                    stepwise=True, suppress_warnings=True, 
                                    error_action='ignore', max_p=3, max_q=3)
        return stepwise_model.predict(n_periods=horizon)
    return np.repeat(train.iloc[-1], horizon)

# -----------------------------
# 레이아웃
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
        
        # 1. 메인 예측 섹션
        with st.container(border=True):
            chart_title = f"시평 = {horizon}({unit})"
            st.subheader(f"📊 Test Data vs 예측 결과 ({chart_title})")
            
            try:
                with st.spinner(f"{model_type} 최적화 학습 중..."):
                    forecast = get_best_forecast(train, horizon, model_type)

                freq_map = {"일": "D", "주": "W", "월": "M", "년": "Y"}
                future_idx = pd.date_range(df.index[-1], periods=horizon+1, freq=freq_map[unit])[1:]
                
                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(x=train.index, y=train, name="Train", mode='lines+markers', line=dict(color='#1f77b4')))
                fig_res.add_trace(go.Scatter(x=test.index, y=test, name="Test", mode='lines+markers', line=dict(color='#ff7f0e')))
                fig_res.add_trace(go.Scatter(x=future_idx, y=forecast, name="Predicted", mode='lines+markers', line=dict(color='#2ca02c', dash='dot')))
                fig_res.update_layout(hovermode="x unified")
                st.plotly_chart(fig_res, use_container_width=True)

                st.info(f"💡 **예상 수요:** 평균 **{round(np.mean(forecast), 2)}** (범위: {round(np.min(forecast), 2)}~{round(np.max(forecast), 2)})")
            except Exception as e:
                st.error(f"모델 실행 에러: {e}")

        # 2. 평가 결과 및 로그 (이미지 스타일 시각화 포함)
        with st.container(border=True):
            st.subheader("📏 평가 결과 및 로그")
            try:
                with st.spinner("성능 평가 및 비교 시각화 생성 중..."):
                    # 현재 모델의 Test 구간 예측값 계산 (평가용)
                    if eval_type == "rolling":
                        current_preds = rolling_forecast(train, test, model_type)
                    else:
                        current_preds = expanding_forecast(train, test, model_type)
                    
                    # 세션에 현재 모델 결과 저장 (멀티 모델 시각화용)
                    st.session_state.eval_preds[model_type] = current_preds
                    
                    # [추가] 이미지 스타일의 멀티 모델 비교 그래프
                    fig_eval = go.Figure()
                    fig_eval.add_trace(go.Scatter(x=train.index[-20:], y=train.iloc[-20:], name="y_train (일부)", mode='lines+markers'))
                    fig_eval.add_trace(go.Scatter(x=test.index, y=test, name="y_test", mode='lines+markers', line=dict(color='orange')))
                    
                    for m_name, p_val in st.session_state.eval_preds.items():
                        fig_eval.add_trace(go.Scatter(x=test.index, y=p_val, name=f"Predicted ({m_name})", mode='lines+markers', line=dict(dash='dash')))
                    
                    fig_eval.update_layout(title="Validation 구간 모델별 성능 비교", height=400)
                    st.plotly_chart(fig_eval, use_container_width=True)

                    # 지표 계산 및 출력
                    m, r, ts = mae(test, current_preds), mdrae(test, current_preds), tracking_signal(test, current_preds)
                    status = "PASS" if abs(ts) < 4 else "FAIL"
                    
                    # 로그 누적
                    new_data = pd.DataFrame([{
                        "모델 종류": model_type, "평가 방법": eval_type, "상태": status,
                        "MAE": round(m, 2), "MdRAE": round(r, 2), "TS": round(ts, 2)
                    }])
                    st.session_state.results_df = pd.concat([st.session_state.results_df, new_data], ignore_index=True)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("MAE (정확도)", round(m, 2))
                    c2.metric("MdRAE (상대오차)", round(r, 2))
                    c3.metric("TS (편향)", round(ts, 2))

                    if status == "PASS": st.success("✅ 적절한 예측입니다.")
                    else: st.warning("⚠️ 모델 재검토가 필요합니다.")

                    st.divider()
                    st.write("📋 **누적 분석 로그**")
                    st.dataframe(st.session_state.results_df, use_container_width=True)
                    
                    st.session_state.run_model = False # 실행 완료 후 대기 상태로 전환
                
            except Exception as e:
                st.error(f"평가 에러: {e}")
                st.session_state.run_model = False

    elif file:
        st.info("👈 왼쪽에서 모델 옵션을 선택한 후 '예측 실행' 버튼을 눌러주세요.")
