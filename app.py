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
    return np.mean(np.abs(np.array(y) - np.array(yhat)))

def mdrae(y, yhat):
    y = np.array(y)
    yhat = np.array(yhat)
    naive = y[1:] # t 시점의 실제값
    y_t_minus_1 = y[:-1] # t-1 시점의 실제값 (Naive 예측)
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
        model = ExponentialSmoothing(train, trend=None, seasonal=None).fit()
        return model.forecast(horizon).values
    elif model_type == "Holt-Winters":
        try:
            model = ExponentialSmoothing(train, trend="add", seasonal="mul", seasonal_periods=12).fit()
        except:
            model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
        return model.forecast(horizon).values
    elif model_type in ["ARIMA", "SARIMA"]:
        is_seasonal = (model_type == "SARIMA")
        stepwise_model = auto_arima(train, seasonal=is_seasonal, m=12 if is_seasonal else 1,
                                    stepwise=True, suppress_warnings=True, error_action='ignore')
        return stepwise_model.predict(n_periods=horizon).values
    return np.repeat(train.iloc[-1], horizon)

def run_eval_simulation(train, test, model_type, eval_method):
    history = list(train)
    preds = []
    for i in range(len(test)):
        yhat = get_best_forecast(history, 1, model_type)[0]
        preds.append(yhat)
        history.append(test.iloc[i])
    return np.array(preds)

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
        
        with st.container(border=True):
            chart_title = f"시평 = {horizon}({unit})"
            st.subheader(f"📊 Test Data vs 예측 결과 ({chart_title})")
            try:
                with st.spinner("최적 모델 학습 중..."):
                    forecast = get_best_forecast(train, horizon, model_type)
                freq_map = {"일": "D", "주": "W", "월": "M", "년": "Y"}
                future_idx = pd.date_range(df.index[-1], periods=horizon+1, freq=freq_map[unit])[1:]
                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(x=train.index, y=train, name="Train", mode='lines+markers', line=dict(color='#1f77b4')))
                fig_res.add_trace(go.Scatter(x=test.index, y=test, name="Test", mode='lines+markers', line=dict(color='#ff7f0e')))
                fig_res.add_trace(go.Scatter(x=future_idx, y=forecast, name="Predicted", mode='lines+markers', line=dict(color='#2ca02c', dash='dot')))
                st.plotly_chart(fig_res, use_container_width=True)
                st.info(f"💡 예상 수요: 평균 **{round(float(np.mean(forecast)), 2)}** (범위: {round(float(np.min(forecast)), 2)}~{round(float(np.max(forecast)), 2)})")
            except Exception as e:
                st.error(f"모델 실행 에러: {e}")

        with st.container(border=True):
            st.subheader("📏 평가 결과 및 로그")
            try:
                with st.spinner("성능 평가 및 비교 중..."):
                    current_preds = run_eval_simulation(train, test, model_type, eval_type)
                    st.session_state.eval_preds[model_type] = current_preds
                    
                    fig_eval = go.Figure()
                    fig_eval.add_trace(go.Scatter(x=train.index[-24:], y=train.iloc[-24:], name="y_train", mode='lines+markers'))
                    fig_eval.add_trace(go.Scatter(x=test.index, y=test, name="y_test", mode='lines+markers', line=dict(color='orange')))
                    for m_name, p_val in st.session_state.eval_preds.items():
                        fig_eval.add_trace(go.Scatter(x=test.index, y=p_val, name=f"Predicted ({m_name})", mode='lines+markers', line=dict(dash='dash')))
                    st.plotly_chart(fig_eval, use_container_width=True)

                    # 지표 계산
                    m_val = float(mae(test, current_preds))
                    r_val = float(mdrae(test, current_preds))
                    ts_val = float(tracking_signal(test, current_preds))
                    status = "PASS" if abs(ts_val) < 4 else "FAIL"
                    
                    # 로그 누적
                    new_log = pd.DataFrame([{"모델 종류": model_type, "평가 방법": eval_type, "상태": status, "MAE": round(m_val, 2), "MdRAE": round(r_val, 2), "TS": round(ts_val, 2)}])
                    st.session_state.results_df = pd.concat([st.session_state.results_df, new_log], ignore_index=True)

                    # [복구] 이전 버전의 상세 지표 설명 레이아웃
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("MAE (평균 오차)", round(m_val, 2))
                        st.caption("정확도 지표 (낮을수록 좋음)")
                    with c2:
                        st.metric("MdRAE (상대 오차)", round(r_val, 2))
                        st.caption("1보다 작으면 Naive 모델보다 우수")
                    with c3:
                        st.metric("TS (추적 신호)", round(ts_val, 2))
                        st.caption("±4 범위를 벗어나면 편향 발생")

                    if status == "PASS":
                        st.success("✅ 현재 예측 모델의 성능이 안정적인 범위 안에 있습니다.")
                    else:
                        st.warning("⚠️ 모델 재검토가 필요합니다. 예측값이 한쪽으로 치우치고 있습니다.")

                    st.divider()
                    st.write("📋 **누적 분석 로그**")
                    st.dataframe(st.session_state.results_df, use_container_width=True)
                    st.session_state.run_model = False
            except Exception as e:
                st.error(f"평가 에러: {e}")
                st.session_state.run_model = False

    elif file:
        st.info("👈 왼쪽에서 '예측 실행' 버튼을 눌러주세요.")
