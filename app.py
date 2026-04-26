import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

# -----------------------------
# 기본 설정 및 헤더
# -----------------------------
st.set_page_config(layout="wide")

header_left, header_right = st.columns([4, 1])
with header_left:
    st.title("📈 시계열 분석 Project1 수요 예측")
with header_right:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("C321032 박하율")

# 세션 상태 초기화 (상태 컬럼 제외)
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame(
        columns=["모델 종류", "평가 방법", "MAE", "MdRAE", "TS", "예측 평균"]
    )

if "eval_preds" not in st.session_state:
    st.session_state.eval_preds = {}

# -----------------------------
# 분석 함수 (기존 로직 유지)
# -----------------------------
def load_data(file):
    for enc in ["utf-8", "cp949", "euc-kr"]:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except: continue
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

def mae(y, yhat): return np.mean(np.abs(np.array(y) - np.array(yhat)))

def mdrae(y, yhat):
    y, yhat = np.array(y), np.array(yhat)
    naive = y[1:]; y_prev = y[:-1]
    denom = np.abs(naive - y_prev)
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
        model = ExponentialSmoothing(train).fit()
        return model.forecast(horizon).values
    elif model_type == "Holt-Winters":
        try:
            model = ExponentialSmoothing(train, trend="add", seasonal="mul", seasonal_periods=12).fit()
        except:
            model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
        return model.forecast(horizon).values
    elif model_type in ["ARIMA", "SARIMA"]:
        is_seasonal = (model_type == "SARIMA")
        step_m = auto_arima(train, seasonal=is_seasonal, m=12 if is_seasonal else 1, stepwise=True)
        return step_m.predict(n_periods=horizon).values
    return np.repeat(train.iloc[-1], horizon)

def run_eval_simulation(train, test, model_type):
    history = list(train)
    preds = []
    for i in range(len(test)):
        yhat = get_best_forecast(history, 1, model_type)[0]
        preds.append(yhat)
        history.append(test.iloc[i])
    return np.array(preds)

# -----------------------------
# [상단 레이아웃]
# -----------------------------
top_left, top_right = st.columns(2)

with top_left:
    with st.container(border=True):
        st.subheader("📂 데이터 업로드")
        file = st.file_uploader("CSV 파일을 선택하세요", label_visibility="collapsed")
        if file:
            df = load_data(file)
            if df is not None:
                date_col = df.columns[0]
                value_col = df.select_dtypes(include=np.number).columns[0]
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.sort_values(date_col).set_index(date_col)
                raw = df[value_col].copy()
                proc = raw.interpolate().pipe(hampel_filter).pipe(fft_denoise)
                df[value_col] = proc
                
                fig_p = go.Figure()
                fig_p.add_trace(go.Scatter(x=df.index, y=raw, name="원본", line=dict(color="gray", width=1), opacity=0.3))
                fig_p.add_trace(go.Scatter(x=df.index, y=proc, name="전처리", line=dict(color="#00CC96")))
                fig_p.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_p, use_container_width=True)

with top_right:
    with st.container(border=True):
        st.subheader("⚙️ 모델 선택 및 설정")
        c1, c2 = st.columns(2)
        with c1:
            m_type = st.selectbox("예측 모델", ["이동평균", "지수평활", "Holt-Winters", "ARIMA", "SARIMA"])
            h_len = st.number_input("예측 길이(시평)", 5, 100, 12)
        with c2:
            u_type = st.selectbox("시간 단위", ["일", "주", "월", "년"], index=2)
            e_type = st.selectbox("평가 방식", ["Rolling", "Expanding"])
        
        btn_run = st.button("🚀 예측 실행", use_container_width=True, type="primary")
        if st.button("🗑️ 로그 초기화", use_container_width=True):
            st.session_state.results_df = pd.DataFrame(columns=["모델 종류", "평가 방법", "MAE", "MdRAE", "TS", "예측 평균"])
            st.session_state.eval_preds = {}
            st.rerun()

# -----------------------------
# [하단 레이아웃]
# -----------------------------
if file and btn_run:
    split_idx = int(len(df) * 0.8)
    train_set, test_set = df[value_col][:split_idx], df[value_col][split_idx:]
    
    # 예측 및 평가
    forecast_vals = get_best_forecast(train_set, h_len, m_type)
    avg_f = round(float(np.mean(forecast_vals)), 2)
    test_preds = run_eval_simulation(train_set, test_set, m_type)
    st.session_state.eval_preds[m_type] = test_preds
    
    # 로그 업데이트
    m_val, r_val, ts_val = mae(test_set, test_preds), mdrae(test_set, test_preds), tracking_signal(test_set, test_preds)
    new_entry = pd.DataFrame([{
        "모델 종류": m_type, "평가 방법": e_type, 
        "MAE": round(m_val, 2), "MdRAE": round(r_val, 2), "TS": round(ts_val, 2),
        "예측 평균": avg_f
    }])
    st.session_state.results_df = pd.concat([st.session_state.results_df, new_entry], ignore_index=True)

# 하단 결과 표시
if not st.session_state.results_df.empty:
    bottom_left, bottom_right = st.columns(2)
    
    with bottom_left:
        with st.container(border=True):
            st.subheader("📏 평가 결과 및 로그")
            st.dataframe(st.session_state.results_df, use_container_width=True, hide_index=True)
            
            # 지표 가이드 (복구 및 추가)
            g1, g2, g3 = st.columns(3)
            with g1:
                st.caption("**MAE** (평균 오차)")
                st.write(f"최근: `{st.session_state.results_df['MAE'].iloc[-1]}`")
                st.help("낮을수록 정확도가 높음을 의미합니다.")
            with g2:
                st.caption("**MdRAE** (상대 오차)")
                st.write(f"최근: `{st.session_state.results_df['MdRAE'].iloc[-1]}`")
                st.help("1보다 작으면 기본 모델(Naive)보다 성능이 우수합니다.")
            with g3:
                st.caption("**TS** (추적 신호)")
                st.write(f"최근: `{st.session_state.results_df['TS'].iloc[-1]}`")
                st.help("±4를 벗어나면 모델의 편향(Bias)이 발생한 것입니다.")

            # 비교 시각화 (Test Data 구간)
            fig_eval = go.Figure()
            fig_eval.add_trace(go.Scatter(x=test_set.index, y=test_set, name="실제값(Test)", mode='lines+markers'))
            for name, p_v in st.session_state.eval_preds.items():
                fig_eval.add_trace(go.Scatter(x=test_set.index, y=p_v, name=f"예측({name})", line=dict(dash='dash')))
            fig_eval.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig_eval, use_container_width=True)

    with bottom_right:
        with st.container(border=True):
            st.subheader(f"📊 수요 예측 결과 (전체 흐름)")
            
            # 미래 날짜 생성
            freq_dict = {"일": "D", "주": "W", "월": "MS", "년": "YS"}
            future_dates = pd.date_range(df.index[-1], periods=h_len+1, freq=freq_dict[u_type])[1:]
            
            # 전체 데이터 시각화 (Train + Test + Forecast)
            fig_all = go.Figure()
            fig_all.add_trace(go.Scatter(x=df.index, y=df[value_col], name="과거 데이터", line=dict(color="#1f77b4")))
            fig_all.add_trace(go.Scatter(x=future_dates, y=forecast_vals, name="미래 예측", line=dict(color="#ef553b", width=3)))
            
            # 예측 평균 지점 강조 (선택 사항)
            fig_all.add_hline(y=avg_f, line_dash="dot", annotation_text=f"평균: {avg_f}", annotation_position="bottom right")
            
            fig_all.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10), showlegend=True)
            st.plotly_chart(fig_all, use_container_width=True)
            
            st.info(f"💡 **예상 수요 분석**: 향후 {h_len}{u_type}간 평균 약 **{avg_f}**의 수요가 예상됩니다.")

elif file:
    st.info("👈 설정 후 '예측 실행' 버튼을 눌러 분석을 시작하세요.")
