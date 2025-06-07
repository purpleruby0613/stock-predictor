import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 안정화를 위한 백엔드 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="AI 주가 예측 챗봇", layout="wide")
st.title("🤖 AI 주가 예측 챗봇")

# 사용자 입력
user_input = st.text_input("궁금한 종목명을 입력하세요 (예: 삼성전자, 카카오 등)", "삼성전자")

# 종목코드 매핑
ticker_map = {
    "삼성전자": "005930",
    "카카오": "035720",
    "LG화학": "051910"
}
ticker = ticker_map.get(user_input.strip(), None)

if not ticker:
    st.warning("현재는 삼성전자, 카카오, LG화학만 지원됩니다.")
    st.stop()

# 데이터 수집
try:
    df = fdr.DataReader(ticker)
    df = df[['Close']].dropna()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 데이터 전처리
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

X = []
for i in range(60, len(scaled)):
    X.append(scaled[i-60:i, 0])
X = np.array(X).reshape(-1, 60, 1)
y = scaled[60:]

# 모델 구성 및 학습
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=5, batch_size=32, verbose=0)

# 예측
latest_input = scaled[-60:].reshape(1, 60, 1)
predicted = model.predict(latest_input, verbose=0)
predicted_price = scaler.inverse_transform(predicted)[0][0]

# 결과 출력
st.subheader(f"🔮 내일 {user_input}의 예측 종가")
st.success(f"{predicted_price:,.2f} 원")

# 차트 표시
df['예측값'] = np.nan
df.iloc[-1, df.columns.get_loc('예측값')] = predicted_price
st.line_chart(df[-100:])
