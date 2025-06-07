import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import FinanceDataReader as fdr
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI 주가 예측 챗봇", layout="wide")
st.title("🤖 AI 주가 예측 챗봇")

user_input = st.text_input("궁금한 종목명을 입력하세요 (예: 삼성전자, 카카오 등)", "삼성전자")

# 간단한 종목코드 매핑 (필요 시 확장 가능)
ticker_map = {
    "삼성전자": "005930",
    "카카오": "035720",
    "LG화학": "051910"
}

ticker = ticker_map.get(user_input.strip(), None)

if not ticker:
    st.warning("해당 종목은 아직 지원되지 않아요. (삼성전자, 카카오, LG화학만 지원 중)")
else:
    # 데이터 수집
    df = fdr.DataReader(ticker)
    df = df[['Close']].dropna()

    # 정규화
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    # 입력 시퀀스 생성
    X = []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i, 0])
    X = np.array(X).reshape(-1, 60, 1)
    y = scaled[60:]

    # 모델 구성 및 학습
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    # 예측
    latest_input = scaled[-60:].reshape(1, 60, 1)
    predicted = model.predict(latest_input, verbose=0)
    predicted_price = scaler.inverse_transform(predicted)[0][0]

    st.subheader(f"🔮 내일 {user_input} 예상 종가")
    st.success(f"{predicted_price:,.2f} 원")

    # 차트 출력
    df['Predicted'] = np.nan
    df.iloc[-1, df.columns.get_loc('Predicted')] = predicted_price
    st.subheader("📈 최근 주가 흐름 + 예측 종가")
    st.line_chart(df[-100:])
