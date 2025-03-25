import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import streamlit as st

# データ収集
def fetch_data(symbol, interval="15m", period="60d"):
    st.write("データを取得中...")
    df = yf.download(symbol, interval=interval, period=period)
    df['Returns'] = df['Close'].pct_change()
    return df.dropna()

# データ準備
def prepare_data(df, look_back=50):
    st.write("データを準備中...")
    X, y = [], []
    for i in range(len(df) - look_back):
        X.append(df.iloc[i:i+look_back, :4].values)  # Open, High, Low, Closeを使用
        y.append(df['Close'].iloc[i+look_back])
    return np.array(X), np.array(y)

# モデル構築
def build_model(input_shape):
    st.write("モデルを構築中...")
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# 学習と予測
def train_and_predict(model, X_train, y_train, X_test):
    st.write("モデルを学習中...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    st.write("予測中...")
    return model.predict(X_test)

# Streamlitアプリ
def main():
    st.title("FX未来相場予測ツール")
    st.write("このツールはドル円（USD/JPY）とGOLDの15分足データを使用して未来の価格を予測します。")

    # 銘柄選択
    symbol = st.selectbox("銘柄を選択してください", ["USDJPY=X", "GOLD=X"])
    look_back = st.slider("学習データの期間（ローソク足本数）", min_value=10, max_value=100, value=50, step=10)

    # データ取得
    df = fetch_data(symbol)
    st.line_chart(df['Close'])

    # データ準備
    X, y = prepare_data(df, look_back)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # モデル構築
    model = build_model(X_train.shape[1:])
    
    # 学習と予測
    if st.button("予測を開始する"):
        predictions = train_and_predict(model, X_train, y_train, X_test)
        st.write("予測結果：")
        st.line_chart(predictions.flatten())

if __name__ == "__main__":
    main()
