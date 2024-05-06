import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score

st.markdown("# PLS回帰")
st.markdown("PLS回帰が多次元かつ多重共線性のあるデータに対してどのように動作するか確認するページです。")


def get_pls_data(n_samples, n_features, n_informative, noise, n_redundant):
    X, y = make_regression(
        n_samples=int(n_samples * 2),
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=42,
    )

    res = pd.DataFrame(X)
    with warnings.catch_warnings():
        for i in range(n_redundant):
            res[f"{i}と共線性あり"] = res[i].copy() * i * 0.2

    res.columns = [f"特徴{i}" for i in res.columns]
    return res, y


st.markdown("## データ作成")
n_samples = st.slider("データ数", 50, 5000, 800)
n_features = st.slider("次元数", 6, 500, 1000)
n_informative = st.slider("意味のある特徴数", 2, np.min([n_features - 5, 100]), 50)
noise = st.slider("ノイズの強さ", 1, 5, 10)
n_redundant = st.slider("多重共線性のある次元数", 0, n_features, n_features - 5)

df, y = get_pls_data(n_samples, n_features, n_informative, noise, n_redundant)
st.table(df.head(5))

train_y, test_y = y[:n_samples], y[n_samples:]
train_X, test_X = df.iloc[:n_samples], df.iloc[n_samples:]

col1, col2 = st.columns(2)


def display_metric(y, pred, pr2=None, pmabse=None, pmae=None):
    r2 = np.round(r2_score(y, pred), 4)
    mabse = np.round(median_absolute_error(y, pred), 4)
    mae = np.round(mean_absolute_error(y, pred), 4)

    if pr2 is None:
        st.metric("決定係数", r2)
        st.metric("絶対誤差中央値", mabse)
        st.metric("絶対誤差平均値", mae)
    else:
        st.metric("決定係数", r2, delta=r2 - pr2)
        st.metric("絶対誤差中央値", mabse, delta=mabse - pmabse)
        st.metric("絶対誤差平均値", mae, delta=mae - pmae)
    return r2, mabse, mae


with col1:
    st.markdown("### 線形回帰")
    n_features_in_ = st.slider("n_features_in_", 2, n_features, n_features)
    linreg = LinearRegression().fit(train_X, train_y, n_features_in_)

    pred = linreg.predict(test_X)

    fig = plt.figure(figsize=(5, 5))
    plt.scatter(test_y, pred, s=1)
    st.pyplot(fig)
    r2, mabse, mae = display_metric(test_y, pred)

with col2:
    st.markdown("### PLS回帰")
    n_components = st.slider("n_components", 2, n_features, np.min([n_features, 40]))
    linreg = PLSRegression(n_components=n_components).fit(train_X, train_y)

    pred = linreg.predict(test_X)

    fig = plt.figure(figsize=(5, 5))
    plt.scatter(test_y, pred, s=1)
    st.pyplot(fig)
    display_metric(test_y, pred, pr2=r2, pmabse=mabse, pmae=mae)
