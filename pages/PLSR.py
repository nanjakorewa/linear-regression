import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

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

    for i in range(X.shape[1]):
        X[:, i] += i * 3.0

    res = pd.DataFrame(X)

    np.random.seed(777)
    with warnings.catch_warnings():
        for i in range(n_redundant):
            res[f"{i}と共線性あり"] = res[i].copy() * np.random.randint(-10, 10)

    res.columns = [f"特徴{i}" for i in res.columns]
    return res, y


st.markdown("## データ作成")
n_samples = st.slider("データ数", 50, 5000, 1500)
n_features = st.slider("次元数", 6, 500, 1000)
n_informative = st.slider("意味のある特徴数", 2, np.min([n_features - 5, 100]), 20)
noise = st.slider("ノイズの強さ", 1, 5, 10)
n_redundant = st.slider("多重共線性のある次元数", 0, n_features, np.max([50, n_features - 5]))

df, y = get_pls_data(n_samples, n_features, n_informative, noise, n_redundant)
st.table(df.head(5))

train_y, test_y = y[:n_samples], y[n_samples:]
train_X, test_X = df.iloc[:n_samples], df.iloc[n_samples:]


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


st.markdown("## 線形回帰との比較")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 線形回帰")
    n_features_in_ = st.slider(
        "n_features_in_", 2, n_features, n_features, disabled=True
    )
    is_apply_standerdscaler_lin = st.checkbox("標準化を適用する")

    if is_apply_standerdscaler_lin:
        ss = StandardScaler()
        ss.fit(train_X)
        ss_train_X = ss.transform(train_X)
        ss_test_X = ss.transform(test_X)
        linreg = LinearRegression().fit(ss_train_X, train_y, n_features_in_)
        pred = linreg.predict(ss_test_X)
    else:
        linreg = LinearRegression().fit(train_X, train_y, n_features_in_)
        pred = linreg.predict(test_X)

    fig = plt.figure(figsize=(5, 5))
    plt.scatter(test_y, pred, s=1)
    st.pyplot(fig)
    r2, mabse, mae = display_metric(test_y, pred)

    st.markdown("")
    st.markdown("")
    st.markdown("##### 線形回帰モデルの係数")
    st.markdown("X=特徴のインデックス、Y=`LinearRegression.coef_`の値")
    fig = plt.figure(figsize=(12, 4))
    plt.plot(linreg.coef_)
    st.pyplot(fig)

with col2:
    st.markdown("### PLS回帰")
    n_components = st.slider("n_components", 2, n_features, np.min([n_features, 40]))
    is_apply_standerdscaler = st.checkbox("標準化を適用する", value=True)

    if is_apply_standerdscaler:
        ss = StandardScaler()
        ss.fit(train_X)
        ss_train_X = ss.transform(train_X)
        ss_test_X = ss.transform(test_X)
        plsreg = PLSRegression(n_components=n_components).fit(ss_train_X, train_y)
        pls_pred = plsreg.predict(ss_test_X)
    else:
        plsreg = PLSRegression(n_components=n_components, scale=False).fit(
            train_X, train_y
        )
        pls_pred = plsreg.predict(test_X)

    fig = plt.figure(figsize=(5, 5))
    plt.scatter(test_y, pls_pred, s=1)
    st.pyplot(fig)
    pls_r2, pls_mabse, pls_mae = display_metric(
        test_y, pls_pred, pr2=r2, pmabse=mabse, pmae=mae
    )

    st.markdown("")
    st.markdown("")
    st.markdown("##### PLS回帰モデルの係数")
    st.markdown("X=特徴のインデックス、Y=`PLSRegression.coef_`の値")
    fig = plt.figure(figsize=(12, 4))
    plt.plot(plsreg.coef_[0])
    st.pyplot(fig)
