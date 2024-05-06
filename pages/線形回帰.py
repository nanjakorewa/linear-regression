import japanize_matplotlib as jam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

rng = np.random.default_rng()


def _do_residual_analysis(
    X, y, model=LinearRegression(), pmae=None, pr2=None, applylog=False
):
    if applylog:
        model.fit(X, np.sqrt(y))
        pred = model.predict(X)
        pred = pred**2 * np.sign(pred)
    else:
        model.fit(X, y)
        pred = model.predict(X)

    jam.japanize()
    st.markdown("## 正解と予測の散布図")
    fig = plt.figure(figsize=(4, 4))
    plt.scatter(y, pred, s=1)
    plt.xlabel("正解")
    plt.ylabel("予測")
    st.pyplot(fig)

    st.markdown("## 残差のヒストグラム")
    fig = plt.figure(figsize=(4, 2))
    plt.hist(y - pred)
    plt.xlabel("残差")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("## 各変数と残差の比較")
    fig = plt.figure(figsize=(8, 8))
    for i in range(1, X.shape[1] + 1):
        plt.subplot(3, 3, i)
        plt.scatter(X[:, i - 1], y - pred, s=1)
        plt.xlabel(f"x{i}")
        if i in [1, 4, 7]:
            plt.ylabel("残差")
    plt.tight_layout()
    st.pyplot(fig)

    col1, col2, col3 = st.columns(3)
    mae = np.round(mean_absolute_error(y, pred), 3)
    r2 = np.round(r2_score(y, pred), 3)
    with col1:
        if pmae:
            st.metric("Mean Absolute Error", mae, mae - pmae)
        else:
            st.metric("Mean Absolute Error", mae)
    with col2:
        if pr2:
            st.metric("R2 score", r2, r2 - pr2)
        else:
            st.metric("R2 score", r2)

    return mae, r2


@st.cache_data
def get_sample_data(n, pattern=0, allpositive=False, add_dummy=False):
    data = []

    for i in range(n):
        x1 = np.random.randint(10, 100)
        if allpositive:
            x2 = np.random.randint(10, 40)
        else:
            x2 = np.random.randint(-10, 20)
        x3 = np.random.randint(100, 200)

        y = (
            2.3 * (x1 + rng.standard_normal())
            - 10 * (x2 + rng.standard_normal())
            + 0.1 * (x3 + rng.standard_normal())
            + rng.standard_normal() * 10
        )

        if add_dummy:
            x4 = np.random.randint(-10, 10)
            x5 = np.random.randint(-200, -100)
            y += x4 * x1
            y += x5

        data.append([x1, x2, x3, y])

    df = pd.DataFrame(data)
    df.columns = ["x1", "x2", "x3", "y"]
    return df[["x1", "x2", "x3"]].values, df["y"].values


st.markdown("# 線形回帰")

N = int(st.slider("プロットするデータ数", 100, 1000, 500))

tab1, tab2, tab3, tab4 = st.tabs(
    ["線形回帰が適切な例", "外れ値がある例", "非線形な関係の例", "説明変数がXに含まれていない時"]
)

with tab1:
    X, y = get_sample_data(N)
    _do_residual_analysis(X, y, model=LinearRegression())

with tab2:
    anom_N = int(st.slider("はずれ値の数", 1, int(N / 10), 20))
    st.markdown("## はずれ値を除外しない場合")
    X, y = get_sample_data(N)
    for i in range(anom_N):
        y[i] *= 4
    mae, r2 = _do_residual_analysis(X, y, model=LinearRegression())
    st.markdown("## はずれ値を除外する場合")
    _do_residual_analysis(
        X[anom_N + 1 :], y[anom_N + 1 :], model=LinearRegression(), pmae=mae, pr2=r2
    )

with tab3:
    st.markdown("## 非線形の関係がある場合")
    X, y = get_sample_data(N, allpositive=True)
    X[:, 0] = X[:, 0] ** 2
    X[:, 1] = X[:, 1] ** 2
    X[:, 2] = X[:, 2] ** 2

    mae, r2 = _do_residual_analysis(X, y, model=LinearRegression())

    st.markdown("## 対数変換した場合")
    options = st.multiselect("対数変換する変数を選んでください", ["x1", "x2", "x3"], ["x1", "x2"])

    if "x1" in options:
        X[:, 0] = np.sqrt(X[:, 0])
    if "x2" in options:
        X[:, 1] = np.sqrt(X[:, 1])
    if "x3" in options:
        X[:, 2] = np.sqrt(X[:, 2])

    _do_residual_analysis(X, y, model=LinearRegression(), pmae=mae, pr2=r2)

with tab4:
    X, y = get_sample_data(N, allpositive=True, add_dummy=True)

    _do_residual_analysis(X[:, :3], y, model=LinearRegression())
