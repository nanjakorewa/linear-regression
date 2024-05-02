import japanize_matplotlib as jam
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


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

    st.markdown("## 各変数と残差の比較")
    fig = plt.figure(figsize=(8, 8))
    for i in range(1, 10):
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


N = int(st.slider("プロットするデータ数", 100, 1000, 500))

tab1, tab2, tab3, tab4 = st.tabs(["線形回帰が適切な例", "外れ値がある例", "非線形な例", "説明変数がXに含まれていない時"])

with tab1:
    X, y = make_regression(
        n_samples=N, n_features=9, noise=10, random_state=777, tail_strength=0.0
    )
    _do_residual_analysis(X, y, model=LinearRegression())

with tab2:
    anom_N = int(st.slider("はずれ値の数", 1, int(N / 10), 20))
    st.markdown("## はずれ値を除外しない場合")
    X, y = make_regression(
        n_samples=N, n_features=9, noise=10, random_state=777, bias=1000
    )
    for i in range(anom_N):
        y[i] *= 4
    mae, r2 = _do_residual_analysis(X, y, model=LinearRegression())
    st.markdown("## はずれ値を除外する場合")
    _do_residual_analysis(
        X[anom_N + 1 :], y[anom_N + 1 :], model=LinearRegression(), pmae=mae, pr2=r2
    )

with tab3:
    X, y = make_regression(
        n_samples=N, n_features=9, noise=10, random_state=777, bias=1000
    )
    y += 10
    y = y**3

    _do_residual_analysis(X, y, model=LinearRegression())

with tab4:
    X, y = make_regression(
        n_samples=N, n_features=12, noise=10, random_state=777, tail_strength=0.0
    )

    y = X[:, 10] + X[:, 11]
    _do_residual_analysis(X[:, :9], y, model=LinearRegression())
