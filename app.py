import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.optimize import minimize

st.set_page_config(page_title="聚丙烯拉伸强度模型", layout="wide")
st.title("🧪 聚丙烯拉伸强度模型：性能预测 与 逆向设计")

page = st.sidebar.selectbox("🔧 选择功能", ["性能预测", "逆向设计"])

# 加载模型与缩放器
data = joblib.load("model_and_scaler_ts1.pkl")
model = data["model"]
scaler = data["scaler"]

df = pd.read_excel("trainrg3.xlsx")
feature_names = df.columns.tolist()
if "TS" in feature_names:
    feature_names.remove("TS")

unit_type = st.radio("📏 请选择配方输入单位", ["质量 (g)", "质量分数 (wt%)", "体积分数 (vol%)"], horizontal=True)

if page == "性能预测":
    st.subheader("🔬 正向预测：配方 → 拉伸强度 (TS)")

    with st.form("input_form"):
        user_input = {}
        total = 0
        cols = st.columns(3)
        for i, name in enumerate(feature_names):
            unit_label = {
                "质量 (g)": "g",
                "质量分数 (wt%)": "wt%",
                "体积分数 (vol%)": "vol%"
            }[unit_type]
            val = cols[i % 3].number_input(f"{name} ({unit_label})", value=0.0, step=0.1 if "质量" in unit_type else 0.01)
            user_input[name] = val
            total += val

        # 仅当输入的总和不为0时才检查是否为100
        inputs_valid = True
        if unit_type != "质量 (g)" and total != 0 and abs(total - 100) > 1e-3:
            st.warning("⚠️ 当前输入为分数单位，总和必须为 100。请检查输入是否正确。")
            inputs_valid = False

        submitted = st.form_submit_button("📊 开始预测", disabled=not inputs_valid)

    if submitted:
        # 若是分数单位，则再归一化一遍
        if unit_type != "质量 (g)" and total > 0:
            user_input = {k: v / total * 100 for k, v in user_input.items()}

        input_array = np.array([list(user_input.values())])
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]

        st.markdown("### 🎯 预测结果")
        st.metric(label="拉伸强度 (TS)", value=f"{prediction:.2f} MPa")

elif page == "逆向设计":
    st.subheader("🎯 逆向设计：拉伸强度 (TS) → 配方")

    target_ts = st.number_input("🎯 请输入目标 TS 值 (MPa)", value=50.0, step=0.1)

    if st.button("🔄 开始逆向设计"):
        with st.spinner("正在反推出最优配方，请稍候..."):

            # 初始猜测：随机生成各个特征的初始值，确保 PP 的初始值合理
            x0 = np.random.rand(len(feature_names))
            pp_index = feature_names.index("PP")
            x0[pp_index] = 0.7  # 初始PP较高

            bounds = [(0, 1)] * len(feature_names)
            bounds[pp_index] = (0.5, 1.0)

            # 目标函数：最小化预测 TS 与目标 TS 之间的差异
            def objective(x):
                x_norm = x / np.sum(x) * 100
                x_scaled = scaler.transform([x_norm])
                pred = model.predict(x_scaled)[0]
                return abs(pred - target_ts)

            # 约束：配方总和为 100
            cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

            result = minimize(objective, x0, bounds=bounds, constraints=cons, method='SLSQP')

            if result.success:
                best_x = result.x / np.sum(result.x) * 100
                pred_ts = model.predict(scaler.transform([best_x]))[0]

                st.success("🎉 成功反推配方！")
                st.metric("预测 TS", f"{pred_ts:.2f} MPa")

                unit_suffix = "wt%" if "质量" in unit_type else "vol%"
                df_result = pd.DataFrame([best_x], columns=feature_names)
                df_result.columns = [f"{col} ({unit_suffix})" for col in df_result.columns]

                st.markdown("### 📋 最优配方参数")
                st.dataframe(df_result.round(2))
            else:
                st.error("❌ 优化失败，请尝试更改目标 TS 或检查模型")
