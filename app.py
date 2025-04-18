import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.optimize import minimize
import base64

# Function to convert image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# 设置页面配置（保持原样，图标依然是显示在浏览器标签页中）
image_path = "图片1.png"  # 使用上传的图片路径
icon_base64 = image_to_base64(image_path)  # 转换为 base64

# 设置页面标题和图标
st.set_page_config(page_title="聚丙烯拉伸强度模型", layout="wide", page_icon=f"data:image/png;base64,{icon_base64}")

# 图标原始尺寸：507x158，计算出比例
width = 200  # 设置图标的宽度为100px
height = int(158 * (width / 507))  # 计算保持比例后的高度

# 在页面上插入图标与标题
st.markdown(
    f"""
    <h1 style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{icon_base64}" style="width: {width}px; height: {height}px; margin-right: 15px;" />
        聚丙烯拉伸强度模型：性能预测 与 逆向设计
    </h1>
    """, 
    unsafe_allow_html=True
)

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

        submitted = st.form_submit_button("📊 开始预测")

    if submitted:
        # 判断总和是否满足为100
        if unit_type != "质量 (g)" and abs(total - 100) > 1e-3:
            st.warning("⚠️ 配方加和不为100，无法预测。请确保总和为100后再进行预测。")
        else:
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
