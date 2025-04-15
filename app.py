import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.optimize import minimize

# 页面设置：确保是第一个Streamlit命令
st.set_page_config(page_title="聚丙烯性能预测与逆向设计", layout="wide")

# 页面标题
st.title("聚丙烯拉伸强度岭回归模型：性能预测 与 逆向设计")

# 选择功能
page = st.sidebar.selectbox("🔧 选择功能", ["性能预测", "逆向设计"])

# 加载模型和 scaler
data = joblib.load("model_and_scaler_ts1.pkl")  # 修改为加载 TS 模型
model = data["model"]
scaler = data["scaler"]

# 加载特征名（已删除 TS 列）
df = pd.read_excel("trainrg3.xlsx")
feature_names = df.columns.tolist()

# 保险处理，剔除 TS
if "TS" in feature_names:
    feature_names.remove("TS")

# 单位选择（用户选择）
unit_option = st.sidebar.selectbox("选择配方的单位", ["质量分数 (wt%)", "体积分数 (vol%)", "质量 (g)"])

# 性能预测页面
if page == "性能预测":
    st.subheader("🔬 根据配方预测拉伸强度（TS）")
    
    user_input = {}
    for name in feature_names:
        # 根据用户选择单位显示输入框
        if unit_option == "质量分数 (wt%)":
            user_input[name] = st.number_input(f"{name} (wt%)", value=0.0, step=0.1)
        elif unit_option == "体积分数 (vol%)":
            user_input[name] = st.number_input(f"{name} (vol%)", value=0.0, step=0.1)
        else:
            user_input[name] = st.number_input(f"{name} (g)", value=0.0, step=0.1)
    
    if st.button("开始预测"):
        input_array = np.array([list(user_input.values())])
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        # 预测结果显示 MPa 单位
        st.success(f"预测结果：TS = **{prediction:.3f} MPa**")

# 逆向设计页面
elif page == "逆向设计":
    st.subheader("🎯 逆向设计：根据目标拉伸强度反推配方")

    target_ts = st.number_input("目标 TS 值 (MPa)", value=50.0, step=0.1)  # 修改为目标 TS 单位为 MPa

    if st.button("开始逆向设计"):
        with st.spinner("正在反推出最优配方，请稍候..."):

            # 初始猜测：随机生成各个特征的初始值，确保 PP 的初始值合理
            x0 = np.random.uniform(0, 100, len(feature_names))  # 随机初始化配方比例
            pp_index = feature_names.index("PP")  # 找到 PP 在特征中的索引
            x0[pp_index] = np.random.uniform(70, 100)  # 设置 PP 初始值为 70 到 100 之间的随机值

            # 设置边界，PP 的范围是 70 到 100 之间，其他特征为 0 到 100 之间
            bounds = [(0, 100)] * len(feature_names)
            bounds[pp_index] = (50, 100)  # PP 的比例范围是 50 到 100

            # 目标函数：最小化预测 TS 与目标 TS 之间的差异
            def objective(x):
                # 将配方比例归一化，使其总和为 100
                x_sum = np.sum(x)
                if x_sum != 0:
                    x = x / x_sum * 100  # 归一化

                x_scaled = scaler.transform([x])  # 对配方进行标准化
                pred = model.predict(x_scaled)[0]  # 使用模型预测 TS
                return abs(pred - target_ts)  # 目标是最小化 TS 与目标值的差距

            # 约束：配方总和为 100
            def constraint(x):
                return np.sum(x) - 100  # 配方比例和应该等于 100

            # 将约束加入到优化过程中
            cons = ({'type': 'eq', 'fun': constraint})  # 使用eq约束确保总和为100

            # 执行优化
            result = minimize(objective, x0, bounds=bounds, constraints=cons, method='SLSQP')

            if result.success:
                best_x = result.x
                # 反推的最佳配
