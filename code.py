# 基础库导入
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt

# 忽略警告信息，保持输出界面整洁
warnings.filterwarnings('ignore')

# 设置随机种子，保证每次运行的结果一致
SEED = 43
random.seed(SEED)
np.random.seed(SEED)


# Matplotlib 全局参数设置
plt.style.use('bmh')
plt.rcParams['figure.figsize'] = [16, 8]
plt.rcParams['font.size'] = 18

# Pandas 全局显示设置 (显示所有列和行，不换行)
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option('display.expand_frame_repr', False)

sns.set()


try:
    df = pd.read_csv('data.csv')
    print("数据加载成功！")
except FileNotFoundError:
    print("错误：未找到数据文件，请检查文件路径是否正确！")

# 预览前5行数据

df.head()

df.shape

df.describe().T


# 转换时间格式并设为索引
if 'datetime' in df.columns:
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    
df.isnull().sum().sum()

# 1. 剔除违反物理常识的异常值
for col in ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO', 'humidity']:
    df.loc[df[col] < 0, col] = np.nan # 浓度和湿度不能为负
df.loc[df['humidity'] > 100, 'humidity'] = np.nan # 湿度不能大于100

# 2. 时间插值填补刚才挖空的异常值，保证时序连续
df = df.interpolate(method='time')


# 时间列是索引（DatetimeIndex）
df['Hour'] = df.index.hour
df['DayOfWeek'] = df.index.dayofweek
df['Is_Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
df['Month'] = df.index.month

# 供暖季特征 (假设3月15日停暖)
df['Is_Heating'] = df.index.map(lambda x: 1 if (x.month < 3) or (x.month == 3 and x.day <= 15) else 0)

# 时间列是索引（DatetimeIndex）
df['Hour'] = df.index.hour
df['DayOfWeek'] = df.index.dayofweek
df['Is_Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
df['Month'] = df.index.month

# 供暖季特征 (假设3月15日停暖)
df['Is_Heating'] = df.index.map(lambda x: 1 if (x.month < 3) or (x.month == 3 and x.day <= 15) else 0)

# 过去3小时的平均温度和湿度
df['temp_rolling3_mean'] = df['temperature'].rolling(window=3).mean()
df['humid_rolling3_mean'] = df['humidity'].rolling(window=3).mean()

# 过去6小时的最大温差 (辅助判断大气稳定性)
df['temp_rolling6_diff'] = df['temperature'].rolling(window=6).max() - df['temperature'].rolling(window=6).min()

# --- 进阶特征构造  ---
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['Temp_Humid_Interact'] = df['temperature'] * df['humidity']

# 删除因构造滞后特征产生的缺失值
df = df.dropna()

# 【关键】：在 shift 之前，克隆一份纯净的原始数据，专门留给 第7部分画图 和 第9部分提取真实温湿度用
df_raw = df.copy() 



# --- 进阶特征构造  ---
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['Temp_Humid_Interact'] = df['temperature'] * df['humidity']

# 删除因构造滞后特征产生的缺失值
df = df.dropna()

# 【关键】：在 shift 之前，克隆一份纯净的原始数据，专门留给 第7部分画图 和 第9部分提取真实温湿度用
df_raw = df.copy() 

train_df = df[:'2026-03-15']
test_df = df['2026-03-16':]

target_cols = [f'{col}_target' for col in pollutants]
feature_cols = [col for col in df.columns if col not in target_cols]

X_train = train_df[feature_cols].copy()
Y_train = train_df[target_cols].copy()
X_test = test_df[feature_cols].copy()
Y_test = test_df[target_cols].copy()

from sklearn.preprocessing import StandardScaler
# 找出需要标准化的连续变量 (排除时间编码、分类变量)
cols_to_scale = [col for col in feature_cols if col not in ['Hour', 'Hour_sin', 'Hour_cos', 'DayOfWeek', 'Is_Weekend', 'Month', 'Is_Heating']]

scaler = StandardScaler()
X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

print("数据集划分与特征工程重构完成！")

# ======== 中文字体设置（防止图表里的中文变成方块） ========
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] 
plt.rcParams['axes.unicode_minus'] = False

# 需要分析的六种污染物
pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']

from sklearn.preprocessing import StandardScaler

# 1. 我们临时拷贝一份原始数据，专门用来画这张对比图
plot_df = df_raw.copy()

# 2. 仅针对画图，对六种污染物进行 Z-score 标准化
temp_scaler = StandardScaler()
pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
plot_df[pollutants] = temp_scaler.fit_transform(plot_df[pollutants])

# 3. 按小时分组求均值 (使用我们临时标准化的 plot_df)
hourly_mean_norm = plot_df.groupby('Hour')[pollutants].mean()

# 4. 画折线图
plt.figure(figsize=(12, 6))
sns.lineplot(data=hourly_mean_norm, dashes=False, markers=True, linewidth=2)
plt.title('房山良乡地区六种污染物 24 小时平均浓度标准化趋势', fontsize=16)
plt.xlabel('时间 (小时)', fontsize=12)
plt.ylabel('标准化浓度 (Z-Score)', fontsize=12)
plt.xticks(range(0, 24)) # X轴显示 0-23
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='污染物种类', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(3, 2, figsize=(15, 15))
axes = axes.flatten()
hourly_mean_raw = df_raw.groupby('Hour')[pollutants].mean()

colors = ['#d62728', '#8c564b', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']

for i, col in enumerate(pollutants):
    axes[i].plot(hourly_mean_raw.index, hourly_mean_raw[col], marker='o', color=colors[i])
    axes[i].set_title(f'{col} 24小时浓度日变化', fontsize=14)
    axes[i].set_xlabel('小时 (0-23)')
    
    # 根据污染物不同设置不同的纵坐标单位
    if col == 'CO':
        axes[i].set_ylabel('浓度 (mg/m³)')
    else:
        axes[i].set_ylabel('浓度 (μg/m³)')
        
    axes[i].set_xticks(range(0, 24))
    axes[i].grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 8))
# 0代表工作日，1代表周末。我们看NO2和PM2.5的均值对比
weekend_compare = df_raw.groupby('Is_Weekend')[['NO2', 'PM2.5', 'PM10']].mean().reset_index()

# 转换数据格式以适应柱状图绘制
weekend_melted = weekend_compare.melt(id_vars='Is_Weekend', var_name='污染物', value_name='平均浓度')
weekend_melted['Is_Weekend'] = weekend_melted['Is_Weekend'].map({0: '工作日 (周一至周五)', 1: '周末 (周六日)'})

sns.barplot(data=weekend_melted, x='污染物', y='平均浓度', hue='Is_Weekend', palette='Set2')
plt.title('工作日与周末空气污染物平均浓度对比 (周末效应)', fontsize=16)
plt.ylabel('平均浓度 (μg/m³)', fontsize=12)
plt.xlabel('')
plt.legend(title='')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

plt.figure(figsize=(15, 8))
sns.boxplot(data=df_raw, x='Month', y='SO2', palette='pastel')
plt.title('1月至3月 SO2 浓度分布变化 (供暖季末期趋势)', fontsize=16)
plt.xlabel('月份', fontsize=12)
plt.ylabel('SO2 浓度 (μg/m³)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()


plt.figure(figsize=(10, 6))
# 选取需要的列
cols_for_corr = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO', 'temperature', 'humidity']
# 计算斯皮尔曼秩相关系数（由于空气污染数据非正态，Spearman比Pearson更严谨）
corr_matrix = df_raw[cols_for_corr].corr(method='spearman')

# 画热力图 (使用 coolwarm 色系，红正蓝负)
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, 
            square=True, linewidths=.5, cbar_kws={"shrink": .8})
plt.title('污染物与气象条件相关性热力图 (Spearman)', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



## ==========================================
# 气象因素非线性影响散点拟合图 
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ---- 图 1：温度 vs 臭氧 (O3) ----
sns.regplot(ax=axes[0], data=df_raw, x='temperature', y='O3', 
            scatter_kws={'alpha':0.3, 'color':'#1f77b4', 's': 15}, 
            line_kws={'color':'#d62728', 'linewidth': 3}, 
            order=2) 

axes[0].set_title('温度与 O3 浓度的非线性响应关系', fontsize=15)
axes[0].set_xlabel('温度 (℃)', fontsize=12)
axes[0].set_ylabel('O3 浓度 (μg/m³)', fontsize=12)
axes[0].grid(True, linestyle=':', alpha=0.6)

# ---- 图 2：温度 vs 一氧化碳 (CO) ----
sns.regplot(ax=axes[1], data=df_raw, x='temperature', y='CO', 
            scatter_kws={'alpha':0.3, 'color':'#7f7f7f', 's': 15}, 
            line_kws={'color':'#1f77b4', 'linewidth': 3}, 
            order=2) 

axes[1].set_title('温度与 CO 浓度的对流清除效应', fontsize=15)
axes[1].set_xlabel('温度 (℃)', fontsize=12)
axes[1].set_ylabel('CO 浓度 (mg/m³)', fontsize=12)
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()



pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']

# 定义要分析的气象列名 (构造了滞后 1, 2, 3 小时的特征)
# 注意：lag0 就是当前时刻的温湿度
temp_cols = ['temperature', 'temperature_lag1', 'temperature_lag2', 'temperature_lag3']
humid_cols = ['humidity', 'humidity_lag1', 'humidity_lag2', 'humidity_lag3']
x_labels = ['当前(Lag 0)', '滞后1小时', '滞后2小时', '滞后3小时']

# 存储计算结果的字典
temp_corrs = {p: [] for p in pollutants}
humid_corrs = {p: [] for p in pollutants}

# 计算相关系数
for p in pollutants:
    # 临时删除当前计算对中的 NaN 值，保证 Spearman 计算准确
    for t_col in temp_cols:
        valid_data = df_raw[[p, t_col]].dropna()
        corr = valid_data[p].corr(valid_data[t_col], method='spearman')
        temp_corrs[p].append(corr)
        
    for h_col in humid_cols:
        valid_data = df_raw[[p, h_col]].dropna()
        corr = valid_data[p].corr(valid_data[h_col], method='spearman')
        humid_corrs[p].append(corr)


# ==========================================
# 绘图设置
# ==========================================
# 定义一套好看且容易区分的颜色 (Matplotlib tab10)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
markers = ['o', 's', '^', 'D', 'v', '*']

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# ----------------- 图 1: 温度的滞后效应 -----------------
ax1 = axes[0]
for i, p in enumerate(pollutants):
    ax1.plot(x_labels, temp_corrs[p], marker=markers[i], markersize=8, 
             color=colors[i], linewidth=2.5, label=p)

ax1.set_title('温度对各污染物的滞后效应分析 (Spearman相关性)', fontsize=15)
ax1.set_xlabel('温度历史时间点', fontsize=12)
ax1.set_ylabel('相关系数', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)
# 在 0 轴画一条加粗虚线作为基准
ax1.axhline(0, color='black', linewidth=1.5, linestyle='--')
ax1.legend(title='污染物', fontsize=10)

# ----------------- 图 2: 湿度的滞后效应 -----------------
ax2 = axes[1]
for i, p in enumerate(pollutants):
    ax2.plot(x_labels, humid_corrs[p], marker=markers[i], markersize=8, 
             color=colors[i], linewidth=2.5, label=p)

ax2.set_title('相对湿度对各污染物的滞后效应分析 (Spearman相关性)', fontsize=15)
ax2.set_xlabel('相对湿度历史时间点', fontsize=12)
ax2.set_ylabel('相关系数', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.axhline(0, color='black', linewidth=1.5, linestyle='--')
ax2.legend(title='污染物', fontsize=10)

# 调整布局并显示
plt.tight_layout()
plt.show()



import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

print("--- 开始构建第一层：BiLSTM 深度时序网络 ---")

# 1. 深度学习数据形状转换 (样本数, 时间步长, 特征数)
# 由于之前我们将 lag 特征展平了，对于基础的多层感知机/LSTM，我们可以直接作为一个 timestep 输入
X_train_dl = np.array(X_train).reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_dl = np.array(X_test).reshape((X_test.shape[0], 1, X_test.shape[1]))
Y_train_dl = np.array(Y_train)

# 2. 构建 BiLSTM 多输出网络
dl_model = Sequential([
    Input(shape=(X_train_dl.shape[1], X_train_dl.shape[2])),
    Bidirectional(LSTM(64, activation='relu', return_sequences=False)),
    Dropout(0.2), # 防止过拟合
    Dense(32, activation='relu'),
    Dense(6)      # 输出层：对应6种污染物
])

dl_model.compile(optimizer='adam', loss='mse')

# 设置早停法，防止深度学习过拟合
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 训练深度学习模型
history = dl_model.fit(
    X_train_dl, Y_train_dl,
    epochs=100,
    batch_size=32,
    validation_split=0.2, # 划分20%作为验证集
    callbacks=[early_stop],
    verbose=0 # 设置为1可看训练过程，0保持界面整洁
)

print("第一层 BiLSTM 训练完成！正在生成深度特征...")

# 3. 获取第一层深度学习的预测结果 (作为新特征)
train_dl_preds = dl_model.predict(X_train_dl, verbose=0)
test_dl_preds = dl_model.predict(X_test_dl, verbose=0)

# 为 BiLSTM 的预测结果命名
dl_pred_cols = [f'DL_pred_{col}' for col in target_cols]
train_dl_df = pd.DataFrame(train_dl_preds, columns=dl_pred_cols, index=X_train.index)
test_dl_df = pd.DataFrame(test_dl_preds, columns=dl_pred_cols, index=X_test.index)

# 4. 特征融合：将 DL 的预测结果拼接回原始特征集中
X_train_cascade = pd.concat([X_train, train_dl_df], axis=1)
X_test_cascade = pd.concat([X_test, test_dl_df], axis=1)

print("--- 开始构建第二层：XGBoost 级联修正模型 ---")

# 5. 第二层 XGBoost 训练 (使用融合后的增强特征集)
xgb_estimator = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

final_model = MultiOutputRegressor(xgb_estimator)
final_model.fit(X_train_cascade, Y_train)

print("第二层 XGBoost 训练完成！级联架构搭建完毕。")

# 6. 最终预测与评估
Y_pred_final = final_model.predict(X_test_cascade)
Y_pred_final_df = pd.DataFrame(Y_pred_final, columns=target_cols, index=Y_test.index)

eval_results_cascade = []
for i, col in enumerate(target_cols):
    rmse = np.sqrt(mean_squared_error(Y_test[col], Y_pred_final_df[col]))
    mae = mean_absolute_error(Y_test[col], Y_pred_final_df[col])
    r2 = r2_score(Y_test[col], Y_pred_final_df[col])
    eval_results_cascade.append({'污染物': col, 'RMSE': rmse, 'MAE': mae, 'R2': r2})

eval_df_cascade = pd.DataFrame(eval_results_cascade).round(4)
print("\n--- BiLSTM-XGBoost 级联模型 测试集评估结果 ---")
print(eval_df_cascade)


# 截取测试集的前 5 天 (120个小时)
plot_length = 120 
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
axes = axes.flatten()

# 定义一组好看的颜色对 (真实值较浅，预测值较深带有虚线)
color_pairs = [
    ('#1f77b4', '#08306b'), ('#ff7f0e', '#8c2d04'), 
    ('#2ca02c', '#00441b'), ('#d62728', '#67000d'), 
    ('#9467bd', '#3f007d'), ('#8c564b', '#4c2912')
]

for i, col in enumerate(target_cols):
    clean_name = col.replace('_target', '')
    
    # 画真实值
    axes[i].plot(Y_test.index[:plot_length], Y_test[col].iloc[:plot_length], 
                 label='真实观测值 (Ground Truth)', color=color_pairs[i][0], alpha=0.7, linewidth=2.5)
    
    # 画预测值
    axes[i].plot(Y_test.index[:plot_length], Y_pred_final_df[col].iloc[:plot_length], 
                 label='级联模型预测值 (Predicted)', color=color_pairs[i][1], linestyle='--', linewidth=2)
    
    # 未来1小时
    axes[i].set_title(f'未来1小时 {clean_name} 浓度预测追踪曲线 (截取120小时)', fontsize=14)
    
    #  Y 轴标签为真实物理量纲
    if clean_name == 'CO':
        axes[i].set_ylabel('浓度 (mg/m³)', fontsize=12)
    else:
        axes[i].set_ylabel('浓度 (μg/m³)', fontsize=12)
        
    axes[i].legend(loc='upper right')
    axes[i].grid(True, linestyle=':', alpha=0.6)
    
    # 倾斜 X 轴标签防止重叠
    for tick in axes[i].get_xticklabels():
        tick.set_rotation(15)

plt.tight_layout()
plt.show()




cascade_feature_cols = list(X_train.columns) + dl_pred_cols

# 提取 final_model (第二层 XGBoost) 中每一个子模型的特征重要性
importance_dict = {}
for i, col in enumerate(target_cols):
    estimator = final_model.estimators_[i]
    
    clean_col_name = col.replace('_target', '')
    importance_dict[clean_col_name] = estimator.feature_importances_

importance_df = pd.DataFrame(importance_dict, index=cascade_feature_cols)

# 我们取总体贡献度排名前 15 的核心特征 (因为特征变多了，可以多展示几个)
top_features = importance_df.mean(axis=1).sort_values(ascending=False).head(15).index
top_importance = importance_df.loc[top_features]

plt.figure(figsize=(12, 10))
# 画热力图
sns.heatmap(top_importance, cmap='YlGnBu', annot=True, fmt=".3f", 
            linewidths=.5, cbar_kws={'label': '级联模型 特征重要性权重'})
plt.title('级联预测模型：深度先验特征与气象历史驱动重要性矩阵', fontsize=16)
plt.ylabel('模型输入特征 (包含 BiLSTM 提取的深度先验)', fontsize=14)
plt.xlabel('目标预测污染物 (t+1 时刻)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()



import shap

# 初始化 JavaScript 可视化环境
shap.initjs()

# 为了图表美观，我们挑选两个最具代表性的污染物进行 SHAP 分析
# target_cols 顺序为: ['PM2.5_target', 'PM10_target', 'O3_target', 'NO2_target', 'SO2_target', 'CO_target']
# 索引 0 对应 PM2.5, 索引 2 对应 O3

# 1. 提取 PM2.5 和 O3 的 XGBoost 子模型
xgb_pm25 = final_model.estimators_[0]
xgb_o3 = final_model.estimators_[2]

# 2. 构建 SHAP 树解释器
explainer_pm25 = shap.TreeExplainer(xgb_pm25)
explainer_o3 = shap.TreeExplainer(xgb_o3)

# 3. 计算测试集上的 SHAP 值 (取前 500 个样本画图，防止点太多看不清)
X_test_sample = X_test_cascade.head(500)
shap_values_pm25 = explainer_pm25.shap_values(X_test_sample)
shap_values_o3 = explainer_o3.shap_values(X_test_sample)

# 4. 绘图：PM2.5 的 SHAP 蜂群图 (Summary Plot)
plt.figure(figsize=(10, 6))
plt.title("PM2.5 预测模型的 SHAP 协同驱动机制解释", fontsize=15, pad=20)
shap.summary_plot(shap_values_pm25, X_test_sample, max_display=10, show=False)
plt.tight_layout()
plt.show()

# 5. 绘图：O3 的 SHAP 蜂群图
plt.figure(figsize=(10, 6))
plt.title("O3 (臭氧) 预测模型的 SHAP 协同驱动机制解释", fontsize=15, pad=20)
shap.summary_plot(shap_values_o3, X_test_sample, max_display=10, show=False)
plt.tight_layout()
plt.show()



# 注意：虽然我们的目标列叫 PM2.5_target，但为了后续计算 AQI 代码兼容，这里把列名改回原始的污染物名
original_pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']

Y_pred_true_df = pd.DataFrame(
    Y_pred_real,
    columns=original_pollutants,  # 映射回基础列名
    index=X_test.index
)


# (从最开始未归一化的 df_raw 中按索引提取)
final_eval_df = Y_pred_true_df.copy()
final_eval_df['temperature'] = df_raw.loc[X_test.index, 'temperature']
final_eval_df['humidity'] = df_raw.loc[X_test.index, 'humidity']

print("--- 预测完成，直接输出真实浓度与气象数据 (前5行) ---")
print(final_eval_df.head())



# 基于 GB 3095-2012 标准的 AQI 与首要污染物计算

def calculate_iaqi(cp, pollutant):
    """采用分段线性插值计算单项污染物的 IAQI"""
    if cp < 0: cp = 0 # 消除模型极少概率预测出的负值噪声
        
    # 国标浓度限值 (BP_Lo, BP_Hi) 和对应的指数 (IAQI_Lo, IAQI_Hi)
    breakpoints = {
        'PM2.5': [(0,0), (35,50), (75,100), (115,150), (150,200), (250,300), (350,400), (500,500)],
        'PM10':  [(0,0), (50,50), (150,100), (250,150), (350,200), (420,300), (500,400), (600,500)],
        'O3':    [(0,0), (160,50), (200,100), (300,150), (400,200), (800,300), (1000,400), (1200,500)],
        'NO2':   [(0,0), (100,50), (200,100), (700,150), (1200,200), (2340,300), (3090,400), (3840,500)],
        'SO2':   [(0,0), (150,50), (500,100), (650,150), (800,200), (1600,300), (2100,400), (2620,500)],
        'CO':    [(0,0), (5,50), (10,100), (35,150), (60,200), (90,300), (120,400), (150,500)]
    }
    if pollutant not in breakpoints: return 0
    bp_list = breakpoints[pollutant]
    
    for i in range(len(bp_list) - 1):
        bp_lo, iaqi_lo = bp_list[i]
        bp_hi, iaqi_hi = bp_list[i+1]
        if bp_lo <= cp <= bp_hi:
            iaqi = ((iaqi_hi - iaqi_lo) / (bp_hi - bp_lo)) * (cp - bp_lo) + iaqi_lo
            return int(np.ceil(iaqi))
    return bp_list[-1][1] # 爆表按最大值处理

def evaluate_air_quality(df_pred):
    """计算综合AQI、首要污染物并进行等级映射"""
    df_eval = df_pred.copy()
    iaqi_cols = []

    base_pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
    
    for col in base_pollutants:
        iaqi_col = f'IAQI_{col}'
        # 计算单项 IAQI
        df_eval[iaqi_col] = df_eval[col].apply(lambda x: calculate_iaqi(x, col))
        iaqi_cols.append(iaqi_col)
        
    # 计算综合 AQI
    df_eval['AQI'] = df_eval[iaqi_cols].max(axis=1)
    
    # 确定首要污染物
    df_eval['Primary_Pollutant'] = df_eval[iaqi_cols].idxmax(axis=1).str.replace('IAQI_', '')
    df_eval.loc[df_eval['AQI'] <= 50, 'Primary_Pollutant'] = '无'
    
    # 等级划分
    bins = [-1, 50, 100, 150, 200, 300, 9999]
    labels = ['一级(优)', '二级(良)', '三级(轻度污染)', '四级(中度污染)', '五级(重度污染)', '六级(严重污染)']
    df_eval['AQI_Level'] = pd.cut(df_eval['AQI'], bins=bins, labels=labels)
    
    return df_eval

# 执行评价模块
final_eval_df = evaluate_air_quality(final_eval_df)
print("AQI 评价与等级划分完成！")





# 校园健康生活决策生成引擎

def generate_campus_advice(row):
    """基于多维环境感知矩阵，生成校园活动指导建议"""
    aqi = row['AQI']
    primary = row['Primary_Pollutant']
    temp = row['temperature'] 
    humid = row['humidity']   
    
    # 级别 1：重度/严重污染 (红色警戒)
    if aqi > 200:
        return "[红色警戒] 停止一切户外体测和露天体育课！外出需佩戴KN95口罩。宿舍严禁开窗，建议开启净化器。"
        
    # 级别 2：轻度/中度污染 (适度防护，结合温湿机理)
    elif 100 < aqi <= 200:
        if primary in ['PM2.5', 'PM10']:
            if humid > 70:
                return "[高湿雾霾] 颗粒物吸湿膨胀。前往理教/文教上课请佩戴口罩，停止操场高耗氧长跑，早晚避免宿舍开窗。"
            else:
                return "[颗粒物污染] 存在扬尘或轻度霾。建议佩戴口罩通勤，尽量减少大运动量户外社团活动。"
        elif primary in ['NO2', 'SO2', 'CO']:
            if temp < 10:
                return "[低温废气聚集] 逆温抑制对流。请勿在校区外围交通干道（良乡东路）晨跑，防范吸入尾气。"
            else:
                return "[一次污染预警] 局部存在前体物污染。尽量避免在早晚交通高峰期外出散步。"
        elif primary == 'O3':
            return "[臭氧污染] 臭氧超标！减少午后暴露，敏感同学若觉呼吸道刺激请多饮水并留在室内。"
        else:
            return "[轻中度污染] 空气质量欠佳。易感人群请注意防护，减少露天长时间活动。"

    # 级别 3：优/良 (最佳活动或防高温臭氧)
    else: 
        if primary == 'O3' and temp > 28:
            return "[防晒防臭氧] 紫外线强催化臭氧。13:00-16:00 避免在无遮挡的中轴路徒步，体育课建议转至室内馆。"
        elif 15 <= temp <= 25 and 30 <= humid <= 60:
            return "[黄金舒适区] 气象扩散条件极佳！强烈建议在北湖或操场晨跑、体测。宿舍可全天开窗通风换气。"
        elif temp < 5:
            return "[空气清新严寒] 空气优良但气温偏低。外出上课请加衣防寒，预防感冒，适度短时开窗通风。"
        else:
            return "[适宜出行] 空气质量良好。适宜在良乡校区正常安排各类学习、社团及大部分户外运动。"

# 执行决策模块
final_eval_df['Campus_Advice'] = final_eval_df.apply(generate_campus_advice, axis=1)

# 展示最终的预测结果与建议 
pd.set_option('display.max_colwidth', None) # 防止建议被折叠
pd.set_option('display.max_columns', None)  # 显示所有列

# 把 6 种污染物的预测浓度也加进展示列表里
display_cols = [
    'temperature', 'humidity', 
    'PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO', # 预测的具体浓度
    'AQI', 'AQI_Level', 'Primary_Pollutant',   # 评价结果 (含首要污染物)
    'Campus_Advice'                            # 决策建议
]

# 提取需要展示的数据
presentation_df = final_eval_df[display_cols].copy()

# 为了论文表格美观，把气象和浓度数值保留 1 位小数 (AQI是整数，不用保留)
cols_to_round = ['temperature', 'humidity', 'PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
presentation_df[cols_to_round] = presentation_df[cols_to_round].round(1)

print("\n=================== 良乡校区未来时段空气质量预测与生活指导建议 =======================")
# 选取测试集结果中最具代表性的 10 个小时进行展示，按时间排序
print(presentation_df.sample(10, random_state=42).sort_index())
