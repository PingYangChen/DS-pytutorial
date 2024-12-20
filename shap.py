# 匯入必要的套件
import os  # 用於執行與作業系統相關的操作，例如檔案路徑管理
from copy import deepcopy  # 用於深層複製物件，確保原始物件不受修改影響
import pandas as pd  # 用於處理與分析資料的資料框架工具
import numpy as np  # 用於數學運算與陣列操作
from matplotlib import pyplot as plt  # 用於繪製數據圖表

# 從指定路徑讀取 CSV 檔案並載入至 pandas DataFrame
att_df = pd.read_csv('D:/attrition.csv')  # 讀取名為 attrition.csv 的資料集
# 顯示資料集的前 5 行，用於檢查資料格式與內容
print(att_df.head(5))
# 計算資料集的總行數，確認資料量
print(len(att_df))

# 篩選出類別型變數（判斷首筆資料的型別是否為字串）
categ_vars = [ele for ele in att_df.columns if type(att_df[ele].loc[0]) == type('is string')]

# 將類別變數轉換為虛擬變數（Dummy Variables），方便後續建模
att_df_dummy = pd.get_dummies(att_df, columns=categ_vars, drop_first=True, dtype=int)
# 查看轉換後資料的欄位名稱
att_df_dummy.columns

# 定義目標變數 y 和特徵變數 x
y = att_df_dummy['Attrition_Yes']  # 目標變數（員工是否離職）
x = att_df_dummy[np.setdiff1d(att_df_dummy.columns, ['Attrition_Yes']).tolist()]  # 排除目標變數後的特徵變數

# 匯入 XGBoost 套件並建立分類模型
import xgboost as xgb
mdl = xgb.XGBClassifier(n_estimators=20)  # 設定樹的數量為 20
mdl.fit(x, y)  # 使用特徵變數與目標變數訓練模型

# 匯入 dalex 套件，便於模型解釋
import dalex
# 建立模型解釋器，指定模型類型為分類（classification）
expl = dalex.Explainer(model=mdl, data=x, y=y, model_type='classification')
expl.model_diagnostics()  # 獲取模型的診斷結果

# 使用 SHAP（SHapley Additive exPlanations）解釋模型結果
shap_result = expl.predict_parts(x.iloc[[0]], type='shap', B=10, random_state=0).result

# 初始化 DataFrame，儲存 SHAP 分數與對應變數資訊
shap_df = pd.DataFrame(columns=['name', 'value', 'mean'])
# 獲取所有變數的名稱
x_vars = np.unique(shap_result['variable_name'])
# 遍歷每個變數並計算其 SHAP 分數的平均值
for j in range(len(x_vars)):
    tmp = deepcopy(shap_result.loc[shap_result['variable_name'] == x_vars[j],])  # 深層複製當前變數的數據
    scores = tmp.loc[tmp['B'] > 0, 'contribution'].values  # 提取貢獻值（僅包含 B > 0 的行）
    x_value = tmp.loc[tmp['B'] == 0, 'variable_value'].values[0]  # 提取變數值
    shap_df.loc[j] = [x_vars[j], x_value] + [np.mean(scores)]  # 將變數名稱、值與平均貢獻值存入 DataFrame

# 計算 SHAP 平均值的絕對值並進行排序
shap_df['abs_mean'] = np.abs(shap_df['mean'])  # 計算貢獻值的絕對平均值
shap_df.sort_values('abs_mean', ascending=False, inplace=True, ignore_index=True)  # 按絕對值降序排序
shap_df.replace(np.nan, None, inplace=True)  # 替換遺漏值為 None

# 建立變數名稱與對應值的表示，方便繪圖標籤
var_name_val = ['%s=%s' % (shap_df['name'].iloc[k], shap_df['value'].iloc[k]) for k in range(len(shap_df))]
shap_draw_order = np.flip(np.arange(len(shap_df)))  # 定義繪圖順序
shap_pos_loc = np.where(shap_df['mean'] >= 0)  # 找出 SHAP 貢獻值為正的位置
shap_neg_loc = np.where(shap_df['mean'] < 0)  # 找出 SHAP 貢獻值為負的位置
shap_ext = np.max(np.abs(shap_df['mean'])) * 1.1  # 定義繪圖的 x 軸範圍（略大於最大值）

# 繪製 SHAP 條形圖
fig = plt.figure(figsize=(8, 6))  # 定義圖形大小
ax = plt.subplot(1, 1, 1)  # 建立子圖
ax.barh(shap_draw_order[shap_pos_loc], shap_df['mean'].iloc[shap_pos_loc].tolist(), color='#f57878')  # 繪製正貢獻值條形圖
ax.barh(shap_draw_order[shap_neg_loc], shap_df['mean'].iloc[shap_neg_loc].tolist(), color='#a0e695')  # 繪製負貢獻值條形圖
ax.vlines(0, np.min(shap_draw_order)-1, np.max(shap_draw_order)+1)  # 繪製中心垂直線
ax.set_yticks(shap_draw_order)  # 設定 y 軸刻度
ax.set_yticklabels(var_name_val)  # 設定 y 軸標籤為變數名稱與值
ax.set_xlim((-shap_ext, shap_ext))  # 設定 x 軸範圍
ax.set_ylim((np.min(shap_draw_order)-.5, np.max(shap_draw_order)+.5))  # 設定 y 軸範圍
fig.tight_layout()  # 自動調整圖形布局
fig.show()  # 顯示圖形
