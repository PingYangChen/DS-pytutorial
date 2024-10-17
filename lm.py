# 匯入必要的套件
import os  # 用於作業系統相關的操作
import pandas as pd  # 用於處理和分析資料的資料框架工具
import numpy as np  # 用於數學計算，特別是陣列處理
from matplotlib import pyplot as plt  # 用於繪製圖表

# 從網路讀取 CSV 檔案並載入至 pandas DataFrame
adv_df = pd.read_csv('https://raw.githubusercontent.com/PingYangChen/DS-pytutorial/refs/heads/main/sample_data/Advertising.csv')
# 顯示資料集的前 5 行
adv_df.head(5)
# 計算資料集的總行數
len(adv_df)
# 儲存資料集中的欄位名稱
adv_var = adv_df.columns
# 顯示 TV、radio 和 newspaper 的統計摘要（不包含第一個欄位）
adv_df[adv_var[1:]].describe()

# 建立圖表，設置大小為12x5英吋
fig = plt.figure(figsize=(12, 5))
# 在第一個子圖中繪製 TV 廣告費用與銷售額的關係圖
ax = plt.subplot(1, 3, 1)
ax.plot(adv_df['TV'], adv_df['sales'], marker='.', linestyle='', color='#A00000')  # 繪製散點圖
ax.set_xlabel("TV", fontsize=16)  # 設置X軸標籤
ax.set_ylabel("sales", fontsize=16)  # 設置Y軸標籤
# 在第二個子圖中繪製 radio 廣告費用與銷售額的關係圖
ax = plt.subplot(1, 3, 2)
ax.plot(adv_df['radio'], adv_df['sales'], marker='.', linestyle='', color='#A00000')
ax.set_xlabel("radio", fontsize=16)
ax.set_ylabel("sales", fontsize=16)
# 在第三個子圖中繪製 newspaper 廣告費用與銷售額的關係圖
ax = plt.subplot(1, 3, 3)
ax.plot(adv_df['newspaper'], adv_df['sales'], marker='.', linestyle='', color='#A00000')
ax.set_xlabel("newspaper", fontsize=16)
ax.set_ylabel("sales", fontsize=16)
# 自動調整子圖間的間距
fig.tight_layout()
# 顯示圖表
fig.show()

# 匯入 statsmodels 用於統計建模
import statsmodels.api as sm

# 將 'sales' 欄位的值存為目標變數 y
y = adv_df['sales'].values

# 將 'TV' 欄位的值存為自變數 x1
x1 = adv_df['TV'].values

# 使用 statsmodels 進行簡單線性迴歸，並建立模型
slr_sm_tmp = sm.OLS(y, sm.add_constant(x1))
slr_sm = slr_sm_tmp.fit()
# 顯示迴歸模型摘要
slr_sm.summary()

# 繪製 Q-Q 圖（Quantile-Quantile plot），用來檢查殘差是否符合常態分佈
fig = sm.qqplot(slr_sm.resid, fit=True, line="45")
fig.show()


# 繪製模型的擬合值與殘差之間的關係圖，用於檢查模型的殘差
fig = plt.figure(figsize=(5, 5))
ax = plt.subplot(1, 1, 1)
ax.plot(slr_sm.fittedvalues, slr_sm.resid, marker='.', linestyle='', color='#1f5ff2') 
ax.set_xlabel("fittedvalues", fontsize=16)  # 設置X軸標籤為擬合值
ax.set_ylabel("resid", fontsize=16)  # 設置Y軸標籤為殘差
fig.tight_layout()
fig.show()

# 進行 Breusch-Pagan 檢定，用來檢查殘差的異質變異性
# 異質變異性指的是殘差的變異是否隨自變數的變化而改變，如果殘差的變異不一致，模型可能不適合。
slr_bptest = sm.stats.diagnostic.het_breuschpagan(slr_sm.resid, sm.add_constant(x1))
# 輸出 Breusch-Pagan 檢定結果
print(slr_bptest)


# 建立多項式迴歸模型，將自變數擴展為 TV 和 TV 的平方項
x1s = np.vstack((x1, x1**2)).transpose()
ploy_sm = sm.OLS(np.sqrt(y), sm.add_constant(x1s)).fit()

# 繪製 Q-Q 圖（Quantile-Quantile plot），用來檢查殘差是否符合常態分佈
fig = sm.qqplot(ploy_sm.resid, fit=True, line="45")
fig.show()

# 繪製多項式模型的擬合值與殘差之間的關係圖
fig = plt.figure(figsize=(5, 5))
ax = plt.subplot(1, 1, 1)
ax.plot(ploy_sm.fittedvalues, ploy_sm.resid, marker='.', linestyle='', color='#1f5ff2') 
ax.set_xlabel("fittedvalues", fontsize=16)  
ax.set_ylabel("resid", fontsize=16)  
fig.tight_layout()
fig.show()



# 進行多元線性迴歸，將 TV、radio 和 newspaper 的廣告費用設為自變數
x = adv_df[['TV', 'radio', 'newspaper']].values
mlr_sm = sm.OLS(y, sm.add_constant(x)).fit()

# 顯示多元線性迴歸模型摘要
mlr_sm.summary()


# 繪製多元線性迴歸模型的 Q-Q 圖來檢查殘差是否符合常態分佈
fig = sm.qqplot(mlr_sm.resid, fit=True, line="45")
fig.show()
# 繪製多元線性迴歸模型的擬合值與殘差的關係圖
fig = plt.figure(figsize=(5, 5))
ax = plt.subplot(1, 1, 1)
ax.plot(mlr_sm.fittedvalues, mlr_sm.resid, marker='.', linestyle='', color='#1f5ff2') 
ax.set_xlabel("fittedvalues", fontsize=16)  
ax.set_ylabel("resid", fontsize=16)  
fig.tight_layout()
fig.show()




# 預測一個新的觀察值，廣告費用分別是 TV: 100000, radio: 20000, newspaper: 1000
x_new = np.array([[1, 100000, 20000, 1000]], dtype=float)

# 使用多元線性迴歸模型進行預測
mlr_sm.predict(x_new)



# 獲取預測的詳細結果，包括信賴區間等
mlr_sm_pred = mlr_sm.get_prediction(x_new)
mlr_sm_pred.summary_frame()



from copy import deepcopy
x2fi = deepcopy(adv_df[['TV', 'radio']])
x2fi['TV:radio'] = x2fi['TV']*x2fi['radio']
lr2fi_sm = sm.OLS(y, sm.add_constant(x2fi)).fit()
lr2fi_sm.summary()




# 僅使用 'TV' 進行迴歸分析，計算 R 平方與調整後的 R 平方
x1 = adv_df['TV']
slr_sm = sm.OLS(y, sm.add_constant(x1)).fit()
slr_sm.rsquared  # R 平方
slr_sm.rsquared_adj  # 調整後的 R 平方

# 使用 'TV' 和 'radio' 進行多元迴歸分析
x1x2 = adv_df[['TV', 'radio']]
mlr2_sm = sm.OLS(y, sm.add_constant(x1x2)).fit()
mlr2_sm.summary()
mlr2_sm.rsquared  # R 平方
mlr2_sm.rsquared_adj  # 調整後的 R 平方

# 使用 TV, radio 和 newspaper 進行多元線性迴歸分析
x = adv_df[['TV', 'radio', 'newspaper']].values
mlr_sm = sm.OLS(y, sm.add_constant(x)).fit()
mlr_sm.rsquared  # R 平方
mlr_sm.rsquared_adj  # 調整後的 R 平方






# 使用 sklearn 進行線性迴歸
from sklearn.linear_model import LinearRegression

# 自變數為 TV, radio 和 newspaper
x = adv_df[['TV', 'radio', 'newspaper']].values

# 目標變數為 sales
y = adv_df['sales'].values

# 建立線性迴歸模型
mdl_sk = LinearRegression(fit_intercept=True)

# 使用 sklearn 進行模型擬合
mdl_sk.fit(x, y)

# 獲取截距項
mdl_sk.intercept_

# 獲取迴歸係數
mdl_sk.coef_

x_new = np.array([[100000, 20000, 1000]], dtype=float)

# 使用訓練數據進行預測
mdl_sk.predict(x_new)



# 匯入必要的套件
import os  # 用於作業系統相關的操作
import pandas as pd  # 用於處理和分析資料的資料框架工具
import numpy as np  # 用於數學計算，特別是陣列處理
from matplotlib import pyplot as plt  # 用於繪製圖表

cre_df = pd.read_csv('https://raw.githubusercontent.com/PingYangChen/DS-pytutorial/refs/heads/main/sample_data/Credit.csv')
cre_df.head(5)
len(cre_df)
cre_var = cre_df.columns
cre_df[cre_var[1:]].describe()

import statsmodels.api as sm

y = cre_df['Balance']

cre_df_dummy = pd.get_dummies(cre_df, columns=['Own'], drop_first=True, dtype=int)

x = cre_df_dummy[['Own_Yes']]

slr_sm = sm.OLS(y, sm.add_constant(x)).fit()
# 顯示迴歸模型摘要
slr_sm.summary()


cre_df_dummy = pd.get_dummies(cre_df, columns=['Region'], drop_first=True, dtype=int)
cre_df_dummy.columns
x = cre_df_dummy[['Region_South', 'Region_West']]

slr_sm = sm.OLS(y, sm.add_constant(x)).fit()
# 顯示迴歸模型摘要
slr_sm.summary()



cre_df_dummy = pd.get_dummies(cre_df, columns=['Student'], drop_first=True, dtype=int)
cre_df_dummy.columns
x = cre_df_dummy[['Income', 'Student_Yes']]

lr_main_sm = sm.OLS(y, sm.add_constant(x)).fit()
# 顯示迴歸模型摘要
lr_main_sm.summary()

from copy import deepcopy
cre_df_dummy = pd.get_dummies(cre_df, columns=['Student'], drop_first=True, dtype=int)
cre_df_dummy.columns
x = deepcopy(cre_df_dummy[['Income', 'Student_Yes']])
x['Income:Student_Yes'] = x['Income']*x['Student_Yes']
lr_twofi_sm = sm.OLS(y, sm.add_constant(x)).fit()
# 顯示迴歸模型摘要
lr_twofi_sm.summary()




import pandas as pd
import statsmodels.api as sm
# 從網路讀取 Default 資料集並載入至 pandas DataFrame
def_df = pd.read_csv('https://raw.githubusercontent.com/PingYangChen/DS-pytutorial/refs/heads/main/sample_data/Default.csv')
# 顯示資料集的總行數
len(def_df)
# 顯示資料集的前 5 行
def_df.head(5)
# 將分類變數 'default' 和 'student' 轉換為虛擬變數（dummy variables），並刪除第一個類別以避免多重共線性
# drop_first=True 表示將刪除第一個類別，dtype=int 將類別轉換為整數
def_df_dummy = pd.get_dummies(def_df, columns=['default', 'student'], drop_first=True, dtype=int)
# 設定目標變數 y 為 default_Yes，表示是否發生違約
y = def_df_dummy['default_Yes']
# 設定自變數 x 為 'balance'，表示信用卡餘額
x = def_df_dummy[['balance']]
# 建立邏輯斯迴歸模型（Logistic Regression），並使用 statsmodels 進行擬合
logit_sm = sm.Logit(y, sm.add_constant(x)).fit()
# 顯示邏輯斯迴歸模型的摘要結果，包含係數估計和模型統計信息
logit_sm.summary()


# 設定自變數 x 為 'student'，表示是否為學生身份
x2 = def_df_dummy[['student_Yes']]
# 建立邏輯斯迴歸模型（Logistic Regression），並使用 statsmodels 進行擬合
logit_sm_2 = sm.Logit(y, sm.add_constant(x2)).fit()
# 顯示邏輯斯迴歸模型的摘要結果，包含係數估計和模型統計信息
logit_sm_2.summary()


x3 = def_df_dummy[['balance', 'income', 'student_Yes']]
# 建立邏輯斯迴歸模型（Logistic Regression），並使用 statsmodels 進行擬合
logit_sm_3 = sm.Logit(y, sm.add_constant(x3)).fit()
# 顯示邏輯斯迴歸模型的摘要結果，包含係數估計和模型統計信息
logit_sm_3.summary()


def_df = pd.read_csv('https://raw.githubusercontent.com/PingYangChen/DS-pytutorial/refs/heads/main/sample_data/hsbdemo.csv')


