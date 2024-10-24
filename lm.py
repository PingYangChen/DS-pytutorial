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

adv_df['sales']
adv_df['sales'].values

# 將 'TV' 欄位的值存為自變數 x1
x1 = adv_df['TV']


# 使用 statsmodels 進行簡單線性迴歸，並建立模型
slr_sm = sm.OLS(y, sm.add_constant(x1)).fit()
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
import os  # 用於處理作業系統的操作，例如文件路徑管理
import pandas as pd  # 用於資料處理和分析的資料框架工具
import numpy as np  # 用於數學計算，特別是陣列操作和數據處理
from matplotlib import pyplot as plt  # 用於資料的視覺化繪圖

# 從網路讀取 Credit 資料集並載入至 pandas DataFrame
cre_df = pd.read_csv('https://raw.githubusercontent.com/PingYangChen/DS-pytutorial/refs/heads/main/sample_data/Credit.csv')
# 顯示資料集的前 5 行
cre_df.head(5)
# 計算資料集的總行數
len(cre_df)
# 儲存資料集中的欄位名稱
cre_var = cre_df.columns
# 顯示 Balance、Income、Age 等欄位的統計摘要（不包括第一個欄位 'ID'）
cre_df[cre_var[1:]].describe()

# 匯入 statsmodels 模組用於統計建模
import statsmodels.api as sm

# 設定目標變數 y 為 'Balance' 欄位（信用卡餘額）
y = cre_df['Balance']
# 使用 pd.get_dummies 將分類變數 'Own' 轉換為虛擬變數（dummy variables），只保留一個類別 'Own_Yes'
cre_df_dummy = pd.get_dummies(cre_df, columns=['Own'], drop_first=True, dtype=int)


pd.get_dummies(cre_df, columns=['Own'], drop_first=False, dtype=int)

pd.get_dummies(cre_df, columns=['Own', 'Student', 'Region'], drop_first=True, dtype=int)


# 顯示轉換後資料框中的欄位名稱
cre_df_dummy.columns
# 設定自變數 x 為 'Own_Yes'，表示是否擁有房屋
x = cre_df_dummy[['Own_Yes']]
# 建立線性迴歸模型並進行擬合，使用自變數 'Own_Yes' 預測 'Balance'
slr_sm = sm.OLS(y, sm.add_constant(x)).fit()
# 顯示線性迴歸模型的統計摘要，包含係數估計和模型統計信息
slr_sm.summary()

# 使用 pd.get_dummies 將分類變數 'Region' 轉換為虛擬變數，包含 'Region_South' 和 'Region_West'
cre_df_dummy = pd.get_dummies(cre_df, columns=['Region'], drop_first=True, dtype=int)
# 設定自變數 x 為 'Region_South' 和 'Region_West'，分別表示來自南部和西部的地區
x = cre_df_dummy[['Region_South', 'Region_West']]
# 使用線性迴歸模型來檢驗區域對信用卡餘額的影響
slr_sm = sm.OLS(y, sm.add_constant(x)).fit()
# 顯示區域變數對信用卡餘額影響的迴歸模型統計摘要
slr_sm.summary()

# 使用 pd.get_dummies 將分類變數 'Student' 轉換為虛擬變數，保留 'Student_Yes'
cre_df_dummy = pd.get_dummies(cre_df, columns=['Student'], drop_first=True, dtype=int)
# 設定自變數 x 為 'Income' 和 'Student_Yes'，檢查收入和是否為學生對信用卡餘額的影響
x = cre_df_dummy[['Income', 'Student_Yes']]
# 建立線性迴歸模型，使用收入和是否為學生預測信用卡餘額
lr_main_sm = sm.OLS(y, sm.add_constant(x)).fit()
# 顯示收入和學生身份對信用卡餘額影響的迴歸模型統計摘要
lr_main_sm.summary()

# 匯入 deepcopy 函數，用來做資料的深拷貝
from copy import deepcopy
# 使用 pd.get_dummies 將分類變數 'Student' 轉換為虛擬變數，保留 'Student_Yes'
cre_df_dummy = pd.get_dummies(cre_df, columns=['Student'], drop_first=True, dtype=int)
# 顯示轉換後的資料框中的欄位名稱
cre_df_dummy.columns
# 使用 deepcopy 將 'Income' 和 'Student_Yes' 拷貝到新的資料框 x，確保修改 x 時不會影響原資料框
x = deepcopy(cre_df_dummy[['Income', 'Student_Yes']])
# 新增一個交互作用項 'Income:Student_Yes'，表示收入和學生身份之間的交互作用
x['Income:Student_Yes'] = x['Income'] * x['Student_Yes']
# 建立線性迴歸模型，包含收入、學生身份和它們的交互作用項
lr_twofi_sm = sm.OLS(y, sm.add_constant(x)).fit()
# 顯示收入、學生身份及其交互作用對信用卡餘額影響的迴歸模型統計摘要
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

import pandas as pd  # 用於處理資料的資料框架工具
import numpy as np  # 用於進行數學運算和數據操作
from sklearn.linear_model import LogisticRegression  # 用於執行邏輯斯迴歸的機器學習模型
def_df_dummy = pd.get_dummies(def_df, columns=['default', 'student'], drop_first=True, dtype=int)
# 設定目標變數 y 為 default_Yes，表示是否發生違約
y = def_df_dummy['default_Yes'].values
x = def_df_dummy[['balance', 'income', 'student_Yes']].values
# 建立 LogisticRegression 模型，無懲罰項（penalty=None），使用截距項，最多進行 1000 次迭代以保證收斂
logit_sk = LogisticRegression(penalty=None, fit_intercept=True, max_iter=1000)
# 使用自變數 x 和目標變數 y 來訓練邏輯斯迴歸模型
logit_sk.fit(x, y)
# 輸出邏輯斯迴歸模型的截距項，這是模型的常數項
print(logit_sk.intercept_)
# 輸出邏輯斯迴歸模型的迴歸係數，這些係數表示每個自變數對目標變數的影響
print(logit_sk.coef_)



import pandas as pd  # 用於處理資料的資料框架工具
import numpy as np  # 用於進行數學運算和數據操作
from sklearn.linear_model import LogisticRegression  # 用於執行邏輯斯迴歸的機器學習模型
# 從網路讀取 hsb demo 資料集並載入至 pandas DataFrame
hsb_df = pd.read_csv('https://raw.githubusercontent.com/PingYangChen/DS-pytutorial/refs/heads/main/sample_data/hsbdemo.csv')

pd.crosstab(index=hsb_df['ses'], columns=hsb_df['prog'])

# 使用 pd.get_dummies 將分類變數 'prog' 和 'ses' 轉換為虛擬變數（dummy variables）
# drop_first=False 表示不刪除任何類別，以便保留所有的分類變數
hsb_df_dummy = pd.get_dummies(hsb_df, columns=['prog', 'ses'], drop_first=False, dtype=int)
# 設定目標變數 y 為 'prog' 欄位，這個欄位包含學生的課程類型
y = hsb_df['prog'].values
# 設定自變數 x 為虛擬變數 'ses_middle', 'ses_high'（社會經濟地位），以及 'write'（寫作成績）
x = hsb_df_dummy[['ses_middle', 'ses_high', 'write']].values
# 建立 LogisticRegression 模型，無懲罰項（penalty=None），使用截距項，最多進行 1000 次迭代以保證收斂
logit_sk = LogisticRegression(penalty=None, fit_intercept=True, max_iter=1000)
# 使用自變數 x 和目標變數 y 來訓練邏輯斯迴歸模型
logit_sk.fit(x, y)
# 列出邏輯斯迴歸模型的目標類別，顯示模型預測的分類
print(logit_sk.classes_)
# 輸出邏輯斯迴歸模型的截距項，這是模型的常數項
print(logit_sk.intercept_)
# 輸出邏輯斯迴歸模型的迴歸係數，這些係數表示每個自變數對目標變數的影響
print(logit_sk.coef_)



