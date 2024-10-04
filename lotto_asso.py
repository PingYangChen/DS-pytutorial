import pandas as pd
import numpy as np
from datetime import datetime
from TaiwanLottery import TaiwanLotteryCrawler  # 從 TaiwanLottery 模組中匯入 TaiwanLotteryCrawler，用於抓取台灣樂透數據

# 記錄當前時間，用於計算讀取數據所花的時間
tic = datetime.now()

# 建立一個空的列表來存儲樂透數據
lotto_list = []
lottery = TaiwanLotteryCrawler()  # 創建一個 TaiwanLotteryCrawler 物件，用於抓取樂透資料
for i in range(12):  # 迴圈 12 次，表示抓取 12 個月份的樂透資料
    lotto_list += lottery.lotto649(['2023', '%02d' % (i+1)])  # 依次抓取 2023 年各月份的樂透號碼，並加入到 lotto_list 列表中
print(lotto_list)  # 輸出抓取到的樂透資料

# 將抓取到的樂透數據轉換為 DataFrame
lotto_df = pd.DataFrame(lotto_list)

# 記錄抓取資料結束的時間
toc = datetime.now()
# 計算抓取資料所花費的時間（以秒為單位）
elapsed_read_data = (toc - tic).total_seconds()

# 輸出抓取資料的處理時間
print("[Process] Read big lotto data: %.2f seconds" % (elapsed_read_data))

from mlxtend.frequent_patterns import apriori  # 從 mlxtend 套件中匯入 apriori 演算法
from mlxtend.frequent_patterns import association_rules  # 匯入生成關聯規則的模組

# 記錄當前時間，用於計算數據預處理時間
tic = datetime.now()

# 從 DataFrame 中提取樂透的獎號，轉換為列表
lotto_num_list = lotto_df['獎號'].tolist()

# 將所有樂透號碼展開為一個列表，並將所有樂透號碼平展合併
all_items = sum(lotto_num_list, [])

# 獲取所有樂透號碼中的唯一值，並轉換為列表
uni_items = np.unique(all_items).tolist()

# 輸出資料集中的樣本數量和唯一樂透號碼的數量
print("There are %d samples in the dataset." % (len(lotto_num_list)))
print("There are %d unique items in the dataset." % (len(uni_items)))

# 創建一個 DataFrame，欄位名稱是唯一的樂透號碼，用來進行 one-hot encoding
lotto_encoded = pd.DataFrame(columns=uni_items)
for i in range(len(lotto_num_list)):
    lotto_encoded.loc[i] = False  # 初始化每一行為 False（表示樂透號碼未中獎）
    lotto_encoded.loc[i, lotto_num_list[i]] = True  # 將中獎的號碼標記為 True

# 記錄數據預處理結束時間
toc = datetime.now()
# 計算數據預處理所花費的時間（以秒為單位）
elapsed_data_preproc = (toc - tic).total_seconds()

# 輸出數據預處理的處理時間
print("[Process] Pre-processed transaction data: %.2f seconds" % (elapsed_data_preproc))

# 輸出經過 one-hot encoding 的數據
print(lotto_encoded)

# 記錄執行 apriori 演算法前的時間
tic = datetime.now()

# 使用 apriori 演算法來挖掘頻繁項目集，最小支持度設定為 0.001
lotto_support_items = apriori(lotto_encoded, min_support=0.001, use_colnames=True)

# 記錄執行 apriori 完成的時間
toc = datetime.now()
# 計算執行 apriori 演算法所花費的時間（以秒為單位）
elapsed_apriori = (toc - tic).total_seconds()

# 輸出執行 apriori 演算法的處理時間
print("[Process] Executed Apriori algorithm: %.2f seconds" % (elapsed_apriori))

# 記錄生成關聯規則前的時間
tic = datetime.now()

# 基於頻繁項目集生成關聯規則，使用信賴度（confidence）作為評估標準，最小信賴度閾值為 0.5
lotto_asso_rules = association_rules(lotto_support_items, metric='confidence', min_threshold=0.5)

# 記錄生成關聯規則完成的時間
toc = datetime.now()
# 計算生成關聯規則所花費的時間（以秒為單位）
elapsed_asso_rule = (toc - tic).total_seconds()

# 輸出生成關聯規則的處理時間
print("[Process] Produced Association Rules: %.2f seconds" % (elapsed_asso_rule))

# 輸出生成的關聯規則
lotto_asso_rules
