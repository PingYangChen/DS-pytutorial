import pandas as pd
import numpy as np
from datetime import datetime
# 從 TaiwanLottery 模組中匯入 TaiwanLotteryCrawler，用於抓取台灣樂透數據
from TaiwanLottery import TaiwanLotteryCrawler  
# 從 mlxtend 套件中匯入 apriori 演算法和關聯規則生成方法
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

mset = ['%02d' % (k+1) for k in range(12)]

lotto_list = []
for yyy in ['2023', '2024']:
  for mmm in mset:
    lottery = TaiwanLotteryCrawler()
    result = lottery.lotto649([yyy, mmm])
    lotto_list += result

lotto_df = pd.DataFrame(lotto_list)
lotto_df.sort_values('開獎日期', inplace=True)
print(lotto_df)


lotto_df['獎號'].tolist()

# 創建一個空的 DataFrame，欄位名稱為品項名稱，用來進行 one-hot encoding
lotto_encoded = pd.DataFrame(columns=np.arange(1, 50, dtype=int))
lotto_encoded
# 將每個交易中的購買項目轉換為串列形式
lotto_df_listed = lotto_df['獎號'].tolist()
# 使用 one-hot encoding 轉換每個交易項目
for i in range(len(lotto_df_listed)):
    lotto_encoded.loc[i] = 0                # 將每一列的初始值設為 0
    lotto_encoded.loc[i, lotto_df_listed[i]] = 1 # 將購買的品項標記為 1

# 檢視轉換後的 one-hot encoding 資料
print(lotto_encoded)


# 使用 apriori 演算法來挖掘頻繁項目集，最小支持度為 0.2
Support_items = apriori(lotto_encoded, min_support=0.001, use_colnames=True)
Support_items

# 根據頻繁項目集生成關聯規則，使用 'lift' 來作為評估指標，提升度閾值設為 1
Association_Rules = association_rules(Support_items, metric='confidence', min_threshold=0.4)
# 檢視生成的關聯規則
Association_Rules.sort_values('confidence', ascending=False)