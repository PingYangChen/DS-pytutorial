
import pandas as pd
import numpy as np
from datetime import datetime
from TaiwanLottery import TaiwanLotteryCrawler

tic = datetime.now()

lotto_list = []
lottery = TaiwanLotteryCrawler()
for i in range(12):
    lotto_list += lottery.lotto649(['2023', '%02d' % (i+1)])
print(lotto_list)

lotto_df = pd.DataFrame(lotto_list)

toc = datetime.now()
elapsed_read_data = (toc - tic).total_seconds()

print("[Process] Read big lotto data: %.2f seconds"% (elapsed_read_data))

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

tic = datetime.now()

lotto_num_list = lotto_df['獎號'].tolist()

all_items = sum(lotto_num_list, [])

uni_items = np.unique(all_items).tolist()

print("There are %d samples in the dataset." % (len(lotto_num_list)))
print("There are %d unique items in the dataset." % (len(uni_items)))

lotto_encoded = pd.DataFrame(columns=uni_items)
for i in range(len(lotto_num_list)):
    lotto_encoded.loc[i] = False
    lotto_encoded.loc[i, lotto_num_list[i]] = True

toc = datetime.now()
elapsed_data_preproc = (toc - tic).total_seconds()

print("[Process] Pre-processed transaction data: %.2f seconds"% (elapsed_data_preproc))

print(lotto_encoded)

tic = datetime.now()
lotto_support_items = apriori(lotto_encoded, min_support=0.001, use_colnames = True)

toc = datetime.now()
elapsed_apriori = (toc - tic).total_seconds()
print("[Process] Executed Apriori algorithm: %.2f seconds"% (elapsed_apriori))
g_big_support_items

tic = datetime.now()
lotto_asso_rules = association_rules(lotto_support_items, metric = 'confidence', min_threshold=0.5)
toc = datetime.now()
elapsed_asso_rule = (toc - tic).total_seconds()
print("[Process] Produced Association Rules: %.2f seconds"% (elapsed_asso_rule))
lotto_asso_rules

