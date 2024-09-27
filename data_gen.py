import os
import pandas as pd
import numpy as np
import datetime 
from matplotlib import pyplot as plt

def gen():
    out=np.zeros(24)
    out[:6]=np.random.uniform(0.1, 6, 6)
    out[6:12]=np.random.uniform(15, 20, 6)
    out[12:14]=np.random.uniform(6, 8, 2)
    out[14:20]=np.random.uniform(15, 20, 6)
    out[20:]=np.random.uniform(0.1, 6, 4)
    return np.round(out, 2)

def genf():
    out=np.zeros(24)
    out[:6]=np.random.uniform(0.1, 6, 6)
    out[6:12]=np.random.uniform(6, 8, 6)
    out[12:14]=np.random.uniform(2, 4, 2)
    out[14:20]=np.random.uniform(6, 8, 6)
    out[20:]=np.random.uniform(0.1, 6, 4)
    return np.round(out, 2)


a=genf()
fig = plt.figure(figsize = (4, 4))
ax = plt.subplot(1, 1, 1)
ax.plot(a)
fig.show()

day0 = datetime.date(2013,5,11)
day1 = datetime.date(2014,7,24)
n = (day1-day0).days+1



alldate = np.linspace(day0, day1, n)

dfColumns=['year', 'month', 'day', 'hour', 'consumption']
df = pd.DataFrame(columns=dfColumns)

for i in range(n):
    #i=0
    dftmp = pd.DataFrame(columns=dfColumns)
    dftmp['year'] = np.ones(24, dtype=int)*alldate[0].year
    dftmp['month'] = np.ones(24, dtype=int)*alldate[0].month
    dftmp['day'] = np.ones(24, dtype=int)*alldate[0].day
    dftmp['hour'] = np.arange(24, dtype=int)
    if i < 330:
        dftmp['consumption'] = gen()
    elif i > 410:
        dftmp['consumption'] = genf()
    else:
        dftmp['consumption'] = gen()
    df = pd.concat([df, dftmp], axis = 0)
    df.reset_index(drop=True, inplace=True)

len(df)

df.to_csv(os.path.join('sample_data', 'vis_hw.csv'), index=False)



