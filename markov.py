'''
---------------Imports---------------
'''

import pandas as pd
import numpy as np
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ta.volatility import BollingerBands as BBands
from ta.volatility import AverageTrueRange as ATR
from ta.trend import ADXIndicator as ADX
from ta.volume import VolumeWeightedAveragePrice as VWAP
from ta.momentum import StochRSIIndicator as StochRSI

'''
---------------Data Preparation---------------
'''

df = pd.read_csv("./data/GASCMDUSD.csv")

df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + df['Timestamp'].astype(str), format = '%Y%m%d%H:%M:%S')
df = df.drop(['Date', 'Timestamp'], axis = 1)
df = df[['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']]

bbands = BBands(df['Close'], 20, 2, False)
adx = ADX(df['High'], df['Low'], df['Close'], 14, False)
vwap = VWAP(df['High'], df['Low'], df['Close'], df['Volume'], 20, False)
stochrsi = StochRSI(df['Close'], 14, 3, 3, True)
atr = ATR(df['High'], df['Low'], df['Close'], 14, False)

features = {}

features['DateTime'] = df['DateTime']
features['Volume'] = df['Volume']
features['LogReturn'] = df['Close']/df['Close'].shift(1)

#BB_mavg and VWAP have a corr of 0.999
#features['BB_mavg'] = bbands.bollinger_mavg()

features['BB_width'] = bbands.bollinger_wband()
features['ADX_pos'] = adx.adx_pos()
features['ADX_neg'] = adx.adx_neg()
features['VWAP'] = vwap.volume_weighted_average_price()

#StochRSI d and k have a corr of 0.94
#features['StochRSI_d'] = stochrsi.stochrsi_d()
#features['StochRSI_k'] = stochrsi.stochrsi_k()

features['StochRSI'] = stochrsi.stochrsi()
features['ATR'] = atr.average_true_range()

features = pd.DataFrame(features)[20:]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features.loc[:, features.columns != "DateTime"])
features_scaled = pd.DataFrame(features_scaled, columns = ['Volume', 'LogReturn', 'BB_width', 'ADX_pos', 'ADX_neg', 'VWAP', 'StochRSI', 'ATR'])

train_data, test_data = train_test_split(features_scaled, test_size = 0.2, shuffle=False)

'''
---------------Display---------------
'''

print(df.head())
print(df.dtypes)

plt.plot(features['DateTime'], features['LogReturn'])
plt.xlabel("DateTime")
plt.ylabel("LogReturn")
plt.show()

print(features.head())
print(train_data.head())
print(test_data.head())
