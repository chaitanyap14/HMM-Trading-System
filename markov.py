'''
---------------Imports---------------
'''

import pandas as pd
import numpy as np
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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

features['LogReturn'] = df['Close']/df['Close'].shift(1)
features['Direction'] = (features['LogReturn'] >= 1).astype(int)
features = pd.DataFrame(features)[20:]

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features.loc[:, features.columns != "DateTime"])
features_scaled = pd.DataFrame(features_scaled, columns = ['Volume', 'BB_width', 'ADX_pos', 'ADX_neg', 'VWAP', 'StochRSI', 'ATR', 'LogReturn', 'Direction'])

kmeans = KMeans(n_clusters = 12, random_state=42)
features_scaled['States'] = kmeans.fit_predict(features_scaled)

state_table = pd.DataFrame()
state_table['Last_n'] = [x.values.tolist() for x in features_scaled['States'].rolling(4)]
state_table['Last_n'] = state_table['Last_n'].apply(tuple)
state_table['Next_dir'] = features_scaled['Direction'].shift(-1)
state_table = state_table[4:]

state_transition_count = {}
state_transition_count['States'] = state_table['Last_n'].unique()
state_transition_count = pd.DataFrame(state_transition_count)
upcounts_l = []
downcounts_l = []

for i in range(len(state_transition_count)):
    upcount = len(state_table[(state_table['Last_n'] == state_transition_count.iloc[i]['States']) & (state_table['Next_dir'] == 1.0)])
    downcount = len(state_table[(state_table['Last_n'] == state_transition_count.iloc[i]['States']) & (state_table['Next_dir'] == 0.0)])
    upcounts_l.append(upcount)
    downcounts_l.append(downcount)

state_transition_count['Up_count'] = upcounts_l
state_transition_count['Down_count'] = downcounts_l
state_transition_count['Up_prob'] = state_transition_count['Up_count'] / (state_transition_count['Up_count'] + state_transition_count['Down_count'])
state_transition_count['Down_prob'] = state_transition_count['Down_count'] / (state_transition_count['Up_count'] + state_transition_count['Down_count'])

'''
---------------Display---------------
'''

#print(df.head())
#print(df.dtypes)

#plt.plot(features['DateTime'], features['LogReturn'])
#plt.xlabel("DateTime")
#plt.ylabel("LogReturn")
#plt.show()

#print(features_scaled.describe())
#print(features_scaled)
#print(features)
#print(state_table)
print(state_transition_count['Up_prob'].describe())
print(state_transition_count['Down_prob'].describe())
