import pandas as pd
import numpy as np
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("./GASCMDUSD.csv")

#df['Date'] = pd.to_datetime(df['Date'].astype(str), format = '%Y%m%d')
#df['Timestamp'] = pd.to_datetime(df['Timestamp'], format = '%H:%M:%S')

df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + df['Timestamp'].astype(str), format = '%Y%m%d%H:%M:%S')
df = df.drop(['Date', 'Timestamp'], axis = 1)
df = df[['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']]

print(df.head())
print(df.dtypes)

plt.plot(df['DateTime'], df['Close'])
plt.xlabel("DateTime")
plt.ylabel("Closing Price")
plt.show()
