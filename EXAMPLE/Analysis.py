import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import os
from functools import reduce
from statsmodels.tsa.stattools import coint


sns.set(style='white')

# Retrieve intraday price data and combine them into a DataFrame.
# 1. Load downloaded prices from folder into a list of dataframes.
#folder_path = 'STATICS/PRICE'
folder_path = '../STATICS/S&P500Top20'

curr_sector = 'InformationTechnology'
#curr_sector = 'HealthCare'
folder_path = '../STATICS/S&P500/' + curr_sector

file_names  = os.listdir(folder_path)
tickers     = [name.split('.')[0] for name in file_names]
#df_list     = [pd.read_csv(os.path.join('STATICS/PRICE', name)) for name in file_names]
df_list     = [pd.read_csv(os.path.join(folder_path, name)) for name in file_names]
#df_list = df_list[0:50]
# 2. Replace the closing price column name by the ticker.
for i in range(len(df_list)):
    df_list[i].rename(columns={'close': tickers[i]}, inplace=True)

# 3. Merge all price dataframes. Extract roughly the first 70% data.
df  = reduce(lambda x, y: pd.merge(x, y, on='date'), df_list)
idx = round(len(df) * 0.7)
df  = df.iloc[:idx, :]

# Calculate and plot price correlations.
#pearson_corr  = df[tickers].corr()
#sns.clustermap(pearson_corr).fig.suptitle('Pearson Correlations')
#sns.clustermap(pearson_corr, vmin=-1, vmax=1, xticklabels=True, yticklabels=True).fig.suptitle('Pearson Correlations')

# https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
col_corr = set() # Set of all the names of deleted columns
def correlation(dataset, threshold):
    #col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            #if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
            if (abs(corr_matrix.iloc[i, j]) <= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset
    return dataset
threshold = 0.7
df = correlation(df, threshold)
print(len(df.columns))
pearson_corr  = df.corr()

cg = sns.clustermap(pearson_corr, vmin=threshold, vmax=1, xticklabels=True, yticklabels=True, figsize=(10,10))
#cg.ax_row_dendrogram.set_visible(False)
#cg.cax.set_visible(False)


plt.tight_layout()

# Plot the marginal distributions.

sns.set(style='darkgrid')
#sns.jointplot(df['CRM'], df['FIS'],  kind='hex', color='#2874A6')
#sns.jointplot(df['IBM'],  df['CTSH'], kind='hex', color='#2C3E50')
# 'IQV', 'SYK' 
#selected_pairs = ['IQV', 'SYK']
selected_pairs = ['V', 'MA']
sns.jointplot(df[selected_pairs[0]], df[selected_pairs[1]],  kind='hex', color='#2874A6')


# plt.show()

# Calculate the p-value of cointegration test for JNJ-PG and KO-PEP pairs.
x = df['CRM']
y = df['FIS']
_, p_value, _ = coint(x, y)
print('The p_value of CRM-FIS pair cointegration is: {}'.format(p_value))

x = df['IBM']
y = df['CTSH']
_, p_value, _ = coint(x, y)
print('The p_value of IBM-CTSH pair cointegration is: {}'.format(p_value))


# Plot the linear relationship of the CRM-FIS pair.
df2 = df[['CRM', 'FIS']].copy()
spread = df2['CRM'] - df2['FIS']
mean_spread = spread.mean()
df2['Dev'] = spread - mean_spread
rnd = np.random.choice(len(df), size=500)
sns.scatterplot(x='CRM', y='FIS', hue='Dev', linewidth=0.3, alpha=0.8,
                data=df2.iloc[rnd, :]).set_title('JNJ-PG Price Relationship')

# Plot the linear relationship of the IBM-CTSH pair.
df2 = df[['IBM', 'CTSH']].copy()
spread = df2['IBM'] - df2['CTSH']
mean_spread = spread.mean()
df2['Dev'] = spread - mean_spread
rnd = np.random.choice(len(df), size=500)
sns.scatterplot(x='IBM', y='CTSH', hue='Dev', linewidth=0.3, alpha=0.8,
                data=df2.iloc[rnd, :]).set_title('KO-PEP Price Relationship')



# Plot the historical JNJ-PG prices and the spreads for a sample period.
def plot_spread(df, ticker1, ticker2, idx, th, stop):

    px1 = df[ticker1].iloc[idx] / df[ticker1].iloc[idx[0]]
    px2 = df[ticker2].iloc[idx] / df[ticker2].iloc[idx[0]]

    sns.set(style='white')

    # Set plotting figure
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})

    # Plot the 1st subplot
    sns.lineplot(data=[px1, px2], linewidth=1.2, ax=ax[0])
    ax[0].legend(loc='upper left')

    # Calculate the spread and other thresholds
    spread = df[ticker1].iloc[idx] - df[ticker2].iloc[idx]
    mean_spread = spread.mean()
    sell_th     = mean_spread + th
    buy_th      = mean_spread - th
    sell_stop   = mean_spread + stop
    buy_stop    = mean_spread - stop

    # Plot the 2nd subplot
    sns.lineplot(data=spread, color='#85929E', ax=ax[1], linewidth=1.2)
    ax[1].axhline(sell_th,   color='b', ls='--', linewidth=1, label='sell_th')
    ax[1].axhline(buy_th,    color='r', ls='--', linewidth=1, label='buy_th')
    ax[1].axhline(sell_stop, color='g', ls='--', linewidth=1, label='sell_stop')
    ax[1].axhline(buy_stop,  color='y', ls='--', linewidth=1, label='buy_stop')
    ax[1].fill_between(idx, sell_th, buy_th, facecolors='r', alpha=0.3)
    ax[1].legend(loc='upper left', labels=['Spread', 'sell_th', 'buy_th', 'sell_stop', 'buy_stop'], prop={'size':6.5})

idx = range(11000, 12000)
plot_spread(df, 'CRM', 'FIS', idx, 0.5, 1)

idx = range(13000, 14000)
plot_spread(df, 'IBM', 'CTSH', idx, 0.5, 1)


# Generate correlated time-series.
# 1. Simulate 1000 correlated random variables by Cholesky Decomposition.
corr = np.array([[1.0, 0.9],
                 [0.9, 1.0]])
L = scipy.linalg.cholesky(corr)
rnd = np.random.normal(0, 1, size=(1000, 2))
out = rnd @ L

# 2. Simulate GBM returns and prices.
dt = 1/252
base1 = 110; mu1 = 0.03; sigma1 = 0.05
base2 = 80;  mu2 = 0.01; sigma2 = 0.03
ret1  = np.exp((mu1 - 0.5 * (sigma1 ** 2) ) * dt + sigma1 * out[:, 0] * np.sqrt(dt))
ret2  = np.exp((mu2 - 0.5 * (sigma2 ** 2) ) * dt + sigma2 * out[:, 1] * np.sqrt(dt))

price1 = base1 * np.cumprod(ret1)
price2 = base2 * np.cumprod(ret2)

# 3. Calculate the return correlation and the p-value for cointegration testing.
corr_ret , _   = scipy.stats.pearsonr(ret1, ret2)
corr_price , _ = scipy.stats.pearsonr(price1, price2)
_, p_value, _  = coint(price1, price2)
print('GBM simulation result - return correlation: {}'.format(corr_ret))
print('GBM simulation result - price correlation: {}'.format(corr_price))
print('GBM simulation result - p-value for cointegration testing: {}'.format(p_value))

# 4. Plot the results.
df_gbm = pd.DataFrame()
df_gbm['price1'] = price1
df_gbm['price2'] = price2
idx = range(1000)
plot_spread(df_gbm, 'price1', 'price2', idx, 0.5, 1)

plt.show()