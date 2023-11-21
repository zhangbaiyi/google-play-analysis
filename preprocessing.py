# Google Play Dataset - Preprocessing

import pandas as pd
import numpy as np
import random
import time
import re
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

df = pd.read_csv('Google-Playstore.csv')

random.seed(time.time())

feature_outlook = PrettyTable()
feature_outlook.field_names = ["Feature", "Type", "N/A Count", "N/A Percentage", "Cat/Num", "Unique Count",
                               "Unique Percentage", "Example"]

def install_groupby(value):
    if value < 100:
        return '0-100'
    elif value < 1000:
        return '100-1k'
    elif value < 10000:
        return '1k-10k'
    elif value < 100000:
        return '10k-100k'
    else:
        return '100k+'

def judge_cat_num(col):
    if df[col].dtype == 'float64' or df[col].dtype == 'int64':
        return 'Numerical'
    else:
        return 'Categorical'


def check_na_percentage(col):
    ratio = df[col].isna().sum() / len(df[col])
    return f"{ratio:.2%}"


def check_unique_percentage(col):
    ratio = len(df[col].unique()) / len(df[col])
    return f"{ratio:.2%}"

print("Phase I - Data cleaning")
print("Raw data overview")
for col in df.columns:
    feature_outlook.add_row(
        [col, df[col].dtype, df[col].isna().sum(), check_na_percentage(col), judge_cat_num(col), len(df[col].unique()),
         check_unique_percentage(col), df[col].unique()[random.randint(0, len(df[col].unique()) - 1)]])
print(feature_outlook)

print("Phase I - Data cleaning")
print("NA values percentage")

na_feature_percentage = df.isna().sum().sort_values(ascending=False) / len(df) * 100
na_feature_percentage = na_feature_percentage[na_feature_percentage > 0]
plt.figure(figsize=(10, 5))
na_feature_percentage.plot(kind='barh')
plt.xlabel('Percentage (%)')
plt.ylabel('Features')
plt.title('Percentage of N/A Values')
plt.grid()
plt.show()
df.drop(columns=['Developer Website', 'Developer Email', 'Developer Id', 'Privacy Policy'], inplace=True)
df.drop(columns=['App Id'], inplace=True)
df.drop(columns=['Free'], inplace=True)
df.drop(columns=['Scraped Time'], inplace=True)

print("Phase I - Data duplication")
print("Duplication in Currency column")
currency_values_count = df['Currency'].value_counts()
print("Currency values count")
print(currency_values_count)
currency_pie = currency_values_count.head(1)
currency_pie['Others'] = currency_values_count[1:].sum()
currency_pie.plot(kind='pie', figsize=(10, 10), autopct='%1.1f%%', labels=['USD', 'Others'], ylabel='Currency',
                  title='Currency Distribution')
plt.tight_layout()
plt.show()

# Drop all non-USD currency
df['Currency'] = df['Currency'].apply(lambda x: 'USD' if x == 'USD' else 'Others')
df.drop(df[df['Currency'] == 'Others'].index, inplace=True)
print(df.shape)
df.drop(columns=['Currency'], inplace=True)

print("Phase I - Data duplication")
print("Duplication between 'Installs' and 'Minimum Installs'")
# Test if installs and min-install are the same
installs = df['Installs'].apply(lambda x: x.replace('+', '') if '+' in x else x)
installs = pd.to_numeric(installs.apply(lambda x: x.replace(',', '') if ',' in x else x))
min_installs = df['Minimum Installs']
min_installs = pd.to_numeric(min_installs)
min_installs = min_installs.astype('int64')
print("After conversion, is installs equal to min_installs?")
print(installs.equals(min_installs))
# Drop 'Installs'
df.drop(columns=['Installs'], inplace=True)

print("Phase I - Aggregation")
print("Aggregation of Android versions.")
# Minimum Android Version
# Replace ' and up' and ' - ' in the entire column
df['Minimum Android'] = df['Minimum Android'].str.replace(' and up', '').str.split(' - ').str.get(0)
df['Minimum Android'] = df['Minimum Android'].str.split('.').str.get(0)
# Replace 'Varies with device' with NaN
df['Minimum Android'] = df['Minimum Android'].apply(lambda x: np.nan if x == 'Varies with device' else x)


def classify_size_column(value):
    if pd.isna(value) or value == 'Varies with device':
        return np.nan
    match = re.search(r'([0-9.]+)([kMG]?)', str(value))
    if match:
        number, unit = match.groups()
        if unit == 'k':
            return float(number) / 1024
        elif unit == 'G':
            return float(number) * 1024
        else:
            return float(number)
    else:
        return np.nan


df['Clean Size'] = df['Size'].apply(classify_size_column)
df['Clean Size'].describe()
plt.figure(figsize=(10, 10))
df['Clean Size'].plot(kind='hist', bins=100)
plt.title('Size Distribution')
plt.xlabel('Size (MB)')
plt.ylabel('Count')
plt.show()
df.drop(columns=['Size'], inplace=True)

# Replace date by age (in days)
df['Released'] = pd.to_datetime(df['Released'], format='%b %d, %Y')
df['Last Updated'] = pd.to_datetime(df['Last Updated'], format='%b %d, %Y')
scraped_time = pd.to_datetime('2021-06-15 00:00:00')
df['App Age'] = (scraped_time - df['Released']).dt.days
# Last update age
df['Last Update Age'] = (scraped_time - df['Last Updated']).dt.days
df.drop(columns=['Released', 'Last Updated'], inplace=True)

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
rename_dict = {
    'App Name': 'appName',
    'Category': 'category',
    'Rating': 'rating',
    'Rating Count': 'ratingCount',
    'Maximum Installs': 'installCount',
    'Price': 'priceInUSD',
    'Content Rating': 'contentRating',
    'Ad Supported': 'isAdSupported',
    'In App Purchases': 'isInAppPurchases',
    'Editors Choice': 'isEditorsChoice',
    'Clean Size': 'sizeInMB',
    'Minimum Android': 'minAndroidVersion',
    'Minimum Installs': 'installRange',
    'App Age': 'appAgeInDays',
    'Last Update Age': 'lastUpdateAgeInDays'

}
df.rename(columns=rename_dict, inplace=True)



rfa_X = df.copy()
rfa_y = df['installCount'].copy()
rfa_X.drop(columns=['installCount','installRange'], inplace=True)
rfa_X.drop(columns=['appName'], inplace=True)
rfa_X = pd.get_dummies(rfa_X, columns=['category', 'contentRating'])
rfa_X_train, rfa_X_test, rfa_y_train, rfa_y_test = train_test_split(rfa_X, rfa_y, test_size=0.2, random_state=5805)

rfa = RandomForestRegressor(random_state=5805, max_depth=10)
rfa.fit(rfa_X_train, rfa_y_train)

rfa_y_pred = rfa.predict(rfa_X_test)
features = rfa_X.columns
importances = rfa.feature_importances_
indices = np.argsort(importances)[-14:]
plt.title("Feature Importance - Random Forest")
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.show()
del rfa_X, rfa_y, rfa_X_train, rfa_X_test, rfa_y_train, rfa_y_test, rfa, rfa_y_pred
gc.collect()


pca_X = df.copy()
pca_y = df['installCount'].copy()
pca_X.drop(columns=['installCount','installRange'], inplace=True)
pca_X.drop(columns=['appName'], inplace=True)

pca_columns_to_standardize = ['rating', 'ratingCount', 'priceInUSD', 'sizeInMB', 'appAgeInDays', 'lastUpdateAgeInDays']
pca_X[pca_columns_to_standardize] = StandardScaler().fit_transform(pca_X[pca_columns_to_standardize])

pca_X = pd.get_dummies(pca_X, columns=['category', 'contentRating', 'minAndroidVersion'])
# Replace true/false with 1/0
for col in pca_X.columns:
    if pca_X[col].dtype == 'bool':
        pca_X[col] = pca_X[col].astype(int)


pca = PCA(n_components='mle', svd_solver='full')
pca.fit(pca_X)
pca_X_transform = pca.transform(pca_X)
print("Original shape: {}".format(str(pca_X.shape)))
print("PCA transformed shape: {}".format(str(pca_X_transform.shape)))
print("Original condition number: {:.2f}".format(np.linalg.cond(pca_X)))
print("PCA transformed condition number: {:.2f}".format(np.linalg.cond(pca_X_transform)))
plt.figure(figsize=(10, 10))
plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1, 1), np.cumsum(pca.explained_variance_ratio_))
plt.xticks(np.arange(1, len(pca.explained_variance_ratio_) + 1, 5))
plt.axvline(x=10, color='r', linestyle='--')
plt.axhline(y=0.85, color='b', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA - Cumulative Explained Variance Ratio')
plt.grid()
plt.show()
del pca_X, pca_y, pca, pca_X_transform
gc.collect()

#### SVD Analsyis

svd_X = df.copy()
svd_y = df['installCount'].copy()
svd_X.drop(columns=['installCount', 'installRange'], inplace=True)
svd_X.drop(columns=['appName'], inplace=True)

svd_columns_to_standardize = ['rating', 'ratingCount', 'priceInUSD', 'sizeInMB', 'appAgeInDays', 'lastUpdateAgeInDays']
svd_X[svd_columns_to_standardize] = StandardScaler().fit_transform(svd_X[svd_columns_to_standardize])

svd_X = pd.get_dummies(svd_X, columns=['category', 'contentRating', 'minAndroidVersion'])
# Replace true/false with 1/0
for col in svd_X.columns:
    if svd_X[col].dtype == 'bool':
        svd_X[col] = svd_X[col].astype(int)

svd = TruncatedSVD(n_components=10, n_iter=7, random_state=5805)
svd.fit(svd_X)
svd_X_transform = svd.transform(svd_X)
print("Original shape: {}".format(str(svd_X.shape)))
print("SVD transformed shape: {}".format(str(svd_X_transform.shape)))

print("Original singular values: ", ["{:.2f}".format(val) for val in np.linalg.svd(svd_X, compute_uv=False)])
print("SVD transformed singular values: ",
      ["{:.2f}".format(val) for val in np.linalg.svd(svd_X_transform, compute_uv=False)])

del svd_X, svd_y, svd, svd_X_transform
gc.collect()

##### VIF

vif_X = df.copy()
vif_X.drop(columns=['installCount','installRange'], inplace=True)
vif_X.drop(columns=['appName', 'category','contentRating', 'minAndroidVersion'], inplace=True)
vif_columns_to_standardize = ['rating', 'ratingCount', 'priceInUSD', 'sizeInMB', 'appAgeInDays', 'lastUpdateAgeInDays']
vif_X[vif_columns_to_standardize] = StandardScaler().fit_transform(vif_X[vif_columns_to_standardize])
# Replace true/false with 1/0
for col in vif_X.columns:
    if vif_X[col].dtype == 'bool':
        vif_X[col] = vif_X[col].astype(int)
VIFs = pd.Series([variance_inflation_factor(vif_X, i) for i in range(vif_X.shape[1])], index=vif_X.columns)
print("VIFs: ")
print(VIFs)
del vif_X
gc.collect()

df = pd.get_dummies(df, columns=['category', 'contentRating', 'minAndroidVersion'])
df=df[['ratingCount','sizeInMB','lastUpdateAgeInDays','minAndroidVersion_8','appAgeInDays','rating','category_Productivity', 'category_Social', 'isInAppPurchases', 'minAndroidVersion_7','installCount','installRange']]

for col in df.columns:
    if df[col].dtype == 'bool':
        df[col] = df[col].astype(int)


df['installRange'] = df['installRange'].apply(install_groupby)

#### Srandardization
df_standard = df.copy()

scaler = StandardScaler()
df_standard[['ratingCount','sizeInMB','lastUpdateAgeInDays','appAgeInDays','rating']] = scaler.fit_transform(df_standard[['ratingCount','sizeInMB','lastUpdateAgeInDays','appAgeInDays','rating']])

df_outlier = df_standard.copy()

##### Mahalanobis Distance

df_outlier.drop(columns=['installRange'], inplace=True)
df_outlier['installCount'] = scaler.fit_transform(df_outlier[['installCount']])
mean_vector = df_outlier.mean()
covariance_matrix = df_outlier.cov()
inv_covariance_matrix = np.linalg.inv(covariance_matrix)

mahalanobis_dist = [mahalanobis(df_outlier.iloc[i], mean_vector, inv_covariance_matrix) for i in range(len(df_outlier))]


df_outlier['mahalanobis_dist'] = mahalanobis_dist


significance_level = 0.1  # Adjust as needed
threshold = chi2.ppf((1 - significance_level), df=11)  # df is the number of variables
df_outlier['outlier'] = df_outlier['mahalanobis_dist']**2 > threshold

df_outlier['outlier'].value_counts()

df['outlier'] = df_outlier['outlier']
df_standard['outlier'] = df_outlier['outlier']
df_standard = df_standard[df_standard['outlier']==False]
df = df[df['outlier']==False]
df_standard.drop(columns=['outlier'],inplace=True)
df_standard.reset_index(drop=True,inplace=True)
df.drop(columns=['outlier'],inplace=True)
df.reset_index(drop=True,inplace=True)
del df_outlier
gc.collect()


covariance_matrix = df_standard[['ratingCount','sizeInMB','lastUpdateAgeInDays','appAgeInDays','rating']].cov()
plt.figure(figsize=(12, 10))
sns.heatmap(covariance_matrix, annot=True, fmt=".5f", cmap='coolwarm', linewidths=0.5)
plt.title('Covariance Matrix')
plt.tight_layout()
plt.show()

##### Correlation Matrix
corr_matrix = df_standard[['ratingCount','sizeInMB','lastUpdateAgeInDays','appAgeInDays','rating']].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".5f", cmap='coolwarm', linewidths=0.5)
plt.title('Pearson Correlation Coefficients Matrix')
plt.tight_layout()
plt.show()

value_counts = pd.DataFrame(df['installRange'].value_counts())
total_count = value_counts['count'].sum()
value_counts['percentage'] = value_counts['count'] / total_count * 100
plt.figure(figsize=(10, 5))
plt.barh(value_counts.index, value_counts['percentage'], color='skyblue')
plt.xlabel('Percentage (%)')
plt.ylabel('Range')
plt.title('Target Distribution - Install Range')
plt.grid(axis='x')

# Setting xticks as percentage
xticks = range(0, 51, 10)
plt.xticks(xticks, [f"{x}%" for x in xticks])

# Displaying percentage on each bar
for index, value in enumerate(value_counts['percentage']):
    plt.text(value, index, f"{value:.2f}%", va='center')

plt.show()