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
import os

df = pd.read_csv('Google-Playstore.csv')
random.seed(5805)

def install_groupby(value):
    if value < 100:
        return '0-100'
    elif value < 1000:
        return '100-1k'
    elif value < 10000:
        return '1k-10k'
    elif value < 100000:
        return '10k-100k'
    elif value < 1000000:
        return 'Low'
    elif value < 10000000:
        return 'Medium'
    else:
        return 'High'
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

print("=========================================")
print("Raw data overview")
print("Output: table of feature overview")
print("=========================================")
feature_outlook = PrettyTable()
feature_outlook.field_names = ["Feature", "Type", "N/A Count", "N/A Percentage", "Cat/Num", "Unique Count",
                               "Unique Percentage", "Example"]
for col in df.columns:
    feature_outlook.add_row(
        [col, df[col].dtype, df[col].isna().sum(), check_na_percentage(col), judge_cat_num(col), len(df[col].unique()),
         check_unique_percentage(col), df[col].unique()[random.randint(0, len(df[col].unique()) - 1)]])
print(feature_outlook)
del feature_outlook
gc.collect()

print("=========================================")
print("NA Values Percentage")
print("Output: bar chart of NA values percentage")
print("=========================================")

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
del na_feature_percentage
gc.collect()

print("=========================================")
print("Down sampling")
print("Output: shape of the dataset")
print("=========================================")
df = df.query('`Maximum Installs` > 100000')
df = df.query('`Maximum Installs` < 10000000')
print(df.shape)

print("=========================================")
print("Duplication in Currency column")
print("Output: pie chart of currency distribution")
print("=========================================")
currency_values_count = df['Currency'].value_counts()
currency_pie = currency_values_count.head(1)
currency_pie['Others'] = currency_values_count[1:].sum()
currency_pie.plot(kind='pie', autopct='%1.1f%%', labels=['USD', 'Others'], ylabel='Currency',
                  title='Currency Distribution')
plt.show()
del currency_values_count, currency_pie
gc.collect()

# Drop all non-USD currency
df['Currency'] = df['Currency'].apply(lambda x: 'USD' if x == 'USD' else 'Others')
df.drop(df[df['Currency'] == 'Others'].index, inplace=True)
print(df.shape)
df.drop(columns=['Currency'], inplace=True)
print("Shape of Dataframe after modification: {}".format(str(df.shape)))

print("=========================================")
print("Duplication between 'Installs' and 'Minimum Installs'")
print("Output: boolean value")
print("=========================================")
# Test if installs and min-install are the same
installs = df['Installs'].apply(lambda x: x.replace('+', '') if '+' in x else x)
installs = pd.to_numeric(installs.apply(lambda x: x.replace(',', '') if ',' in x else x))
min_installs = df['Minimum Installs']
min_installs = pd.to_numeric(min_installs)
min_installs = min_installs.astype('int64')
print("After conversion, is installs equal to min_installs?")
print(installs.equals(min_installs))
del installs, min_installs
gc.collect()

# Drop 'Installs'
df.drop(columns=['Installs'], inplace=True)
print("Shape of Dataframe after modification: {}".format(str(df.shape)))


print("=========================================")
print("Aggregation of Android versions.")
print("Output: Null")
print("=========================================")
# Replace ' and up' and ' - ' in the entire column
df['Minimum Android'] = df['Minimum Android'].str.replace(' and up', '').str.split(' - ').str.get(0)
df['Minimum Android'] = df['Minimum Android'].str.split('.').str.get(0)
# Replace 'Varies with device' with NaN
df['Minimum Android'] = df['Minimum Android'].apply(lambda x: np.nan if x == 'Varies with device' else x)
print("Shape of Dataframe after modification: {}".format(str(df.shape)))

print("=========================================")
print("Change size unit to MB")
print("Output: Null")
print("=========================================")
df['Clean Size'] = df['Size'].apply(classify_size_column)
df['Clean Size'].describe()
plt.figure(figsize=(10, 10))
df['Clean Size'].plot(kind='hist', bins=100)
plt.title('Size Distribution')
plt.xlabel('Size (MB)')
plt.ylabel('Count')
plt.show()
df.drop(columns=['Size'], inplace=True)
print("Shape of Dataframe after modification: {}".format(str(df.shape)))

print("=========================================")
print("Replace date by age in days")
print("Output: Null")
print("=========================================")
# Replace date by age (in days)
df['Released'] = pd.to_datetime(df['Released'], format='%b %d, %Y')
df['Last Updated'] = pd.to_datetime(df['Last Updated'], format='%b %d, %Y')
scraped_time = pd.to_datetime('2021-06-15 00:00:00')
df['App Age'] = (scraped_time - df['Released']).dt.days
# Last update age
df['Last Update Age'] = (scraped_time - df['Last Updated']).dt.days
df.drop(columns=['Released', 'Last Updated'], inplace=True)
print("Shape of Dataframe after modification: {}".format(str(df.shape)))

print("=========================================")
print("Drop N/A values and duplicates")
print("Output: Null")
print("=========================================")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print("Shape of Dataframe after modification: {}".format(str(df.shape)))

print("=========================================")
print("Rename columns")
print("Output: Null")
print("=========================================")
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
print("Shape of Dataframe after modification: {}".format(str(df.shape)))

print("=========================================")
print("Random Forest Analysis")
print("Output: Feature Importance Plot")
print("=========================================")
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
plt.figure(figsize=(10, 10))
plt.title("Feature Importance - Random Forest")
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.show()
del rfa_X, rfa_y, rfa_X_train, rfa_X_test, rfa_y_train, rfa_y_test, rfa, rfa_y_pred, importances, indices, features
gc.collect()

print("=========================================")
print("Principal Component Analysis")
print("Output: Explained Variance Ratio Plot")
print("=========================================")
pca_X = df.copy()
pca_y = df['installCount'].copy()
pca_X.drop(columns=['installCount','installRange'], inplace=True)
pca_X.drop(columns=['appName'], inplace=True)
pca_columns_to_standardize = ['rating', 'ratingCount', 'priceInUSD', 'sizeInMB', 'appAgeInDays', 'lastUpdateAgeInDays']
pca_X[pca_columns_to_standardize] = StandardScaler().fit_transform(pca_X[pca_columns_to_standardize])

pca_X = pd.get_dummies(pca_X, columns=['category', 'contentRating', 'minAndroidVersion'])
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
del pca_X, pca_y, pca, pca_X_transform, pca_columns_to_standardize
gc.collect()

print("=========================================")
print("Single Value Decomposition")
print("Output: Singular values of original and SVD transformed matrix")
print("=========================================")
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

del svd_X, svd_y, svd, svd_X_transform, svd_columns_to_standardize
gc.collect()

print("=========================================")
print("Variation Inflation Factor")
print("Output: VIFs")
print("=========================================")
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

print("=========================================")
print("One hot encoding")
print("Output: Null")
print("=========================================")
df = pd.get_dummies(df, columns=['category', 'contentRating', 'minAndroidVersion'])
df=df[['ratingCount','sizeInMB','lastUpdateAgeInDays','minAndroidVersion_8','appAgeInDays','rating','category_Productivity', 'category_Social', 'isInAppPurchases', 'minAndroidVersion_7','installCount','installRange']]
for col in df.columns:
    if df[col].dtype == 'bool':
        df[col] = df[col].astype(int)
df['installRange'] = df['installRange'].apply(install_groupby)
print("Shape of Dataframe after modification: {}".format(str(df.shape)))

print("=========================================")
print("Standardization")
print("Output: Null")
print("=========================================")
df_standard = df.copy()
scaler = StandardScaler()
df_standard[['ratingCount','sizeInMB','lastUpdateAgeInDays','appAgeInDays','rating']] = scaler.fit_transform(df_standard[['ratingCount','sizeInMB','lastUpdateAgeInDays','appAgeInDays','rating']])
print("Shape of Dataframe after modification: {}".format(str(df.shape)))

print("=========================================")
print("Outlier Detection")
print("Output: Outlier Detection Result")
print("=========================================")
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot

sns.kdeplot(data=df, x = np.log10(df['installCount'].values), fill=True)
plt.show()
fig = qqplot(np.log10(df['installCount']), stats.norm, fit=True, line='45')
ax = fig.axes[0]
ax.set_title("QQ Plot - Installs (Log) vs. Normal Distribution")
plt.show()
from scipy.stats import shapiro
def shapiro_test(x, title, alpha = 0.05):
    stat, p = shapiro(x)
    # print(f'Shapiro-Wilk test: statistics= {stat:.2f} p-value = {p:.2f}')
    print(f'Shapiro-Wilk test: statistics= {stat:.2f} p-value = {p}')
    print(f'Shapiro-Wilk test: {title} dataset looks normal (fail to reject H0)' if p > alpha else f'Shapiro-Wilk test: {title} dataset does not look normal (reject H0)')

shapiro_test(df['installCount'], 'Before box-cox', 0.01)

df['box_cox_installs'], fitted_lambda = stats.boxcox(df['installCount'])
plt.show()
sns.kdeplot(data=df, x =df['box_cox_installs'], fill=True)
fig = qqplot(df['box_cox_installs'], stats.norm, fit=True, line='45')
ax = fig.axes[0]
ax.set_title("QQ Plot - Installs vs. Normal Distribution")
plt.show()

shapiro_test(df['box_cox_installs'], 'After box-cox', 0.01)
df_standard['box_cox_installs'], _ = stats.boxcox(df_standard['installCount'])
print("=========================================")
print("Covariance Matrix")
print("Output: Covariance Matrix")
print("=========================================")
covariance_matrix = df_standard[['ratingCount','sizeInMB','lastUpdateAgeInDays','appAgeInDays','rating']].cov()
plt.figure(figsize=(12, 10))
sns.heatmap(covariance_matrix, annot=True, fmt=".5f", cmap='coolwarm', linewidths=0.5)
plt.title('Covariance Matrix')
plt.tight_layout()
plt.show()
del covariance_matrix
gc.collect()

print("=========================================")
print("Correlation Matrix")
print("Output: Correlation Matrix")
print("=========================================")
corr_matrix = df_standard[['ratingCount','sizeInMB','lastUpdateAgeInDays','appAgeInDays','rating']].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".5f", cmap='coolwarm', linewidths=0.5)
plt.title('Pearson Correlation Coefficients Matrix')
plt.tight_layout()
plt.show()
del corr_matrix
gc.collect()

print("=========================================")
print("Balanced/Imbalanced Target Distribution")
print("Output: Bar Chart of Target Distribution")
print("=========================================")
value_counts = pd.DataFrame(df['installRange'].value_counts())
total_count = value_counts['count'].sum()
value_counts['percentage'] = value_counts['count'] / total_count * 100
plt.figure(figsize=(10, 5))
plt.barh(value_counts.index, value_counts['percentage'], color='blue')
plt.xlabel('Percentage (%)')
plt.ylabel('Range')
plt.title('Target Distribution - Install Range')
plt.grid(axis='x')
xticks = range(0, 51, 10)
plt.xticks(xticks, [f"{x}%" for x in xticks])
for index, value in enumerate(value_counts['percentage']):
    plt.text(value, index, f"{value:.2f}%", va='center')
plt.show()
del value_counts, total_count
gc.collect()

df['installQcut'] = pd.qcut(df['installCount'], 2, labels=['Low', 'High'])
df_standard['installQcut'] = pd.qcut(df_standard['installCount'], 2, labels=['Low', 'High'])
df['installQcut'].replace({'Low': 0, 'High': 1}, inplace=True)
df_standard['installQcut'].replace({'Low': 0, 'High': 1}, inplace=True)
value_counts = pd.DataFrame(df['installQcut'].value_counts())
total_count = value_counts['count'].sum()
value_counts['percentage'] = value_counts['count'] / total_count * 100
plt.figure(figsize=(10, 5))
plt.barh(value_counts.index, value_counts['percentage'], color='blue')
plt.xlabel('Percentage (%)')
plt.ylabel('Range')
plt.title('Target Distribution - Install Range')
plt.grid(axis='x')
xticks = range(0, 51, 10)
plt.xticks(xticks, [f"{x}%" for x in xticks])
for index, value in enumerate(value_counts['percentage']):
    plt.text(value, index, f"{value:.2f}%", va='center')
plt.show()
del value_counts, total_count
gc.collect()

print("=========================================")
print("Write to CSV")
print("Output: output/preprocessed.csv and output/preprocessed_standard.csv")
print("=========================================")
if not os.path.exists('output'):
    os.makedirs('output')
    if not os.path.exists('output/preprocessed_standard.csv'):
        df_standard.to_csv('output/preprocessed_standard.csv',index=False)
        print("File preprocessed_standard.csv created")
    if not os.path.exists('output/preprocessed.csv'):
        df.to_csv('output/preprocessed.csv',index=False)
        print("File preprocessed.csv created")
else:
    print("Folder output already exists."
          "To re-create the files, please delete the folder and run the script again.")
