# Google Play Dataset - Preprocessing
import gc

import numpy as np
import pandas as pd
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("=======================================")
print("Pre-processing of dataset")
print("Output: first 5 rows of the result dataset")
print("=======================================")
df = pd.read_csv('Google-Playstore.csv')

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

def install_groupby(value):
    if value < 100:
        return '0-100'
    elif value < 1000:
        return '100-1k'
    elif value < 10000:
        return 'Low'
    elif value < 100000:
        return 'Medium'
    else:
        return 'High'

df.drop(['App Id','Developer Id', 'Developer Website', 'Developer Email', 'Privacy Policy', 'Scraped Time'], axis=1, inplace=True)
df = df.query('`Maximum Installs` > 100000')
df = df.query('`Maximum Installs` < 10000000')
df['Currency'] = df['Currency'].apply(lambda x: 'USD' if x == 'USD' else 'Others')
df.drop(df[df['Currency'] == 'Others'].index, inplace=True)
df.drop(columns=['Currency'], inplace=True)
df.drop(columns=['Installs'], inplace=True)
df['Name Length'] = df['App Name'].str.len()
df['Minimum Android'] = df['Minimum Android'].str.replace(' and up', '').str.split(' - ').str.get(0)
df['Minimum Android'] = df['Minimum Android'].str.split('.').str.get(0)
# Replace 'Varies with device' with NaN
df['Minimum Android'] = df['Minimum Android'].apply(lambda x: np.nan if x == 'Varies with device' else x)

df['Clean Size'] = df['Size'].apply(classify_size_column)
df['Clean Size'].describe()
df['Released'] = pd.to_datetime(df['Released'], format='%b %d, %Y')
df['Last Updated'] = pd.to_datetime(df['Last Updated'], format='%b %d, %Y')
scraped_time = pd.to_datetime('2021-06-15 00:00:00')
df['App Age'] = (scraped_time - df['Released']).dt.days
df['Last Update Age'] = (scraped_time - df['Last Updated']).dt.days
df['Minimum Installs'] = df['Minimum Installs'].apply(install_groupby)

df.drop(columns=['Released', 'Last Updated'], inplace=True)
df.drop(['Size'], axis=1, inplace=True)
df.drop(['App Name'], axis=1, inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.shape)
print(df.head(5))

import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot

sns.kdeplot(data=df, x = np.log10(df['Maximum Installs'].values), fill=True)
plt.show()
fig = qqplot(np.log10(df['Maximum Installs']), stats.norm, fit=True, line='45')
ax = fig.axes[0]
ax.set_title("QQ Plot - Installs vs. Normal Distribution")
plt.show()
from scipy.stats import shapiro
def shapiro_test(x, title, alpha = 0.05):
    stat, p = shapiro(x)
    # print(f'Shapiro-Wilk test: statistics= {stat:.2f} p-value = {p:.2f}')
    print(f'Shapiro-Wilk test: statistics= {stat:.2f} p-value = {p}')
    print(f'Shapiro-Wilk test: {title} dataset looks normal (fail to reject H0)' if p > alpha else f'Shapiro-Wilk test: {title} dataset does not look normal (reject H0)')

shapiro_test(df['Maximum Installs'], 'Before box-cox', 0.01)


df['box_cox_installs'], fitted_lambda = stats.boxcox(df['Maximum Installs'])
plt.show()
sns.kdeplot(data=df, x =df['box_cox_installs'], fill=True)
fig = qqplot(df['box_cox_installs'], stats.norm, fit=True, line='45')
ax = fig.axes[0]
ax.set_title("QQ Plot - Installs vs. Normal Distribution")
plt.show()

shapiro_test(df['box_cox_installs'], 'After box-cox', 0.01)

rename_dict = {
    'box_cox_installs': 'Installs Transformed',
    'Clean Size': 'Size',
    'Minimum Installs': 'Install Range',
    'Maximum Installs': 'Installs'
}
df.rename(columns=rename_dict, inplace=True)

print("=========================================")
print("Principal Component Analysis")
print("Output: Explained Variance Ratio Plot")
print("=========================================")
pca_X = df.copy()
pca_y = df['Installs', 'Install Range', 'Installs Transformed'].copy()
pca_X.drop(['Installs', 'Install Range', 'Installs Transformed'], axis=1, inplace=True)
pca_columns_to_standardize = ['Rating', 'Rating Count', 'Price', 'Size', 'App Age', 'Last Update Age']
pca_X[pca_columns_to_standardize] = StandardScaler().fit_transform(pca_X[pca_columns_to_standardize])
pca_X = pd.get_dummies(pca_X, columns=['Category', 'Content Rating', 'Minimum Android'])
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
print("Covariance Matrix")
print("Output: Covariance Matrix")
print("=========================================")
covariance_matrix = df[['Rating','Size','Last Update Age','App Age','Rating Count', 'Price', 'Name Length']].cov()
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
corr_matrix = df[['Rating','Size','Last Update Age','App Age','Rating Count', 'Price', 'Name Length']].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".5f", cmap='coolwarm', linewidths=0.5)
plt.title('Pearson Correlation Coefficients Matrix')
plt.tight_layout()
plt.show()
del corr_matrix
gc.collect()
