# Google Play Dataset - Preprocessing
import gc

import numpy as np
import pandas as pd
from prettytable import PrettyTable

pd.set_option('mode.chained_assignment', None)
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro

font_title = {'family': 'serif', 'color': 'blue', 'size': '16'}
font_label = {'family': 'serif', 'color': 'darkred', 'size': '12'}


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


def shapiro_test(x, title, alpha=0.05):
    shapiro_test_result = PrettyTable()
    shapiro_test_result.field_names = ['Statistic', 'p-value', 'Result']
    shapiro_test_result.float_format = '.2'
    stat, p = shapiro(x)
    print(f'Shapiro-Wilk test: statistics= {stat:.2f} p-value = {p}')
    print(
        f'Shapiro-Wilk test: {title} dataset looks normal (fail to reject H0)' if p > alpha else f'Shapiro-Wilk test: {title} dataset does not look normal (reject H0)')
    shapiro_test_result.add_row([stat, p, 'Looks normal' if p > alpha else 'Does not look normal'])
    return shapiro_test_result



# del pca_X, pca_y, pca, pca_X_transform, pca_columns_to_standardize
# gc.collect()
#
# print("=========================================")
# print("Covariance Matrix")
# print("Output: Covariance Matrix")
# print("=========================================")
# covariance_matrix = df[['Rating', 'Size', 'Last Update Age', 'App Age', 'Rating Count', 'Price', 'Name Length']].cov()
# plt.figure(figsize=(12, 10))
# sns.heatmap(covariance_matrix, annot=True, fmt=".5f", cmap='coolwarm', linewidths=0.5)
# plt.title('Covariance Matrix')
# plt.tight_layout()
# plt.show()
# del covariance_matrix
# gc.collect()
#
# print("=========================================")
# print("Correlation Matrix")
# print("Output: Correlation Matrix")
# print("=========================================")
# corr_matrix = df[['Rating', 'Size', 'Last Update Age', 'App Age', 'Rating Count', 'Price', 'Name Length']].corr()
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, annot=True, fmt=".5f", cmap='coolwarm', linewidths=0.5)
# plt.title('Pearson Correlation Coefficients Matrix')
# plt.tight_layout()
# plt.show()
# del corr_matrix
# gc.collect()


def preprocessing():
    print("=======================================")
    print("Pre-processing of dataset")
    print("Output: first 5 rows of the result dataset")
    print("=======================================")
    _df = pd.read_csv('Google-Playstore.csv')

    _df.drop(['App Id', 'Developer Id', 'Developer Website', 'Developer Email', 'Privacy Policy', 'Scraped Time'],
             axis=1, inplace=True)
    # _df = _df.query('`Maximum Installs` > 100000')
    # _df = _df.query('`Maximum Installs` < 10000000')
    _df['Currency'] = _df['Currency'].apply(lambda x: 'USD' if x == 'USD' else 'Others')
    _df.drop(_df[_df['Currency'] == 'Others'].index, inplace=True)
    _df.drop(columns=['Currency'], inplace=True)
    _df.drop(columns=['Installs'], inplace=True)
    _df['Name Length'] = _df['App Name'].str.len()
    _df['Minimum Android'] = _df['Minimum Android'].str.replace(' and up', '').str.split(' - ').str.get(0)
    _df['Minimum Android'] = _df['Minimum Android'].str.split('.').str.get(0)
    # Replace 'Varies with device' with NaN
    _df['Minimum Android'] = _df['Minimum Android'].apply(lambda x: np.nan if x == 'Varies with device' else x)

    _df['Clean Size'] = _df['Size'].apply(classify_size_column)
    _df['Clean Size'].describe()
    _df['Released'] = pd.to_datetime(_df['Released'], format='%b %d, %Y')
    _df['Last Updated'] = pd.to_datetime(_df['Last Updated'], format='%b %d, %Y')
    scraped_time = pd.to_datetime('2021-06-15 00:00:00')
    _df['App Age'] = (scraped_time - _df['Released']).dt.days
    _df['Last Update Age'] = (scraped_time - _df['Last Updated']).dt.days
    _df['Minimum Installs'] = _df['Minimum Installs'].apply(install_groupby)

    _df.drop(columns=['Released', 'Last Updated'], inplace=True)
    _df.drop(['Size'], axis=1, inplace=True)
    _df.drop(['App Name'], axis=1, inplace=True)
    _df['Rating'] = _df['Rating'].apply(lambda x: np.nan if x == 0 else x)
    _df['Rating Count'] = _df['Rating Count'].apply(lambda x: np.nan if x == 0 else x)
    _df['Maximum Installs'] = _df['Maximum Installs'].apply(lambda x: np.nan if x == 0 else x)
    _df['App Age'] = _df['App Age'].apply(lambda x: np.nan if x <= 0 else x)
    _df['Last Update Age'] = _df['Last Update Age'].apply(lambda x: np.nan if x <= 0 else x)
    _df.dropna(inplace=True)
    _df.drop_duplicates(inplace=True)
    _df.drop(columns='Minimum Installs', inplace=True)
    rename_dict = {
        'Clean Size': 'Size in MB',
        'Maximum Installs': 'Installs'
    }
    _df.reset_index(drop=True, inplace=True)
    _df.rename(columns=rename_dict, inplace=True)
    print(_df.shape)
    print(_df.head(5))
    return _df

def outlier_detect(df):
    df_numeric = df.select_dtypes(include=np.number).copy()

    # KDE Subplots for Numeric Features
    fig, axes = plt.subplots(4, 2)
    fig.suptitle('KDE Plot for Numeric Features', fontsize=16, fontfamily='serif')
    fig.supylabel('Density', fontsize=12, fontfamily='serif')
    axes = axes.flatten()
    for i, col in enumerate(df_numeric.columns):
        sns.kdeplot(data=df_numeric, x=col, fill=True, ax=axes[i], alpha=0.6, linewidth=0.5)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('')
    plt.tight_layout()
    plt.show()

    # KDE Subplots for Numeric Features - Log10
    fig, axes = plt.subplots(4, 2)
    fig.supylabel('Density', fontsize=12, fontfamily='serif')
    axes = axes.flatten()
    for i, col in enumerate(df_numeric.columns):
        sns.kdeplot(data=df_numeric, x=np.log10(df_numeric[col].values), fill=True, ax=axes[i],
                    alpha=0.6, linewidth=0.5)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('')
    fig.suptitle('KDE Plot for Numeric Features - Logarithmic', fontsize=16, fontfamily='serif')
    plt.tight_layout()
    plt.show()

    # Multivariate Box Plot
    fig, ax = plt.subplots()
    df_numeric_log10 = df_numeric.drop(columns=['Price']).copy()
    df_numeric_log10.replace(0, np.nan, inplace=True)
    df_numeric_log10.dropna(inplace=True)
    df_numeric_log10 = np.log10(df_numeric_log10)
    df_numeric_log10_standard = StandardScaler().fit_transform(df_numeric_log10)
    df_numeric_log10_standard = pd.DataFrame(df_numeric_log10_standard, columns=df_numeric_log10.columns)
    ax.boxplot(df_numeric_log10_standard)
    ax.set_xticklabels(df_numeric_log10_standard.columns, rotation=45)
    plt.title('Multivariate Box Plot', fontdict=font_title)
    plt.xlabel('Features', fontdict=font_label)
    plt.ylabel('Standardized Values', fontdict=font_label)
    plt.tight_layout()
    plt.grid()
    plt.show()

    #IQR Outlier Detection - Installs
    q1 = df_numeric_log10['Installs'].quantile(0.25)
    q3 = df_numeric_log10['Installs'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    lower_bound, upper_bound = 10 ** lower_bound, 10 ** upper_bound
    df = df[(df['Installs'] >= lower_bound) & (df['Installs'] <= upper_bound)]


    fig, ax = plt.subplots()
    df_numeric = df.select_dtypes(include=np.number).copy()
    df_numeric_log10 = df_numeric.drop(columns=['Price']).copy()
    df_numeric_log10.replace(0, np.nan, inplace=True)
    df_numeric_log10.dropna(inplace=True)
    df_numeric_log10 = np.log10(df_numeric_log10)
    df_numeric_log10_standard = StandardScaler().fit_transform(df_numeric_log10)
    df_numeric_log10_standard = pd.DataFrame(df_numeric_log10_standard, columns=df_numeric_log10.columns)
    ax.boxplot(df_numeric_log10_standard)
    ax.set_xticklabels(df_numeric_log10_standard.columns, rotation=45)
    plt.title('Multivariate Box Plot - Outlier Removal', fontdict=font_title)
    plt.xlabel('Features', fontdict=font_label)
    plt.ylabel('Standardized Values', fontdict=font_label)
    plt.tight_layout()
    plt.grid()
    plt.show()

    return df


def principle_component_analysis(df):
    print("=========================================")
    print("Principal Component Analysis")
    print("Output: Explained Variance Ratio Plot")
    print("=========================================")
    pca_X = df.copy()
    pca_X.drop(['Installs'], axis=1, inplace=True)
    pca_columns_to_standardize = ['Rating', 'Rating Count', 'Price', 'Size in MB', 'App Age', 'Last Update Age']
    pca_X[pca_columns_to_standardize] = StandardScaler().fit_transform(pca_X[pca_columns_to_standardize])
    pca_X = pd.get_dummies(pca_X, columns=['Category', 'Content Rating', 'Minimum Android'])
    for col in pca_X.columns:
        if pca_X[col].dtype == 'bool':
            pca_X[col] = pca_X[col].astype(int)
    pca = PCA(n_components='mle', svd_solver='full')
    pca.fit(pca_X)
    pca_X_transform = pca.transform(pca_X)
    pca_results_table = PrettyTable()
    pca_results_table.float_format = '.2'
    pca_results_table.field_names = ['Original Shape', 'PCA Transformed Shape', 'Original Condition Number',
                                     'PCA Transformed Condition Number']
    pca_results_table.add_row([pca_X.shape, pca_X_transform.shape, np.linalg.cond(pca_X),
                                 np.linalg.cond(pca_X_transform)])
    print(pca_results_table)
    plt.figure()
    plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1, 1), np.cumsum(pca.explained_variance_ratio_))
    plt.xticks(np.arange(1, len(pca.explained_variance_ratio_) + 1, 5))
    plt.axvline(x=10, color='r', linestyle='--')
    plt.axhline(y=0.99, color='b', linestyle='--')
    plt.xlabel('Number of Components', fontdict=font_label)
    plt.ylabel('Cumulative Explained Variance Ratio', fontdict=font_label)
    plt.title('PCA - Cumulative Explained Variance Ratio', fontdict=font_title)
    plt.grid()
    plt.show()
    return pca_results_table


def normality_test(_df):
    df = _df.copy()
    fig = qqplot(np.log10(df['Installs']), stats.norm, fit=True, line='45')
    ax = fig.axes[0]
    ax.set_title("QQ Plot - Installs (Logarithmic) vs. Normal Distribution", fontdict=font_title)
    ax.set_xlabel('Theoretical Quantiles', fontdict=font_label)
    ax.set_ylabel('Sample Quantiles', fontdict=font_label)
    ax.grid()
    plt.tight_layout()
    plt.show()
    return shapiro_test(np.log10(df['Installs']), 'Installs(log)', 0.01)


def data_gaussian_transform(df):
    df['Installs (Log)'] = np.log10(df['Installs'])
    return df

def correlation_coefficient(_df):
    print("=========================================")
    print("Correlation Coefficient")
    print("Output: Correlation Coefficient Matrix")
    print("=========================================")
    df = _df.select_dtypes(include=np.number).copy()
    corr_matrix = df.corr()
    plt.figure()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('Pearson Correlation Coefficients Matrix', fontdict=font_title)
    plt.tight_layout()
    plt.show()
    plt.figure()

    plt.figure()
    sns.pairplot(df, corner=True, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 10, 'edgecolor': 'k'},
                 diag_kws={'alpha': 0.6, 'edgecolor': 'k'})
    plt.tight_layout()
    plt.show()

    return corr_matrix


if __name__ == '__main__':
    df = preprocessing()
    # df = outlier_detection_removal(df)
    df = outlier_detect(df)
    pca_results = principle_component_analysis(df)
    shapiro_results = normality_test(df)
    df = data_gaussian_transform(df)
    corr_matrix = correlation_coefficient(df)

    print("debug")
