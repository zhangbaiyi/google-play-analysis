# Google Play Dataset - Preprocessing
import gc
import os
from collections import Counter

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from scipy.interpolate import griddata
from sklearn.cluster import DBSCAN, KMeans

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
    if not os.path.exists('output'):
        os.makedirs('output')
    _df.to_csv('output/visualization.csv', index=False)
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
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/outlier_detect_kde_subplots.png', dpi=300)
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
    plt.savefig('plots/outlier_detect_kde_subplots_log.png', dpi=300)
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
    plt.savefig('plots/outlier_detect_multivariate_boxplot.png', dpi=300)
    plt.show()

    # IQR Outlier Detection - Installs
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
    plt.savefig('plots/outlier_detect_multivariate_boxplot_outlier_removal.png', dpi=300)
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
    plt.savefig('plots/pca_cumulative_explained_variance_ratio.png', dpi=300)
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
    plt.savefig('plots/normality_test_qqplot.png', dpi=300)
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
    plt.savefig('plots/correlation_coefficient_matrix.png', dpi=300)
    plt.show()
    plt.figure()

    plt.figure()
    sns.pairplot(df, corner=True, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 10, 'edgecolor': 'k'},
                 diag_kws={'alpha': 0.6, 'edgecolor': 'k'})
    plt.tight_layout()
    plt.savefig('plots/correlation_coefficient_pairplot.png', dpi=300)
    plt.show()

    return corr_matrix


def count_plot(_df):
    df = _df.copy()
    plt.figure(figsize=(12, 8))
    sns.countplot(data=df, x='Category', hue='Free', order=df['Category'].value_counts().index, palette='crest')
    plt.xticks(rotation=90)
    plt.legend()
    plt.title('Count Plot - Category', fontdict=font_title)
    plt.xlabel('Category', fontdict=font_label)
    plt.ylabel('Count', fontdict=font_label)
    plt.tight_layout()
    plt.grid()
    plt.savefig('plots/statistics_analysis_countplot_category.png', dpi=300)
    plt.show()


def joint_plot(_df):
    df = _df.copy()
    top_5_category = df['Category'].value_counts().index[:5]
    print("=========================================")
    print("Joint plot with KDE and scatter representation")
    print("=========================================")
    plt.figure()
    sns.jointplot(data=df[df['Category'].isin(top_5_category)], x='Name Length', y='Size in MB', hue='Category',
                  kind='scatter', palette='crest')
    plt.title('Joint Plot - Name Length vs. Size in MB', fontdict=font_title)
    plt.xlabel('Name Length', fontdict=font_label)
    plt.ylabel('Size in MB', fontdict=font_label)
    plt.tight_layout()
    plt.grid()
    plt.savefig('plots/statistics_analysis_jointplot_name_length_size_in_mb.png', dpi=300)
    plt.show()

    print("=========================================")
    print("Joint plot with KDE and scatter representation")
    print("=========================================")
    plt.figure()
    sns.jointplot(data=df[df['Category'].isin(top_5_category)], x='Rating Count', y='Installs', hue='Category',
                  kind='scatter', palette='crest')
    plt.title('Joint Plot - Rating Count vs. Installs', fontdict=font_title)
    plt.xlabel('Rating Count', fontdict=font_label)
    plt.ylabel('Installs', fontdict=font_label)
    plt.tight_layout()
    plt.grid()
    plt.savefig('plots/statistics_analysis_jointplot_rating_count_installs.png', dpi=300)
    plt.show()

    print("=========================================")
    print("Joint plot with KDE and scatter representation")
    print("=========================================")
    plt.figure()
    sns.jointplot(data=df[df['Category'].isin(top_5_category)], x='App Age', y='Last Update Age', hue='Category',
                  kind='kde', palette='crest')
    plt.title('Joint Plot - App Age vs. Last Update Age', fontdict=font_title)
    plt.xlabel('App Age', fontdict=font_label)
    plt.ylabel('Last Update Age', fontdict=font_label)
    plt.tight_layout()
    plt.grid()
    plt.savefig('plots/statistics_analysis_jointplot_app_age_last_update_age.png', dpi=300)
    plt.show()


def lm_plot(_df):
    df = _df.copy()
    df_productivity_social_entertainment = df[df['Category'].isin(['Productivity', 'Social', 'Entertainment'])].copy()
    df_productivity_social_entertainment.reset_index(drop=True, inplace=True)
    plt.figure()
    g = sns.lmplot(data=df_productivity_social_entertainment,
                   x='Rating Count',
                   y='Installs',
                   col='Category',
                   palette='crest')
    g.fig.suptitle('Regression Line Plot - Rating Count vs. Installs per Category', fontfamily='serif', fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/statistics_analysis_regplot_price_installs_category.png', dpi=300)
    plt.show()


def violin_plot(_df):
    df = _df.copy()
    df_paid = df[df['Free'] == False].copy()
    df_paid.reset_index(drop=True, inplace=True)
    df_productivity_social_entertainment = df[df['Category'].isin(['Productivity', 'Social', 'Entertainment'])].copy()
    df_productivity_social_entertainment.reset_index(drop=True, inplace=True)
    df_productivity_social_entertainment_paid = df_productivity_social_entertainment[
        df_productivity_social_entertainment['Free'] == False].copy()
    df_productivity_social_entertainment_paid.reset_index(drop=True, inplace=True)
    df_productivity_social_entertainment_paid = df_productivity_social_entertainment_paid.query('Price < 50')
    plt.figure()
    sns.violinplot(data=df_productivity_social_entertainment_paid, x='Category', y='Price', palette='crest',
                   hue='Category', legend=False)
    plt.title('Violin Plot - Price per Category', fontdict=font_title)
    plt.xlabel('Category', fontdict=font_label)
    plt.ylabel('Price', fontdict=font_label)
    plt.tight_layout()
    plt.grid()
    plt.savefig('plots/statistics_analysis_violinplot_price_category.png', dpi=300)
    plt.show()


def dist_plot(_df):
    df = _df.copy()
    df_productivity_social_entertainment = df[df['Category'].isin(['Productivity', 'Social', 'Entertainment'])].copy()
    df_productivity_social_entertainment.reset_index(drop=True, inplace=True)
    df_productivity_social_entertainment_paid = df_productivity_social_entertainment[
        df_productivity_social_entertainment['Free'] == False].copy()
    df_productivity_social_entertainment_paid.reset_index(drop=True, inplace=True)
    df_productivity_social_entertainment_paid = df_productivity_social_entertainment_paid.query('Price < 50')
    plt.figure()
    sns.displot(data=df_productivity_social_entertainment_paid, x='Price', kind='kde', hue='Category', palette='crest')
    plt.title('Dist Plot of Price', fontdict=font_title)
    plt.xlabel('Price', fontdict=font_label)
    plt.ylabel('Density', fontdict=font_label)
    plt.tight_layout()
    plt.grid()
    plt.savefig('plots/statistics_analysis_distplot_price_installs.png', dpi=300)
    plt.show()


def strip_plot(_df):
    df = _df.copy()
    plt.figure()
    df_productivity_social_entertainment = df[df['Category'].isin(['Productivity', 'Social', 'Entertainment'])].copy()
    df_productivity_social_entertainment.reset_index(drop=True, inplace=True)
    df_productivity_social_entertainment_paid = df_productivity_social_entertainment[
        df_productivity_social_entertainment['Free'] == False].copy()
    df_productivity_social_entertainment_paid.reset_index(drop=True, inplace=True)
    df_productivity_social_entertainment_paid = df_productivity_social_entertainment_paid.query('Price < 50')
    sns.stripplot(data=df_productivity_social_entertainment_paid, y='Category', x='Price', palette='crest',
                  hue='Content Rating', dodge=True)
    plt.title('Strip Plot - Price per Category', fontdict=font_title)
    plt.xlabel('Price', fontdict=font_label)
    plt.ylabel('Category', fontdict=font_label)
    plt.tight_layout()
    plt.grid()
    plt.savefig('plots/statistics_analysis_stripplot_price_category.png', dpi=300)
    plt.show()


def swarm_plot(_df):
    df = _df.copy()
    df = df[df['Editors Choice'] == True]
    df.reset_index(drop=True, inplace=True)
    plt.figure()
    sns.swarmplot(data=df, x='Minimum Android', y='Rating', hue='In App Purchases', palette='crest')
    plt.title('Swarm Plot - Minimum Android vs. Rating', fontdict=font_title)
    plt.xlabel('Minimum Android', fontdict=font_label)
    plt.ylabel('Rating', fontdict=font_label)
    plt.grid()
    plt.tight_layout()
    plt.show()


def bar_plot(_df):
    df = _df.copy()
    plt.figure()
    category = df['Category'].value_counts().index[:10]
    sns.barplot(data=df, x=df['Category'], y=df['Rating'], hue='In App Purchases', palette='crest', ci=None,
                order=category)
    plt.title('Bar Plot (Grouped) - Rating vs. Category', fontdict=font_title)
    plt.xlabel('Category', fontdict=font_label)
    plt.xticks(rotation=90)
    plt.ylabel('Rating', fontdict=font_label)
    plt.grid()
    plt.tight_layout()
    plt.legend(title='In App Purchases', loc='lower left')
    plt.savefig('plots/statistics_analysis_barplot_rating_installs_category.png', dpi=300)
    plt.show()

    df_groupby_category_ad_supported = df.groupby(['Category', 'Ad Supported'])['Rating'].mean().reset_index()
    top_categories = df['Category'].value_counts().index[:3]
    df_groupby_category_ad_supported = df_groupby_category_ad_supported[
        df_groupby_category_ad_supported['Category'].isin(top_categories)]
    plt.figure()
    bottom = np.zeros(3)
    df_groupby_category_ad_supported_yes = df_groupby_category_ad_supported[
        df_groupby_category_ad_supported['Ad Supported'] == True]
    df_groupby_category_ad_supported_no = df_groupby_category_ad_supported[
        df_groupby_category_ad_supported['Ad Supported'] == False]
    p_no = plt.bar(x=df_groupby_category_ad_supported_no['Category'], height=df_groupby_category_ad_supported_no['Rating'], label='No', bottom=bottom)
    bottom += df_groupby_category_ad_supported_no['Rating']
    plt.bar_label(p_no, label_type='center', fmt='%.2f')
    p_yes = plt.bar(x=df_groupby_category_ad_supported_yes['Category'], height=df_groupby_category_ad_supported_yes['Rating'], label='Yes', bottom=bottom)
    plt.bar_label(p_yes, label_type='center', fmt='%.2f')
    plt.title('Bar Plot (Stacked) - Rating vs. Category', fontdict=font_title)
    plt.xlabel('Category', fontdict=font_label)
    plt.ylabel('Rating', fontdict=font_label)
    plt.grid()
    plt.tight_layout()
    plt.legend(title='Ad Supported', loc='lower left')
    plt.savefig('plots/statistics_analysis_barplot_rating_installs_category_stacked.png', dpi=300)
    plt.show()

def pie_chart(_df):
    df = _df.copy()
    plt.figure(figsize=(12, 8))
    selected_category = df['Category'].value_counts().index[:15]
    df['Category'] = df['Category'].apply(lambda x: 'Others' if x not in selected_category else x)
    index = df['Category'].values
    value = df['Category'].value_counts().values
    percent = 100. * value / value.sum()
    patches, texts, _ = plt.pie(value, startangle=90, radius=1.2, autopct='%1.2f%%', pctdistance=0.8, explode=[0.05] * len(value))
    labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(index, percent)]

    patches, labels, _ = zip(*sorted(zip(patches, labels, value),
                                    key=lambda x: x[2],
                                    reverse=True))
    plt.legend(patches, labels, loc='best', bbox_to_anchor=(-0.1, 1.), fontsize=8)
    plt.title('Pie Chart - Category', fontdict=font_title)
    plt.tight_layout()
    plt.savefig('plots/statistics_analysis_piechart_category.png', dpi=300)
    plt.show()

def hexbin_plot(_df):
    df = _df.copy()
    df = df.sample(frac=0.01, random_state=42)
    plt.figure()
    sns.jointplot(data=df, x='Rating', y='Installs (Log)', kind='hex', color='k')
    plt.title('Hexbin Plot - Rating vs. Installs (Log)', fontdict=font_title)
    plt.xlabel('Rating', fontdict=font_label)
    plt.ylabel('Installs (Log)', fontdict=font_label)
    plt.tight_layout()
    plt.grid()
    plt.savefig('plots/statistics_analysis_hexbinplot_rating_installs.png', dpi=300)
    plt.show()


def plot_pca_clusters(X, clusters, model, title, cluster_ids_to_plot=None):
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(X)
    plt.figure(figsize=(10, 7))
    unique_clusters = set(clusters)
    if cluster_ids_to_plot is not None:
        clusters_to_plot = set(cluster_ids_to_plot)
    else:
        clusters_to_plot = unique_clusters
    for cluster_id in clusters_to_plot:
        if cluster_id in unique_clusters:  # Check if cluster exists
            plt.scatter(df_pca[clusters == cluster_id, 0], df_pca[clusters == cluster_id, 1], label=f'Cluster {cluster_id + 1}')
    if model is not None and hasattr(model, 'cluster_centers_'):
        pca_centroids = pca.transform(model.cluster_centers_)
        plt.scatter(pca_centroids[:, 0], pca_centroids[:, 1], s=50, c='black', label='Centroids', marker='X')
    plt.title(title, fontdict=font_title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'plots/{title}.png', dpi=300)
    plt.show()


def cluster_plot(_df):
    X = _df.copy()
    X.drop(columns=['Category', 'Content Rating', 'Installs', 'Installs (Log)'], inplace=True)
    X = StandardScaler().fit_transform(X)
    n_clusters = 7
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10)
    clusters = kmeans.fit_predict(X)
    plot_pca_clusters(X, clusters, kmeans, 'Cluster Visualization with PCA - K-means++')


def contour_plot(_df):
    df = _df.copy()
    df = df.sample(frac=0.01, random_state=42)
    x = df['Rating']
    y = df['Installs (Log)']
    z = df['App Age']
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method='linear')
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Installs (Log)')
    ax.set_zlabel('App Age')
    ax.set_title('3D Surface Plot of App Data', fontdict=font_title, fontsize=30)
    plt.tight_layout()
    plt.savefig('plots/statistics_analysis_piechart_category.png', dpi=300)
    plt.show()


def rug_plot(_df):
    df = _df.copy()
    df = df.query('Price < 50')
    df = df.query('Price > 0')
    plt.figure()
    sns.scatterplot(data=df, x='Price', y='Rating', hue='In App Purchases', palette='crest')
    sns.rugplot(data=df, x='Price', y='Rating', hue='In App Purchases', palette='crest')
    plt.title('Rug Plot - Rating vs. Price', fontdict=font_title)
    plt.xlabel('Price', fontdict=font_label)
    plt.ylabel('Rating', fontdict=font_label)
    plt.grid()
    plt.tight_layout()
    plt.savefig('plots/statistics_analysis_rugplot_price_rating.png', dpi=300)
    plt.show()


def area_plot(_df):
    df = _df.copy()
    df['Rating'] = df['Rating'].apply(lambda x: int(x))
    df_rating = df.groupby(['Category', 'Rating']).size().reset_index(name='Count')
    df_rating_education = df_rating[df_rating['Category'] == 'Education'].copy()
    df_rating_education.set_index('Rating', inplace=True)
    df_rating_education.drop(columns=['Category'], inplace=True)
    df_rating_social = df_rating[df_rating['Category'] == 'Social'].copy()
    df_rating_social.set_index('Rating', inplace=True)
    df_rating_social.drop(columns=['Category'], inplace=True)
    df_rating_entertainment = df_rating[df_rating['Category'] == 'Entertainment'].copy()
    df_rating_entertainment.set_index('Rating', inplace=True)
    df_rating_entertainment.drop(columns=['Category'], inplace=True)
    df_combined = pd.concat([df_rating_education, df_rating_social, df_rating_entertainment], axis=1)
    df_combined.columns = ['Education', 'Social', 'Entertainment']
    plt.figure(figsize=(10, 6))
    df_combined.plot(kind='area', stacked=False, alpha=0.6)
    plt.title('Area Plot - Rating vs. Count', fontdict=font_title)
    plt.xlabel('Rating', fontdict=font_label)
    plt.ylabel('Count', fontdict=font_label)
    plt.tight_layout()
    plt.grid()
    plt.savefig('plots/statistics_analysis_areaplot_rating_count_stack_false.png', dpi=300)
    plt.show()




if __name__ == '__main__':
    start_time = pd.Timestamp.now()
    df = preprocessing()
    df = outlier_detect(df)
    pca_results = principle_component_analysis(df)
    shapiro_results = normality_test(df)
    df = data_gaussian_transform(df)
    corr_matrix = correlation_coefficient(df)
    count_plot(df)
    joint_plot(df)
    lm_plot(df)
    violin_plot(df)
    dist_plot(df)
    strip_plot(df)
    bar_plot(df)
    pie_chart(df)
    area_plot(df)
    rug_plot(df)
    contour_plot(df)
    cluster_plot(df)
    hexbin_plot(df)
    swarm_plot(df)
    print("Finished, total time: ", pd.Timestamp.now() - start_time)
