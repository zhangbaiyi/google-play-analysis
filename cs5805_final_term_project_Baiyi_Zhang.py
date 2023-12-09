# Google Play Dataset - Preprocessing

# Standard library imports
import os
import re
import time
import gc
import random
from collections import Counter

# Related third party imports
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.spatial import distance
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (mean_squared_error, make_scorer, classification_report, roc_auc_score,
                             roc_curve, f1_score, accuracy_score, confusion_matrix)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
from prettytable import PrettyTable

# Disable specific warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)

# Setting pandas options
pd.options.mode.chained_assignment = None

# import re
# import time
# import pandas as pd
#
# pd.options.mode.chained_assignment = None
# from sklearn.model_selection import KFold, cross_val_score
# from sklearn.metrics import mean_squared_error, make_scorer
# import statsmodels.api as sm
# import random
# import gc
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.decomposition import TruncatedSVD
# from sklearn.preprocessing import StandardScaler
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# import os
# import scipy.stats as stats
# import seaborn as sns
# from statsmodels.graphics.gofplots import qqplot
# from sklearn.model_selection import GridSearchCV
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import StackingClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, roc_auc_score, roc_curve, f1_score
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
# from sklearn.neighbors import KNeighborsClassifier
# from scipy.spatial import distance
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.metrics import silhouette_score
# from sklearn.cluster import DBSCAN
# from collections import Counter
# from mlxtend.frequent_patterns import apriori
# from mlxtend.frequent_patterns import association_rules
# import pandas as pd
# from prettytable import PrettyTable
# import warnings
#
# warnings.filterwarnings('ignore', category=UserWarning)
# warnings.simplefilter("ignore", DeprecationWarning)


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


def judge_cat_num(col, df):
    if df[col].dtype == 'float64' or df[col].dtype == 'int64':
        return 'Numerical'
    else:
        return 'Categorical'


def check_na_percentage(col, df):
    ratio = df[col].isna().sum() / len(df[col])
    return f"{ratio:.2%}"


def check_unique_percentage(col, df):
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


def shapiro_test(x, title, alpha=0.05):
    from scipy.stats import shapiro
    stat, p = shapiro(x)
    # print(f'Shapiro-Wilk test: statistics= {stat:.2f} p-value = {p:.2f}')
    print(f'Shapiro-Wilk test: statistics= {stat:.2f} p-value = {p}')
    print(
        f'Shapiro-Wilk test: {title} dataset looks normal (fail to reject H0)' if p > alpha else f'Shapiro-Wilk test: {title} dataset does not look normal (reject H0)')


def preprocessing():
    _df = pd.read_csv('Google-Playstore.csv')
    random.seed(5805)
    print("=========================================")
    print("Raw data overview")
    print("Output: table of feature overview")
    print("=========================================")
    feature_outlook = PrettyTable()
    feature_outlook.field_names = ["Feature", "Type", "N/A Count", "N/A Percentage", "Cat/Num", "Unique Count",
                                   "Unique Percentage", "Example"]
    for col in _df.columns:
        feature_outlook.add_row(
            [col, _df[col].dtype, _df[col].isna().sum(), check_na_percentage(col, _df), judge_cat_num(col, _df),
             len(_df[col].unique()),
             check_unique_percentage(col, _df), _df[col].unique()[random.randint(0, len(_df[col].unique()) - 1)]])
    print(feature_outlook)
    del feature_outlook
    gc.collect()

    print("=========================================")
    print("NA Values Percentage")
    print("Output: bar chart of NA values percentage")
    print("=========================================")

    na_feature_percentage = _df.isna().sum().sort_values(ascending=False) / len(_df) * 100
    na_feature_percentage = na_feature_percentage[na_feature_percentage > 0]
    plt.figure(figsize=(10, 5))
    na_feature_percentage.plot(kind='barh')
    plt.xlabel('Percentage (%)')
    plt.ylabel('Features')
    plt.title('Percentage of N/A Values')
    plt.grid()
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/na_percentage.png')
    plt.show()
    _df.drop(columns=['Developer Website', 'Developer Email', 'Developer Id', 'Privacy Policy'], inplace=True)
    _df.drop(columns=['App Id'], inplace=True)
    _df.drop(columns=['Free'], inplace=True)
    _df.drop(columns=['Scraped Time'], inplace=True)
    del na_feature_percentage
    gc.collect()

    print("=========================================")
    print("Down sampling")
    print("Output: shape of the dataset")
    print("=========================================")
    _df = _df.query('`Maximum Installs` > 100000')
    _df = _df.query('`Maximum Installs` < 10000000')
    print(_df.shape)

    print("=========================================")
    print("Duplication in Currency column")
    print("Output: pie chart of currency distribution")
    print("=========================================")
    currency_values_count = _df['Currency'].value_counts()
    currency_pie = currency_values_count.head(1)
    currency_pie['Others'] = currency_values_count[1:].sum()
    currency_pie.plot(kind='pie', autopct='%1.1f%%', labels=['USD', 'Others'], ylabel='Currency',
                      title='Currency Distribution')
    plt.savefig('plots/currency_distribution.png')
    plt.show()
    del currency_values_count, currency_pie
    gc.collect()

    # Drop all non-USD currency
    _df['Currency'] = _df['Currency'].apply(lambda x: 'USD' if x == 'USD' else 'Others')
    _df.drop(_df[_df['Currency'] == 'Others'].index, inplace=True)
    print(_df.shape)
    _df.drop(columns=['Currency'], inplace=True)
    print("Shape of Dataframe after modification: {}".format(str(_df.shape)))

    print("=========================================")
    print("Duplication between 'Installs' and 'Minimum Installs'")
    print("Output: boolean value")
    print("=========================================")
    # Test if installs and min-install are the same
    installs = _df['Installs'].apply(lambda x: x.replace('+', '') if '+' in x else x)
    installs = pd.to_numeric(installs.apply(lambda x: x.replace(',', '') if ',' in x else x))
    min_installs = _df['Minimum Installs']
    min_installs = pd.to_numeric(min_installs)
    min_installs = min_installs.astype('int64')
    print("After conversion, is installs equal to min_installs?")
    print(installs.equals(min_installs))
    del installs, min_installs
    gc.collect()

    # Drop 'Installs'
    _df.drop(columns=['Installs'], inplace=True)
    print("Shape of Dataframe after modification: {}".format(str(_df.shape)))

    print("=========================================")
    print("Aggregation of Android versions.")
    print("Output: Null")
    print("=========================================")
    # Replace ' and up' and ' - ' in the entire column
    _df['Minimum Android'] = _df['Minimum Android'].str.replace(' and up', '').str.split(' - ').str.get(0)
    _df['Minimum Android'] = _df['Minimum Android'].str.split('.').str.get(0)
    # Replace 'Varies with device' with NaN
    _df['Minimum Android'] = _df['Minimum Android'].apply(lambda x: np.nan if x == 'Varies with device' else x)
    print("Shape of Dataframe after modification: {}".format(str(_df.shape)))

    print("=========================================")
    print("Change size unit to MB")
    print("Output: Null")
    print("=========================================")
    _df['Clean Size'] = _df['Size'].apply(classify_size_column)
    _df['Clean Size'].describe()
    plt.figure(figsize=(10, 10))
    _df['Clean Size'].plot(kind='hist', bins=100)
    plt.title('Size Distribution')
    plt.xlabel('Size (MB)')
    plt.ylabel('Count')
    plt.savefig('plots/size_distribution.png')
    plt.show()
    _df.drop(columns=['Size'], inplace=True)
    print("Shape of Dataframe after modification: {}".format(str(_df.shape)))

    print("=========================================")
    print("Replace date by age in days")
    print("Output: Null")
    print("=========================================")
    # Replace date by age (in days)
    _df['Released'] = pd.to_datetime(_df['Released'], format='%b %d, %Y')
    _df['Last Updated'] = pd.to_datetime(_df['Last Updated'], format='%b %d, %Y')
    scraped_time = pd.to_datetime('2021-06-15 00:00:00')
    _df['App Age'] = (scraped_time - _df['Released']).dt.days
    # Last update age
    _df['Last Update Age'] = (scraped_time - _df['Last Updated']).dt.days
    _df.drop(columns=['Released', 'Last Updated'], inplace=True)
    print("Shape of Dataframe after modification: {}".format(str(_df.shape)))

    print("=========================================")
    print("Drop N/A values and duplicates")
    print("Output: Null")
    print("=========================================")
    _df.dropna(inplace=True)
    _df.drop_duplicates(inplace=True)
    print("Shape of Dataframe after modification: {}".format(str(_df.shape)))

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
    _df.rename(columns=rename_dict, inplace=True)
    print("Shape of Dataframe after modification: {}".format(str(_df.shape)))

    print("=========================================")
    print("Random Forest Analysis")
    print("Output: Feature Importance Plot")
    print("=========================================")
    rfa_X = _df.copy()
    rfa_y = _df['installCount'].copy()
    rfa_X.drop(columns=['installCount', 'installRange'], inplace=True)
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
    plt.savefig('plots/feature_importance_1.png')
    plt.show()
    del rfa_X, rfa_y, rfa_X_train, rfa_X_test, rfa_y_train, rfa_y_test, rfa, rfa_y_pred, importances, indices, features
    gc.collect()

    print("=========================================")
    print("Principal Component Analysis")
    print("Output: Explained Variance Ratio Plot")
    print("=========================================")
    pca_X = _df.copy()
    pca_y = _df['installCount'].copy()
    pca_X.drop(columns=['installCount', 'installRange'], inplace=True)
    pca_X.drop(columns=['appName'], inplace=True)
    pca_columns_to_standardize = ['rating', 'ratingCount', 'priceInUSD', 'sizeInMB', 'appAgeInDays',
                                  'lastUpdateAgeInDays']
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
    plt.figure()
    plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1, 1), np.cumsum(pca.explained_variance_ratio_))
    plt.xticks(np.arange(1, len(pca.explained_variance_ratio_) + 1, 5))
    plt.axvline(x=10, color='r', linestyle='--')
    plt.axhline(y=0.85, color='b', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA - Cumulative Explained Variance Ratio')
    plt.grid()
    plt.savefig('plots/pca_explained_variance_ratio.png')
    plt.show()
    del pca_X, pca_y, pca, pca_X_transform, pca_columns_to_standardize
    gc.collect()

    print("=========================================")
    print("Single Value Decomposition")
    print("Output: Singular values of original and SVD transformed matrix")
    print("=========================================")
    svd_X = _df.copy()
    svd_y = _df['installCount'].copy()
    svd_X.drop(columns=['installCount', 'installRange'], inplace=True)
    svd_X.drop(columns=['appName'], inplace=True)

    svd_columns_to_standardize = ['rating', 'ratingCount', 'priceInUSD', 'sizeInMB', 'appAgeInDays',
                                  'lastUpdateAgeInDays']
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
    vif_X = _df.copy()
    vif_X.drop(columns=['installCount', 'installRange'], inplace=True)
    vif_X.drop(columns=['appName', 'category', 'contentRating', 'minAndroidVersion'], inplace=True)
    vif_columns_to_standardize = ['rating', 'ratingCount', 'priceInUSD', 'sizeInMB', 'appAgeInDays',
                                  'lastUpdateAgeInDays']
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
    _df = pd.get_dummies(_df, columns=['category', 'contentRating', 'minAndroidVersion'])
    _df = _df[['ratingCount', 'sizeInMB', 'lastUpdateAgeInDays', 'minAndroidVersion_8', 'appAgeInDays', 'rating',
               'category_Productivity', 'category_Social', 'isInAppPurchases', 'minAndroidVersion_7', 'installCount',
               'installRange']]
    for col in _df.columns:
        if _df[col].dtype == 'bool':
            _df[col] = _df[col].astype(int)
    _df['installRange'] = _df['installRange'].apply(install_groupby)
    print("Shape of Dataframe after modification: {}".format(str(_df.shape)))

    print("=========================================")
    print("Standardization")
    print("Output: Null")
    print("=========================================")
    _df_standard = _df.copy()
    scaler = StandardScaler()
    _df_standard[['ratingCount', 'sizeInMB', 'lastUpdateAgeInDays', 'appAgeInDays', 'rating']] = scaler.fit_transform(
        _df_standard[['ratingCount', 'sizeInMB', 'lastUpdateAgeInDays', 'appAgeInDays', 'rating']])
    print("Shape of Dataframe after modification: {}".format(str(_df.shape)))

    print("=========================================")
    print("Outlier Detection")
    print("Output: Outlier Detection Result")
    print("=========================================")

    sns.kdeplot(data=_df, x=np.log10(_df['installCount'].values), fill=True)
    plt.savefig('plots/kde_1.png')
    plt.show()
    fig = qqplot(np.log10(_df['installCount']), stats.norm, fit=True, line='45')
    ax = fig.axes[0]
    ax.set_title("QQ Plot - Installs (Log) vs. Normal Distribution")
    plt.savefig('plots/qqplot_1.png')
    plt.show()

    shapiro_test(_df['installCount'], 'Before box-cox', 0.01)

    _df['box_cox_installs'], fitted_lambda = stats.boxcox(_df['installCount'])
    sns.kdeplot(data=_df, x=_df['box_cox_installs'], fill=True)
    plt.savefig('plots/kde_2.png')
    plt.show()

    fig = qqplot(_df['box_cox_installs'], stats.norm, fit=True, line='45')
    ax = fig.axes[0]
    ax.set_title("QQ Plot - Installs vs. Normal Distribution")
    plt.savefig('plots/qqplot_2.png')
    plt.show()

    shapiro_test(_df['box_cox_installs'], 'After box-cox', 0.01)
    _df_standard['box_cox_installs'], _ = stats.boxcox(_df_standard['installCount'])
    print("=========================================")
    print("Covariance Matrix")
    print("Output: Covariance Matrix")
    print("=========================================")
    covariance_matrix = _df_standard[['ratingCount', 'sizeInMB', 'lastUpdateAgeInDays', 'appAgeInDays', 'rating']].cov()
    plt.figure(figsize=(12, 10))
    sns.heatmap(covariance_matrix, annot=True, fmt=".5f", cmap='coolwarm', linewidths=0.5)
    plt.title('Covariance Matrix')
    plt.tight_layout()
    plt.savefig('plots/covariance_matrix.png')
    plt.show()
    del covariance_matrix
    gc.collect()

    print("=========================================")
    print("Correlation Matrix")
    print("Output: Correlation Matrix")
    print("=========================================")
    corr_matrix = _df_standard[['ratingCount', 'sizeInMB', 'lastUpdateAgeInDays', 'appAgeInDays', 'rating']].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".5f", cmap='coolwarm', linewidths=0.5)
    plt.title('Pearson Correlation Coefficients Matrix')
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png')
    plt.show()
    del corr_matrix
    gc.collect()

    print("=========================================")
    print("Balanced/Imbalanced Target Distribution")
    print("Output: Bar Chart of Target Distribution")
    print("=========================================")
    value_counts = pd.DataFrame(_df['installRange'].value_counts())
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
    plt.savefig('plots/target_distribution.png')
    plt.show()
    del value_counts, total_count
    gc.collect()

    _df['installQcut'] = pd.qcut(_df['installCount'], 2, labels=['Low', 'High'])
    _df_standard['installQcut'] = pd.qcut(_df_standard['installCount'], 2, labels=['Low', 'High'])
    _df['installQcut'].replace({'Low': 0, 'High': 1}, inplace=True)
    _df_standard['installQcut'].replace({'Low': 0, 'High': 1}, inplace=True)
    value_counts = pd.DataFrame(_df['installQcut'].value_counts())
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
    plt.savefig('plots/target_distribution_2.png')
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
            _df_standard.to_csv('output/preprocessed_standard.csv', index=False)
            print("File preprocessed_standard.csv created")
        if not os.path.exists('output/preprocessed.csv'):
            _df.to_csv('output/preprocessed.csv', index=False)
            print("File preprocessed.csv created")
    else:
        print("Folder output already exists."
              "To re-create the files, please delete the folder and run the script again.")

    _df.reset_index(drop=True, inplace=True)
    _df_standard.reset_index(drop=True, inplace=True)
    return _df, _df_standard


def reverse_standardize(x, installs_mean, installs_std):
    return x * installs_std + installs_mean


# def read_and_preprocess_data(file_path):
#     df = pd.read_csv(file_path)
#     df.drop(columns=['installRange', 'box_cox_installs', 'installQcut'], inplace=True)
#     df['installCount'] = StandardScaler().fit_transform(df[['installCount']])
#     return df


def backward_stepwise_regression(_X_train, _y_train):
    table = PrettyTable()
    table.title = "Backward Stepwise Regression"
    table.field_names = ['Attempt', 'Removed Feature', 'Removed p-value', 'AIC', 'BIC', 'R^2',
                         'Adjusted R^2', 'MSE']
    table.float_format = '.3'

    model = sm.OLS(_y_train, _X_train).fit()
    print(model.summary())
    table.add_row(
        [0, 'N/A', 'N/A', model.aic, model.bic, model.rsquared, model.rsquared_adj, model.mse_model])
    print(table)

    print(model.summary())
    removed_p_value = model.pvalues.max()
    _X_train.drop(['category_Productivity'], axis=1, inplace=True)
    model = sm.OLS(_y_train, _X_train).fit()
    table.add_row(
        [1, 'category_Productivity', removed_p_value, model.aic, model.bic, model.rsquared, model.rsquared_adj,
         model.mse_model])
    print(table)

    print(model.summary())
    removed_p_value = model.pvalues.max()
    _X_train.drop(['minAndroidVersion_8'], axis=1, inplace=True)
    model = sm.OLS(_y_train, _X_train).fit()
    table.add_row(
        [2, 'minAndroidVersion_8', removed_p_value, model.aic, model.bic, model.rsquared, model.rsquared_adj,
         model.mse_model])
    print(table)

    print(model.summary())
    removed_p_value = model.pvalues.max()
    _X_train.drop(['category_Social'], axis=1, inplace=True)
    model = sm.OLS(_y_train, _X_train).fit()
    table.add_row(
        [3, 'category_Social', removed_p_value, model.aic, model.bic, model.rsquared, model.rsquared_adj,
         model.mse_model])
    print(table)

    return table, model.summary()


def train_random_forest(X, y):
    table = PrettyTable()
    _X_train, _X_test, _y_train, _y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    _rf = RandomForestRegressor(n_estimators=100, random_state=42)
    _rf.fit(_X_train, _y_train)
    features = _X_train.columns
    importances = _rf.feature_importances_
    indices = np.argsort(importances)[-13:]
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig('plots/feature_importance_2.png')
    plt.show()

    feature_importance_df = pd.DataFrame({'feature': features, 'importance': importances})
    threshold = 0.05
    rf_important_features = feature_importance_df[feature_importance_df['importance'] > threshold]['feature']
    print(rf_important_features)
    _X_train, _X_test, _y_train, _y_test = train_test_split(X[rf_important_features], y, test_size=0.2, random_state=42)
    _rf = RandomForestRegressor(n_estimators=100, random_state=42)
    _rf.fit(_X_train, _y_train)
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(_rf, _X_train, _y_train, cv=kfold, scoring=make_scorer(mean_squared_error), n_jobs=-1)

    table.title = "Random Forest Regression - Final Results"
    table.field_names = ['Training R^2', 'Testing R^2', 'Training MSE', 'Testing MSE']
    table.float_format = '.3'
    table.add_row([_rf.score(_X_train, _y_train), _rf.score(_X_test, _y_test),
                   np.mean((_rf.predict(_X_train) - _y_train) ** 2),
                   np.mean((_rf.predict(_X_test) - _y_test) ** 2)])
    print(table)
    return _rf, table, scores, _rf.predict(_X_test)


def plot_results(_results_df, _scores, _installs_mean, _installs_std):
    plt.figure(figsize=(30, 10))
    plt.title("Predicted vs Actual")
    _results_df['Actual'] = _results_df['Actual'].apply(reverse_standardize, args=(_installs_mean, _installs_std))
    _results_df['Predicted'] = _results_df['Predicted'].apply(reverse_standardize, args=(_installs_mean, _installs_std))
    _results_df['Predicted Upper'] = _results_df['Predicted'] + 1.96 * np.sqrt(_scores.mean())
    _results_df['Predicted Lower'] = _results_df['Predicted'] - 1.96 * np.sqrt(_scores.mean())
    _results_df['Actual'].plot(kind='line', label='Actual')
    _results_df['Predicted'].plot(kind='line', label='Predicted')
    plt.xlabel("# of Samples")
    plt.ylabel("Installs")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.title("Predicted vs Actual")
    plt.savefig('plots/regression_predicted_vs_actual.png')
    plt.show()

    plt.figure(figsize=(12, 8))
    results_df_head = _results_df.head(100)
    results_df_head['Actual'].plot(kind='line', label='Actual')
    results_df_head['Predicted'].plot(kind='line', label='Predicted')
    plt.fill_between(range(len(results_df_head['Predicted'])), results_df_head['Predicted Lower'],
                     results_df_head['Predicted Upper'], color='orange', alpha=0.4)
    plt.xlabel("# of Samples")
    plt.ylabel("Installs")
    plt.grid()
    plt.tight_layout()
    plt.legend()
    plt.title("Predicted vs Actual with Confidence Interval - First 100 Samples")
    plt.savefig('plots/regression_predicted_vs_actual_ci.png')
    plt.show()


def regression(_df):
    df = _df.copy()
    df.drop(columns=['installRange', 'box_cox_installs', 'installQcut'], inplace=True)
    df['installCount'] = StandardScaler().fit_transform(df[['installCount']])
    _df = df.copy()
    X = _df.drop(columns=['installCount'])
    installs_mean = _df['installCount'].mean()
    installs_std = _df['installCount'].std()
    y = _df['installCount']
    X = sm.add_constant(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    regression_table, ols_summary = backward_stepwise_regression(X_train, y_train)

    rf, random_forest_results_table, kfold_scores, y_pred = train_random_forest(X, y)

    results_df = pd.DataFrame({'Actual': y_test.reset_index(drop=True), 'Predicted': y_pred})
    plot_results(results_df, kfold_scores, installs_mean, installs_std)

    print(regression_table)
    print(ols_summary)
    print(random_forest_results_table)
    return regression_table, ols_summary, random_forest_results_table


class ClassifierMetrics:
    def __init__(self, y_true, y_pred, grid_search=None):
        self.classification_report = classification_report(y_true, y_pred)
        self.accuracy = accuracy_score(y_true, y_pred)
        self.confusion_matrix = confusion_matrix(y_true, y_pred)
        self.precision = self._calculate_precision()
        self.recall = self._calculate_recall()
        self.specificity = self._calculate_specificity()
        self.f1_score = f1_score(y_true, y_pred)
        self.roc_fpr, self.roc_tpr, self.roc_thresholds = roc_curve(y_true, y_pred)
        self.roc_auc = roc_auc_score(y_true, y_pred)
        self.k_fold_results = grid_search.cv_results_ if grid_search else None

    def _calculate_precision(self):
        tn, fp, fn, tp = self.confusion_matrix.ravel()
        return tp / (tp + fp)

    def _calculate_recall(self):
        tn, fp, fn, tp = self.confusion_matrix.ravel()
        return tp / (tp + fn)

    def _calculate_specificity(self):
        tn, fp, fn, tp = self.confusion_matrix.ravel()
        return tn / (tn + fp)


def plot_roc_curve_plt(fpr, tpr, auc, title):
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.gca().set_aspect('equal', 'box')
    plt.tight_layout()

    plt.show()


def elbow_method_knn(my_X, my_y):
    plt.figure()
    X_train, X_test, y_train, y_test = train_test_split(my_X, my_y, test_size=0.2, random_state=42,
                                                        stratify=my_y)
    error_rate = []
    for i in range(1, 30):
        scope_knn = KNeighborsClassifier(n_neighbors=i, metric='manhattan', weights='distance')
        scope_knn.fit(X_train, y_train)
        pred = scope_knn.predict(X_test)
        error_rate.append(np.mean(pred != y_test))
    plt.plot(range(1, 30, 1), error_rate, marker='o', markersize=9)
    plt.ylabel('Error Rate')
    plt.xlabel('k')
    plt.xticks(np.arange(1, 30, 2))
    plt.title('Error Rate vs. K Value - (Elbow Method)')
    plt.grid()
    plt.tight_layout()
    plt.savefig('plots/elbow_method_knn.png', dpi= 300)
    plt.show()


def optimize_cost_complexity_pruning_alpha(my_X, my_y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(my_X, my_y, test_size=0.2, random_state=random_state,
                                                        stratify=my_y)
    ccp_alphas = np.linspace(0.0, 0.5, 50)
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)
    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]
    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs Alpha for Training and Testing sets")
    ax.plot(ccp_alphas, train_scores, marker='o', label="train",
            drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker='o', label="test",
            drawstyle="steps-post")
    ax.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('plots/accuracy_vs_alpha.png', dpi= 300)
    plt.show()


def output_performance_to_table(classifer_title, classifier_metrics, master_table):
    master_table.add_row([classifer_title,
                          classifier_metrics.confusion_matrix,
                          classifier_metrics.precision,
                          classifier_metrics.recall,
                          classifier_metrics.specificity,
                          classifier_metrics.f1_score,
                          classifier_metrics.roc_auc])


def classifier_pipeline(classifier, tuned_parameters, X, y, test_size=0.2, random_state=42, k_fold=5,
                        scoring='accuracy'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        stratify=y)
    grid_search = GridSearchCV(classifier, tuned_parameters, cv=k_fold, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("====================================")
    print("Best parameters set found on development set:")
    print(grid_search.best_params_)
    print("====================================")
    print("Grid scores on development set:")
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print("====================================")
    return grid_search.best_estimator_, X_test, y_test, grid_search


def plot_roc_curve_individual(fpr, tpr, auc, title):
    fig, axis = plt.subplots()
    axis.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
    axis.plot([0, 1], [0, 1], 'k--')
    axis.set_xlim([-0.05, 1.0])
    axis.set_ylim([0.0, 1.05])
    axis.set_xlabel('False Positive Rate')
    axis.set_ylabel('True Positive Rate')
    axis.set_title(title)
    axis.grid(True)
    axis.set_aspect('equal', 'box')
    axis.legend(loc="lower right")
    plt.savefig('plots/individual_roc_curve_{}.png'.format(title))
    plt.show()
    plt.close(fig)


def plot_roc_curve(axis, fpr, tpr, auc, title):
    axis.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
    axis.plot([0, 1], [0, 1], 'k--')
    axis.set_xlim([-0.05, 1.0])
    axis.set_ylim([0.0, 1.05])
    axis.set_xlabel('False Positive Rate')
    axis.set_ylabel('True Positive Rate')
    axis.set_title(title)
    axis.grid(True)
    axis.set_aspect('equal', 'box')
    axis.legend(loc="lower right")


def classifier_metrics(classifier, y_test, X_test, grid_search=None):
    y_true, y_pred = y_test, classifier.predict(X_test)
    return ClassifierMetrics(y_true, y_pred, grid_search)


def classifier_fpr_tpr(classifier, y_test, X_test):
    fpr, tpr, _ = roc_curve(y_test, classifier.predict_proba(X_test)[:, 1])
    return fpr, tpr


def classification(outer_df, outer_df_standard):
    # _df = pd.read_csv('output/preprocessed.csv')
    # _df_standard = pd.read_csv('output/preprocessed_standard.csv')
    _df = outer_df.copy()
    _df_standard = outer_df_standard.copy()
    # fig, ax = plt.subplots(3, 3, figsize=(20, 15))
    master_table = PrettyTable()
    master_table.title = "Classifier Performance"
    master_table.float_format = ".3"
    master_table.field_names = ["Classifier", "Confusion Matrix", "Precision", "Recall", "Specificity", "F1 Score",
                                "AUC"]

    print("====================================")
    print("Decision Tree")
    print("Pre-pruning")
    print("====================================")
    decision_tree_pre_tuned_parameters = [{'max_depth': [1, 4, 10],
                                           'min_samples_split': [2, 5, 10],
                                           'min_samples_leaf': [1, 4],
                                           'max_features': [1, 5, 10, 'sqrt', 'log2'],
                                           'splitter': ['best', 'random'],
                                           'criterion': ['gini', 'entropy']}]

    X = _df.drop(columns=['installRange', 'installCount', 'box_cox_installs', 'installQcut'], axis=1)
    y = _df['installQcut']

    decision_tree_pre_pruning, decision_tree_X_test, decision_tree_y_test, decision_tree_grid_search = (
        classifier_pipeline(DecisionTreeClassifier(random_state=42), decision_tree_pre_tuned_parameters, X, y))

    decision_tree_pre_pruning_metrics = classifier_metrics(decision_tree_pre_pruning,
                                                           decision_tree_y_test,
                                                           decision_tree_X_test,
                                                           decision_tree_grid_search)
    decision_tree_pre_pruning_fpr, decision_tree_pre_pruning_tpr = classifier_fpr_tpr(decision_tree_pre_pruning,
                                                                                      decision_tree_y_test,
                                                                                      decision_tree_X_test)
    output_performance_to_table(classifer_title="Decision Tree Pre-pruning",
                                classifier_metrics=decision_tree_pre_pruning_metrics,
                                master_table=master_table)
    # plot_roc_curve(ax[0, 0], decision_tree_pre_pruning_fpr, decision_tree_pre_pruning_tpr,
    #                decision_tree_pre_pruning_metrics.roc_auc, 'Decision Tree Pre-pruning')
    plot_roc_curve_individual(decision_tree_pre_pruning_fpr, decision_tree_pre_pruning_tpr,
                              decision_tree_pre_pruning_metrics.roc_auc, 'Decision Tree Pre-pruning')

    print("====================================")
    print("Decision Tree")
    print("Post-pruning")
    print("====================================")
    decision_tree_post_tuned_parameters = [{'ccp_alpha': [0.0, 0.01, 0.1, 0.2, 0.5, 1.0]}]
    decision_tree_post_pruning, decision_tree_X_test, decision_tree_y_test, decision_tree_post_grid_search = (
        classifier_pipeline(DecisionTreeClassifier(random_state=42), decision_tree_post_tuned_parameters, X, y))
    decision_tree_post_pruning_metrics = classifier_metrics(decision_tree_post_pruning,
                                                            decision_tree_y_test,
                                                            decision_tree_X_test,
                                                            decision_tree_post_grid_search)
    decision_tree_post_pruning_fpr, decision_tree_post_pruning_tpr = classifier_fpr_tpr(decision_tree_post_pruning,
                                                                                        decision_tree_y_test,
                                                                                        decision_tree_X_test)

    output_performance_to_table(classifer_title="Decision Tree Post-pruning",
                                classifier_metrics=decision_tree_post_pruning_metrics,
                                master_table=master_table)
    plot_roc_curve_individual(decision_tree_post_pruning_fpr,
                              decision_tree_post_pruning_tpr,
                              decision_tree_post_pruning_metrics.roc_auc,
                              'Decision Tree Post-pruning')
    # Standardize Data
    X = _df_standard.drop(columns=['installRange', 'installCount', 'box_cox_installs', 'installQcut'], axis=1)
    y = _df_standard['installQcut']

    print("====================================")
    print("Logistic Regression")
    print("====================================")

    logistic_regression_tuned_parameters = [{'penalty': ['l2'],
                                             'C': [0.1, 1, 10],
                                             'solver': ['lbfgs'],
                                             'max_iter': [100, 200, 300, 400, 500]}]
    logistic_regression, logistic_regression_X_test, logistic_regression_y_test, logistic_regression_grid_search = (
        classifier_pipeline(LogisticRegression(random_state=42), logistic_regression_tuned_parameters, X, y))
    logistic_regression_metrics = classifier_metrics(logistic_regression,
                                                     logistic_regression_y_test,
                                                     logistic_regression_X_test,
                                                     logistic_regression_grid_search)
    logistic_regression_fpr, logistic_regression_tpr = classifier_fpr_tpr(logistic_regression,
                                                                          logistic_regression_y_test,
                                                                          logistic_regression_X_test)
    output_performance_to_table(classifer_title="Logistic Regression",
                                classifier_metrics=logistic_regression_metrics,
                                master_table=master_table)
    plot_roc_curve_individual(
        logistic_regression_fpr,
        logistic_regression_tpr,
        logistic_regression_metrics.roc_auc,
        'Logistic Regression')

    print("====================================")
    print("K-Nearest Neighbors")
    print("====================================")

    knn_tuned_parameters = [{'n_neighbors': np.arange(1, 30, 1),
                             'weights': ['distance'],
                             'metric': ['manhattan', 'minkowski']}]
    knn, knn_X_test, knn_y_test, knn_grid_search = (
        classifier_pipeline(KNeighborsClassifier(), knn_tuned_parameters, X, y))
    knn_metrics = classifier_metrics(knn,
                                     knn_y_test,
                                     knn_X_test,
                                     knn_grid_search)
    knn_fpr, knn_tpr = classifier_fpr_tpr(knn,
                                          knn_y_test,
                                          knn_X_test)
    output_performance_to_table(classifer_title="K-Nearest Neighbors",
                                classifier_metrics=knn_metrics,
                                master_table=master_table)
    plot_roc_curve_individual(
        knn_fpr,
        knn_tpr,
        knn_metrics.roc_auc,
        'K-Nearest Neighbors')

    print("====================================")
    print("Supporting Vector Machine")
    print("====================================")

    svc_tuned_parameters = [{'kernel': ['rbf', 'linear', 'poly']}]
    svc, svc_X_test, svc_y_test, svc_grid_search = (
        classifier_pipeline(SVC(random_state=42, probability=True), svc_tuned_parameters, X, y))
    svc_metrics = classifier_metrics(svc,
                                     svc_y_test,
                                     svc_X_test,
                                     svc_grid_search)
    svc_fpr, svc_tpr = classifier_fpr_tpr(svc,
                                          svc_y_test,
                                          svc_X_test)
    output_performance_to_table(classifer_title="Supporting Vector Machine",
                                classifier_metrics=svc_metrics,
                                master_table=master_table)
    plot_roc_curve_individual(
        svc_fpr,
        svc_tpr,
        svc_metrics.roc_auc,
        'Supporting Vector Machine')

    print("====================================")
    print("Naive Bayes")
    print("====================================")

    naive_bayes_tuned_parameters = [{'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]}]
    naive_bayes, naive_bayes_X_test, naive_bayes_y_test, naive_bayes_grid_search = (
        classifier_pipeline(GaussianNB(), naive_bayes_tuned_parameters, X, y))
    naive_bayes_metrics = classifier_metrics(naive_bayes,
                                             naive_bayes_y_test,
                                             naive_bayes_X_test,
                                             naive_bayes_grid_search)
    naive_bayes_fpr, naive_bayes_tpr = classifier_fpr_tpr(naive_bayes,
                                                          naive_bayes_y_test,
                                                          naive_bayes_X_test)
    output_performance_to_table(classifer_title="Naive Bayes",
                                classifier_metrics=naive_bayes_metrics,
                                master_table=master_table)
    plot_roc_curve_individual(
        naive_bayes_fpr,
        naive_bayes_tpr,
        naive_bayes_metrics.roc_auc,
        'Naive Bayes')

    print("====================================")
    print("Ensemble Learning")
    print("Random Forest")
    print("====================================")

    # Best parameters set found on development set: {'criterion': 'gini', 'max_depth': 4, 'max_features': 10,
    # 'min_samples_leaf': 1, 'min_samples_split': 2, 'splitter': 'best'}
    random_forest_tuned_parameters = [{'n_estimators': [100, 500],
                                       'criterion': ['gini'],
                                       'max_depth': [4],
                                       'min_samples_split': [2],
                                       'min_samples_leaf': [1],
                                       'max_features': [10],
                                       'bootstrap': [True, False]}]
    random_forest, random_forest_X_test, random_forest_y_test, random_forest_grid_search = (
        classifier_pipeline(RandomForestClassifier(random_state=42), random_forest_tuned_parameters, X, y))
    random_forest_metrics = classifier_metrics(random_forest,
                                               random_forest_y_test,
                                               random_forest_X_test,
                                               random_forest_grid_search)
    random_forest_fpr, random_forest_tpr = classifier_fpr_tpr(random_forest,
                                                              random_forest_y_test,
                                                              random_forest_X_test)
    output_performance_to_table(classifer_title="Random Forest Bagging",
                                classifier_metrics=random_forest_metrics,
                                master_table=master_table)
    plot_roc_curve_individual(
        random_forest_fpr,
        random_forest_tpr,
        random_forest_metrics.roc_auc,
        'Random Forest Bagging')

    print("====================================")
    print("Ensemble Learning")
    print("Stacking")
    print("====================================")

    stacking_tuned_parameters = [{'n_jobs': [-1]}]
    stacking, stacking_X_test, stacking_y_test, stacking_grid_search = (
        classifier_pipeline(StackingClassifier(estimators=[('knn', knn),
                                                           ('svc', svc),
                                                           ('naive_bayes', naive_bayes)]),
                            stacking_tuned_parameters, X, y))
    stacking_metrics = classifier_metrics(stacking,
                                          stacking_y_test,
                                          stacking_X_test,
                                          stacking_grid_search)
    stacking_fpr, stacking_tpr = classifier_fpr_tpr(stacking,
                                                    stacking_y_test,
                                                    stacking_X_test)
    output_performance_to_table(classifer_title="Stacking",
                                classifier_metrics=stacking_metrics,
                                master_table=master_table)
    plot_roc_curve_individual(
        stacking_fpr,
        stacking_tpr,
        stacking_metrics.roc_auc,
        'Stacking')

    print("====================================")
    print("Ensemble Learning")
    print("Boosting")
    print("====================================")

    boosting_tuned_parameters = [{'n_estimators': [50, 200, 500],
                                  'learning_rate': [0.1, 0.5, 1.0]}]
    boosting, boosting_X_test, boosting_y_test, boosting_grid_search = (
        classifier_pipeline(AdaBoostClassifier(random_state=42), boosting_tuned_parameters, X, y))
    boosting_metrics = classifier_metrics(boosting,
                                          boosting_y_test,
                                          boosting_X_test,
                                          boosting_grid_search)
    boosting_fpr, boosting_tpr = classifier_fpr_tpr(boosting,
                                                    boosting_y_test,
                                                    boosting_X_test)
    output_performance_to_table(classifer_title="Boosting",
                                classifier_metrics=boosting_metrics,
                                master_table=master_table)
    plot_roc_curve_individual(
        boosting_fpr,
        boosting_tpr,
        boosting_metrics.roc_auc,
        'Boosting')

    print("====================================")
    print("Neural Network")
    print("====================================")

    neural_network_tuned_parameters = [{'hidden_layer_sizes': [(100,), (100, 100)],
                                        'activation': ['relu'],
                                        'solver': ['adam'],
                                        'learning_rate': ['constant']}]
    neural_network, neural_network_X_test, neural_network_y_test, neural_network_grid_search = (
        classifier_pipeline(MLPClassifier(random_state=42), neural_network_tuned_parameters, X, y))
    neural_network_metrics = classifier_metrics(neural_network,
                                                neural_network_y_test,
                                                neural_network_X_test,
                                                neural_network_grid_search)
    neural_network_fpr, neural_network_tpr = classifier_fpr_tpr(neural_network,
                                                                neural_network_y_test,
                                                                neural_network_X_test)
    output_performance_to_table(classifer_title="Neural Network",
                                classifier_metrics=neural_network_metrics,
                                master_table=master_table)
    plt.savefig('plots/master_roc_curve.png')
    plt.show()
    plot_roc_curve_individual(neural_network_fpr, neural_network_tpr, neural_network_metrics.roc_auc, 'Neural Network')
    # plot_roc_curve_plt(neural_network_fpr, neural_network_tpr, neural_network_metrics.roc_auc, 'Neural Network')
    elbow_method_knn(X, y)
    optimize_cost_complexity_pruning_alpha(X, y)
    print(master_table)
    master_table_latex = master_table.get_latex_string()
    classifier_metrics_list = [decision_tree_pre_pruning_metrics, decision_tree_post_pruning_metrics,
                               logistic_regression_metrics, knn_metrics, svc_metrics, naive_bayes_metrics,
                               random_forest_metrics, stacking_metrics, boosting_metrics, neural_network_metrics]

    fig, ax = plt.subplots(2, 5, figsize=(30, 15))
    plot_roc_curve(ax[0, 0], decision_tree_pre_pruning_fpr, decision_tree_pre_pruning_tpr,
                   decision_tree_pre_pruning_metrics.roc_auc, 'Decision Tree Pre-pruning')
    plot_roc_curve(ax[0, 1],
                   decision_tree_post_pruning_fpr,
                   decision_tree_post_pruning_tpr,
                   decision_tree_post_pruning_metrics.roc_auc,
                   'Decision Tree Post-pruning')
    plot_roc_curve(ax[0, 2],
                   logistic_regression_fpr,
                   logistic_regression_tpr,
                   logistic_regression_metrics.roc_auc,
                   'Logistic Regression')
    plot_roc_curve(ax[0, 3],
                   knn_fpr,
                   knn_tpr,
                   knn_metrics.roc_auc,
                   'K-Nearest Neighbors')
    plot_roc_curve(ax[0, 4],
                   svc_fpr,
                   svc_tpr,
                   svc_metrics.roc_auc,
                   'Supporting Vector Machine')
    plot_roc_curve(ax[1, 0],
                   naive_bayes_fpr,
                   naive_bayes_tpr,
                   naive_bayes_metrics.roc_auc,
                   'Naive Bayes')
    plot_roc_curve(ax[1, 1],
                   random_forest_fpr,
                   random_forest_tpr,
                   random_forest_metrics.roc_auc,
                   'Random Forest Bagging')
    plot_roc_curve(ax[1, 2],
                   stacking_fpr,
                   stacking_tpr,
                   stacking_metrics.roc_auc,
                   'Stacking')
    plot_roc_curve(ax[1, 3],
                   boosting_fpr,
                   boosting_tpr,
                   boosting_metrics.roc_auc,
                   'Boosting')
    plot_roc_curve(ax[1, 4],
                   neural_network_fpr,
                   neural_network_tpr,
                   neural_network_metrics.roc_auc,
                   'Neural Network')
    plt.savefig('plots/master_roc_curve_new.png', dpi= 300)
    plt.show()



    return master_table, master_table_latex, classifier_metrics_list


def optimize_k(x, kmax):
    sse = []
    sil = []
    for k in range(2, kmax + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10).fit(x)
        labels = kmeans.labels_
        sil.append(silhouette_score(x, labels, metric='euclidean'))
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(x)
        curr_sse = 0

        for i in range(len(x)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += distance.euclidean(x.iloc[i], curr_center)

        sse.append(curr_sse)
    return sse, sil


def plot_pca_clusters(X, clusters, model, title, cluster_ids_to_plot=None):
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(X)

    plt.figure()

    unique_clusters = set(clusters)
    if cluster_ids_to_plot is not None:
        clusters_to_plot = set(cluster_ids_to_plot)
    else:
        clusters_to_plot = unique_clusters

    for cluster_id in clusters_to_plot:
        if cluster_id in unique_clusters:  # Check if cluster exists
            plt.scatter(df_pca[clusters == cluster_id, 0], df_pca[clusters == cluster_id, 1],
                        label=f'Cluster {cluster_id + 1}')

    if model is not None and hasattr(model, 'cluster_centers_'):
        pca_centroids = pca.transform(model.cluster_centers_)
        plt.scatter(pca_centroids[:, 0], pca_centroids[:, 1], s=50, c='black', label='Centroids', marker='X')

    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('plots/{}.png'.format(title), dpi= 300)
    plt.show()


def plot_elbow_method(sse, kmax):
    plt.figure()
    plt.plot(np.arange(2, kmax + 1, 1), sse)
    plt.xticks(np.arange(2, kmax + 1, 1))
    plt.grid()
    plt.xlabel('k')
    plt.ylabel('WSS')
    plt.title('K selection in K-means++ - Elbow Method')
    plt.tight_layout()
    plt.savefig('plots/kmeans_elbow_method.png', dpi= 300)
    plt.show()


def plot_silhouette_method(sil, kmax):
    plt.figure()
    plt.plot(np.arange(2, kmax + 1, 1), sil, 'x-')
    plt.xticks(np.arange(2, kmax + 1, 1))
    plt.grid()
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('K selection in K-means++ - Silhouette Method')
    plt.tight_layout()
    plt.savefig('plots/kmeans_silhouette_method.png', dpi= 300)
    plt.show()


def clustering(outer_df):
    _df = outer_df.copy()
    X = _df.drop(columns=['installCount', 'box_cox_installs', 'installQcut', 'installRange'])
    results_table = PrettyTable()
    k_max = 10
    sse, sil = optimize_k(X, k_max)
    results = pd.DataFrame({'k': np.arange(2, k_max + 1, 1), 'sse': sse, 'sil': sil})
    if not os.path.exists('output'):
        os.makedirs('output')
        results.to_csv('output/kmeans_optimization.csv', index=False)
    else:
        results.to_csv('output/kmeans_optimization.csv', index=False)
    if os.path.exists('output/kmeans_optimization.csv'):
        results = pd.read_csv('output/kmeans_optimization.csv')
    results_table.field_names = ['k', 'SSE', 'Silhouette Score']
    results_table.float_format = '.3'
    results_table.align["k"] = "r"
    results_table.align["SSE"] = "r"
    results_table.align["Silhouette Score"] = "r"
    for index, row in results.iterrows():
        results_table.add_row([row['k'], row['sse'], row['sil']])
    print(results_table)
    sse, sil = results['sse'].values, results['sil'].values

    plot_elbow_method(sse, k_max)
    plot_silhouette_method(sil, k_max)

    print("=========================================")
    print("K-means++")
    print("Output: K-means++ Cluster Visualization by PCA")
    print("=========================================")

    n_clusters = 7
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10)
    clusters = kmeans.fit_predict(X)
    plot_pca_clusters(X, clusters, kmeans, 'Cluster Visualization with PCA - K-means++')

    print("=========================================")
    print("DBSCAN")
    print("Output: DBSCAN Cluster Visualization by PCA")
    print("=========================================")

    # For DBSCAN (top 10 clusters)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(X)
    cluster_counts = Counter(clusters)
    top_clusters = [cluster[0] for cluster in cluster_counts.most_common(10) if cluster[0] != -1]
    plot_pca_clusters(X, clusters, None, 'Cluster Visualization with PCA - DBSCAN - Top 10 Clusters', top_clusters)
    return results_table


def association_rule(df):
    print("=========================================")
    print("Apriori")
    print("Association Rules")
    print("=========================================")

    apriori_result_table = PrettyTable()
    apriori_result_table.field_names = ["Support", "Itemsets"]
    apriori_result_table.float_format = ".3"
    apriori_result_table.align["Itemsets"] = "r"
    association_rule_table = PrettyTable()
    association_rule_table.field_names = ["Antecedents", "Consequents", "Support", "Confidence", "Lift"]
    association_rule_table.float_format = ".3"
    association_rule_table.align["Antecedents"] = "r"

    print("=========================================")
    print("Prepare dataset for Apriori algorithm")
    print("=========================================")

    _df = df.copy()
    _df['rating_High'] = _df['rating'] >= 4.5
    _df['appAgeInDays_Old'] = _df['appAgeInDays'] >= 365
    _df['lastUpdateAgeInDays_Old'] = _df['lastUpdateAgeInDays'] >= 365
    _df['sizeInMB_Large'] = _df['sizeInMB'] >= 100
    _df.drop(columns=['rating', 'appAgeInDays', 'sizeInMB', 'ratingCount', 'lastUpdateAgeInDays', 'minAndroidVersion_7',
                      'minAndroidVersion_8', 'installQcut', 'installCount', 'installRange', 'box_cox_installs'],
             inplace=True)
    _df.replace({True: 1, False: 0}, inplace=True)

    print("=========================================")
    print("Run Apriori algorithm")
    print("=========================================")

    apriori_result = apriori(_df, min_support=0.01, use_colnames=True)
    for index, row in apriori_result.iterrows():
        print(set(row['itemsets']))
        apriori_result_table.add_row([row['support'], set(row['itemsets'])])

    rules = association_rules(apriori_result, metric="lift", min_threshold=0.6)
    rules.sort_values(by=['confidence'], ascending=[False], inplace=True)
    for index, row in rules.iterrows():
        if row['confidence'] > 0.5:
            association_rule_table.add_row([set(row['antecedents']), set(row['consequents']), row['support'],
                                            row['confidence'], row['lift']])

    # rules.to_csv('output/association_rules.csv', index=False)
    print(apriori_result_table)
    print(association_rule_table)
    return apriori_result_table, association_rule_table

def save_table_to_latex_file(table, filename):
    """Saves a PrettyTable object as a LaTeX table to a text file."""
    with open(filename, 'w') as file:
        file.write(table.get_latex_string())

if __name__ == '__main__':
    # Calculate run time
    start_time = time.time()
    df, df_standard = preprocessing()
    # df = pd.read_csv('output/preprocessed.csv')
    # df_standard = pd.read_csv('output/preprocessed_standard.csv')
    backwise_table, backwise_ol_summary, random_forest_table = regression(df_standard)
    regression_time_interval = time.time() - start_time
    classification_start_time = time.time()
    classification_master_table, classification_master_table_latex, classifier_metrics_list = classification(df,
                                                                                                             df_standard)

    classification_time = time.time() - classification_start_time
    clustering_start_time = time.time()
    kmeans_results_table = clustering(df_standard)
    clustering_time = time.time() - clustering_start_time
    association_rule_start_time = time.time()
    apriori_result_table, association_rule_table = association_rule(df)
    association_rule_time = time.time() - association_rule_start_time

    os.makedirs('output', exist_ok=True)
    save_table_to_latex_file(backwise_table, 'output/backwise_table.txt')
    save_table_to_latex_file(random_forest_table, 'output/random_forest_table.txt')
    save_table_to_latex_file(classification_master_table, 'output/classification_master_table.txt')
    save_table_to_latex_file(kmeans_results_table, 'output/kmeans_results_table.txt')
    save_table_to_latex_file(apriori_result_table, 'output/apriori_result_table.txt')
    save_table_to_latex_file(association_rule_table, 'output/association_rule_table.txt')

    print("=========================================")
    print("Total Runtime: %s seconds" % (time.time() - start_time))
    print("=========================================")
    print("Regression Runtime: %s seconds" % regression_time_interval)
    print("Classification Runtime: %s seconds" % classification_time)
    print("Clustering Runtime: %s seconds" % clustering_time)
    print("Association Rule Runtime: %s seconds" % association_rule_time)
    print("=========================================")

