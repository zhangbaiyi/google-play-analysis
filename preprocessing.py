# Google Play Dataset - Preprocessing

import pandas as pd

df = pd.read_csv('Google-Playstore.csv')

from prettytable import PrettyTable
import random
import time

random.seed(time.time())

feature_outlook = PrettyTable()
feature_outlook.field_names = ["Feature", "Type", "N/A Count", "N/A Percentage", "Cat/Num", "Unique Count",
                               "Unique Percentage", "Example"]


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


print("Phase I - Data cleaning")
print("Raw data overview")
for col in df.columns:
    feature_outlook.add_row(
        [col, df[col].dtype, df[col].isna().sum(), check_na_percentage(col), judge_cat_num(col), len(df[col].unique()),
         check_unique_percentage(col), df[col].unique()[random.randint(0, len(df[col].unique()) - 1)]])
print(feature_outlook)

print("Phase I - Data cleaning")
print("NA values percentage")
import matplotlib.pyplot as plt

na_feature_percentage = df.isna().sum().sort_values(ascending=False) / len(df) * 100
na_feature_percentage = na_feature_percentage[na_feature_percentage > 0]
plt.figure(figsize=(10, 5))
na_feature_percentage.plot(kind='barh')
plt.xlabel('Percentage (%)')
plt.ylabel('Features')
plt.title('Percentage of N/A Values')
plt.grid()
plt.show()

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
# Drop 'Minimum Installs'
df.drop(columns=['Minimum Installs'], inplace=True)

print("Phase I - Aggregation")
print("Aggregation of Android versions.")
# Minimum Android Version
# Replace ' and up' and ' - ' in the entire column
df['Minimum Android'] = df['Minimum Android'].str.replace(' and up', '').str.split(' - ').str.get(0)

