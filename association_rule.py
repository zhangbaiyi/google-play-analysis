from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd
from prettytable import PrettyTable
import warnings

warnings.simplefilter("ignore", DeprecationWarning)
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

df = pd.read_csv('output/preprocessed.csv')
df['rating_High'] = df['rating'] >= 4.5
df['appAgeInDays_Old'] = df['appAgeInDays'] >= 365
df['lastUpdateAgeInDays_Old'] = df['lastUpdateAgeInDays'] >= 365
df['sizeInMB_Large'] = df['sizeInMB'] >= 100
df.drop(columns=['rating', 'appAgeInDays', 'sizeInMB', 'ratingCount', 'lastUpdateAgeInDays', 'minAndroidVersion_7',
                 'minAndroidVersion_8', 'installQcut', 'installCount', 'installRange', 'box_cox_installs'],
        inplace=True)
df.replace({True: 1, False: 0}, inplace=True)

print("=========================================")
print("Run Apriori algorithm")
print("=========================================")

apriori_result = apriori(df, min_support=0.01, use_colnames=True)
for index, row in apriori_result.iterrows():
    print(set(row['itemsets']))
    apriori_result_table.add_row([row['support'], set(row['itemsets'])])

rules = association_rules(apriori_result, metric="lift", min_threshold=0.6)
rules.sort_values(by=['confidence'], ascending=[False], inplace=True)
for index, row in rules.iterrows():
    if row['confidence'] > 0.5:
        association_rule_table.add_row([set(row['antecedents']), set(row['consequents']), row['support'],
                                        row['confidence'],row['lift']])

# rules.to_csv('output/association_rules.csv', index=False)
print(apriori_result_table)
print(association_rule_table)
