import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from prettytable import PrettyTable



print("=========================================")
print("Phase 2: Regression")
print("=========================================")
print("Reading preprocessed dataset...")

df = pd.read_csv('output/preprocessed_standard.csv')
df.drop(columns=['installRange', 'box_cox_installs', 'installQcut'], inplace=True)
installs_mean = df['installCount'].mean()
installs_std = df['installCount'].std()
df['installCount'] = StandardScaler().fit_transform(df[['installCount']])

print("=========================================")
print("Splitting dataset into train and test set...")
print("=========================================")
X = df.drop(columns=['installCount'])
y = df['installCount']
X = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=========================================")
print("Model training...")
print("=========================================")
model = sm.OLS(y_train, X_train).fit()

regression_table = PrettyTable()
regression_table.title = "Backward Stepwise Regression"
regression_table.field_names = ['Attempt', 'Removed Feature', 'Removed p-value', 'AIC', 'BIC', 'R^2', 'Adjusted R^2', 'MSE']
regression_table.float_format = '.3'
model = sm.OLS(y_train, X_train).fit()
print(model.summary())
regression_table.add_row([0, 'N/A', 'N/A', model.aic, model.bic, model.rsquared, model.rsquared_adj, model.mse_model])
print(regression_table)

print(model.summary())
removed_p_value = model.pvalues.max()
X_train.drop(['category_Productivity'], axis=1, inplace=True)
model = sm.OLS(y_train, X_train).fit()
regression_table.add_row([1, 'category_Productivity', removed_p_value, model.aic, model.bic, model.rsquared, model.rsquared_adj, model.mse_model])
print(regression_table)

print(model.summary())
removed_p_value = model.pvalues.max()
X_train.drop(['minAndroidVersion_8'], axis=1, inplace=True)
model = sm.OLS(y_train, X_train).fit()
regression_table.add_row([1, 'minAndroidVersion_8', removed_p_value, model.aic, model.bic, model.rsquared, model.rsquared_adj, model.mse_model])
print(regression_table)

print(model.summary())
removed_p_value = model.pvalues.max()
X_train.drop(['category_Social'], axis=1, inplace=True)
model = sm.OLS(y_train, X_train).fit()
regression_table.add_row([1, 'category_Social', removed_p_value, model.aic, model.bic, model.rsquared, model.rsquared_adj, model.mse_model])
print(regression_table)

print("=========================================")
print("Analysis")
print("=========================================")

X = df.drop(columns=['installCount'])
y = df['installCount']
X = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
features = X.columns
importances = rf.feature_importances_
indices = np.argsort(importances)[-13:]
plt.figure(figsize=(12, 8))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.show()

feature_importance_df = pd.DataFrame({'feature': features, 'importance': importances})
threshold = 0.05
rf_important_features = feature_importance_df[feature_importance_df['importance'] > threshold]['feature']
print(rf_important_features)
X = df[rf_important_features]
y = df['installCount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf.fit(X_train, y_train)
print("=========================================")
print("Random Forest Regression")
print("=========================================")
print("Training R^2: ", rf.score(X_train, y_train))
print("Testing R^2: ", rf.score(X_test, y_test))
print("Training MSE: ", np.mean((rf.predict(X_train) - y_train) ** 2))
print("Testing MSE: ", np.mean((rf.predict(X_test) - y_test) ** 2))

print("=========================================")
print("Overfitting - K-fold Cross Validation")
print("=========================================")
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
scores = cross_val_score(rf, X_train, y_train, cv=kfold, scoring=make_scorer(mean_squared_error))
print("MSE for each fold: ", scores)
print("Average MSE: ", scores.mean())


print("=========================================")
print("Confidence Interval")
print("=========================================")
plt.figure(figsize=(100, 15))
plt.title("Predicted vs Actual")
results_df = pd.DataFrame({'Actual': y_test.reset_index(drop=True), 'Predicted': rf.predict(X_test)})
def reverse_standardize(x):
    return x * installs_std + installs_mean
results_df['Actual'] = results_df['Actual'].apply(reverse_standardize)
results_df['Predicted'] = results_df['Predicted'].apply(reverse_standardize)
results_df['Predicted Upper'] = results_df['Predicted'] + 1.96 * np.sqrt(scores.mean())
results_df['Predicted Lower'] = results_df['Predicted'] - 1.96 * np.sqrt(scores.mean())
results_df['Actual'].plot(kind='line', label='Actual')
results_df['Predicted'].plot(kind='line', label='Predicted')
plt.xlabel("# of Samples")
plt.ylabel("Installs")
plt.grid()
plt.tight_layout()
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
results_df_head =results_df.head(100)
results_df_head['Actual'].plot(kind='line', label='Actual')
results_df_head['Predicted'].plot(kind='line', label='Predicted')
plt.fill_between(range(len(results_df_head['Predicted'])), results_df_head['Predicted Lower'], results_df_head['Predicted Upper'], color='black', alpha=1)
plt.xlabel("# of Samples")
plt.ylabel("Installs")
plt.grid()
plt.tight_layout()
plt.legend()
plt.title("Predicted vs Actual for First 100 Samples")
plt.show()




