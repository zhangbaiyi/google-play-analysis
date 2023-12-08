from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer, PredictionErrorDisplay
import statsmodels.api as sm
from prettytable import PrettyTable


def reverse_standardize(x):
    return x * installs_std + installs_mean


def read_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.drop(columns=['installRange', 'box_cox_installs', 'installQcut'], inplace=True)
    df['installCount'] = StandardScaler().fit_transform(df[['installCount']])
    return df


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
    print(model.summary().as_latex())

    # alternatively
    for _table in model.summary().tables:
        print(_table.as_latex_tabular())


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
    plt.figure()
    plt.title("Feature Importances")
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.grid()
    plt.savefig('plots/regressor_feature_importances.png', dpi=300)
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

    n_features = _X_test.shape[1]
    n_samples = _X_test.shape[0]

    testing_r2 = _rf.score(_X_test, _y_test)
    testing_mse = mean_squared_error(_y_test, _rf.predict(_X_test))

    adjusted_r2 = 1 - (1 - testing_r2) * (n_samples - 1) / (n_samples - n_features - 1)

    resid = _y_test - _rf.predict(_X_test)
    sse = np.sum(resid ** 2)
    aic = 2 * n_features + n_samples * np.log(sse / n_samples)
    bic = n_features * np.log(n_samples) + n_samples * np.log(sse / n_samples)

    plt.figure()
    display = PredictionErrorDisplay.from_predictions(
        y_true=_y_test,
        y_pred=_rf.predict(_X_test),
        kind="actual_vs_predicted",
        ax=plt.gca(),
        scatter_kwargs={"alpha": 0.2, "color": "tab:blue"},
        line_kwargs={"color": "tab:red"},
    )
    plt.title(f"Random Forest Regression")
    R_squared = _rf.score(_X_test, _y_test)
    plt.plot([], [], " ", label=f"R_Squared: {R_squared:.2f}")
    plt.legend(loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.savefig('plots/regressor_prediction_error.png', dpi=300)

    table.title = "Random Forest Regression - Testing Results"
    table.field_names = ['R_Squared', 'Adj R^2', 'AIC', 'BIC', 'MSE']
    table.add_row([testing_r2, adjusted_r2, aic, bic, testing_mse])

    print(table)
    return _rf, table, scores, _rf.predict(_X_test)


def plot_results(_results_df, _scores):
    _results_df['Actual'] = _results_df['Actual'].apply(reverse_standardize)
    _results_df['Predicted'] = _results_df['Predicted'].apply(reverse_standardize)
    _results_df['Predicted Upper'] = _results_df['Predicted'] + 1.96 * np.sqrt(_scores.mean())
    _results_df['Predicted Lower'] = _results_df['Predicted'] - 1.96 * np.sqrt(_scores.mean())
    _results_df['Actual'].plot(kind='line', label='Actual')
    _results_df['Predicted'].plot(kind='line', label='Predicted')
    plt.figure()
    results_df_head = _results_df.head(100)
    results_df_head['Actual'].plot(kind='line', label='Actual')
    results_df_head['Predicted'].plot(kind='line', label='Predicted')
    plt.fill_between(range(len(results_df_head['Predicted'])), results_df_head['Predicted Lower'],
                     results_df_head['Predicted Upper'], color='orange', alpha=0.4)
    plt.xlabel("# of Samples")
    plt.ylabel("Values")
    plt.title("Predicted vs Actual with Confidence Interval - First 100 Samples")
    plt.tight_layout()
    plt.grid()
    plt.legend()
    plt.savefig('plots/regressor_predicted_vs_actual.png', dpi=300)
    plt.show()


df = read_and_preprocess_data('output/preprocessed_standard.csv')
X = df.drop(columns=['installCount'])
installs_mean = df['installCount'].mean()
installs_std = df['installCount'].std()
y = df['installCount']
X = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regression_table, ols_summary = backward_stepwise_regression(X_train, y_train)

rf, random_forest_results_table, kfold_scores, y_pred = train_random_forest(X, y)

results_df = pd.DataFrame({'Actual': y_test.reset_index(drop=True), 'Predicted': y_pred})
plot_results(results_df, kfold_scores)

print(regression_table)
print(ols_summary)
print(random_forest_results_table)

