import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from prettytable import PrettyTable
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('output/preprocessed.csv')
df_standard = pd.read_csv('output/preprocessed_standard.csv')

fig, ax = plt.subplots(3, 3, figsize=(20, 15))
master_table = PrettyTable()
master_table.title = "Classifier Performance"
master_table.float_format = ".3"
master_table.field_names = ["Classifier", "Confusion Matrix", "Precision", "Recall", "Specificity", "F1 Score", "AUC"]


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
    plt.figure(figsize=(10, 10))
    X_train, X_test, y_train, y_test = train_test_split(my_X, my_y, test_size=0.2, random_state=42,
                                                        stratify=y)
    error_rate = []
    for i in range(1, 30):
        scope_knn = KNeighborsClassifier(n_neighbors=i, metric='manhattan', weights='distance')
        scope_knn.fit(X_train, y_train)
        pred = scope_knn.predict(X_test)
        error_rate.append(np.mean(pred != y_test))
    plt.plot(range(1, 30, 1), error_rate, marker='o', markersize=9)
    plt.ylabel('Error Rate')
    plt.xlabel('k')
    plt.xticks(np.arange(1, 30, 1))
    plt.title('Error Rate vs. K Value - (Elbow Method)')
    plt.show()


def optimize_cost_complexity_pruning_alpha(my_X, my_y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(my_X, my_y, test_size=0.2, random_state=random_state,
                                                        stratify=y)
    ccp_alphas = np.linspace(0.0, 0.5, 50)
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)
    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker='o', label="train",
            drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker='o', label="test",
            drawstyle="steps-post")
    ax.legend()
    plt.tight_layout()
    plt.savefig('plots/decision_tree_optimize_cost_complexity_pruning_alpha.png', dpi=300)
    plt.show()


def output_performance_to_table(classifer_title, classifier_metrics):
    master_table.add_row([classifer_title,
                          classifier_metrics.confusion_matrix,
                          classifier_metrics.precision,
                          classifier_metrics.recall,
                          classifier_metrics.specificity,
                          classifier_metrics.f1_score,
                          classifier_metrics.roc_auc])


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


def make_individual_table(classifer_title, classifier_metrics):
    table = PrettyTable()
    table.title = classifer_title
    table.float_format = ".3"
    table.field_names = ["Classifier", "Confusion Matrix", "Precision", "Recall", "Specificity", "F1", "AUC"]
    table.add_row([classifer_title,
                   classifier_metrics.confusion_matrix,
                   classifier_metrics.precision,
                   classifier_metrics.recall,
                   classifier_metrics.specificity,
                   classifier_metrics.f1_score,
                   classifier_metrics.roc_auc])
    return table

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

X = df.drop(columns=['installRange', 'installCount', 'box_cox_installs', 'installQcut'], axis=1)
y = df['installQcut']

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
                            classifier_metrics=decision_tree_pre_pruning_metrics)
plot_roc_curve(ax[0, 0], decision_tree_pre_pruning_fpr, decision_tree_pre_pruning_tpr,
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
                            classifier_metrics=decision_tree_post_pruning_metrics)
plot_roc_curve(ax[0, 1],
               decision_tree_post_pruning_fpr,
               decision_tree_post_pruning_tpr,
               decision_tree_post_pruning_metrics.roc_auc,
               'Decision Tree Post-pruning')

# Optimize Cost Complexity Pruning Alpha

# Standardize Data
X = df_standard.drop(columns=['installRange', 'installCount', 'box_cox_installs', 'installQcut'], axis=1)
y = df_standard['installQcut']

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
                            classifier_metrics=logistic_regression_metrics)
plot_roc_curve(ax[0, 2],
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
                            classifier_metrics=knn_metrics)
plot_roc_curve(ax[1, 0],
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
                            classifier_metrics=svc_metrics)
plot_roc_curve(ax[1, 1],
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
                            classifier_metrics=naive_bayes_metrics)
plot_roc_curve(ax[1, 2],
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
                            classifier_metrics=random_forest_metrics)
plot_roc_curve(ax[2, 0],
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
                            classifier_metrics=stacking_metrics)
plot_roc_curve(ax[2, 1],
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
                            classifier_metrics=boosting_metrics)
plot_roc_curve(ax[2, 2],
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
                            classifier_metrics=neural_network_metrics)
plt.show()

plot_roc_curve_plt(neural_network_fpr, neural_network_tpr, neural_network_metrics.roc_auc, 'Neural Network')
elbow_method_knn(X, y)
optimize_cost_complexity_pruning_alpha(X, y)

print(master_table)

master_table_latex = master_table.get_latex_string()
