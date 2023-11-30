import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv('output/preprocessed.csv')

fig, ax = plt.subplots(figsize=(10, 10))


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


def classifier_pipeline(classifier, tuned_parameters, X, y, test_size=0.2, random_state=42, k_fold=5, scoring='accuracy'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    grid_search = GridSearchCV(classifier, tuned_parameters, cv=k_fold, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    dt_pre_pruning = grid_search.best_estimator_
    print("Best parameters set found on development set:")
    print(grid_search.best_params_)
    print("Grid scores on development set:")
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    



print("====================================")
print("Decision Tree")
print("Pre-pruning")
print("====================================")
tuned_parameters = [{'max_depth': [1, 4, 10],
                     'min_samples_split': [2, 5, 10],
                     'min_samples_leaf': [1, 4],
                     'max_features': [1, 5, 10, 'sqrt', 'log2'],
                     'splitter': ['best', 'random'],
                     'criterion': ['gini', 'entropy']}]

X = df.drop(columns=['installRange', 'installCount', 'box_cox_installs', 'installQcut'], axis=1)
y = df['installQcut']

classifier_pipeline(DecisionTreeClassifier(), tuned_parameters, X, y)

print("Detailed classification report:")
# y_true, y_pred = y_test, dt_pre_pruning.predict(X_test)
# dt_pre_pruning_classification_report = classification_report(y_true, y_pred)
# dt_pre_pruning_accuracy = accuracy_score(y_true, y_pred)
# dt_pre_pruning_confusion_matrix = confusion_matrix(y_true, y_pred)
# dt_pre_pruning_precision = dt_pre_pruning_confusion_matrix[1, 1] / (dt_pre_pruning_confusion_matrix[1, 1] + dt_pre_pruning_confusion_matrix[0, 1])
# dt_pre_pruning_recall = dt_pre_pruning_confusion_matrix[1, 1] / (dt_pre_pruning_confusion_matrix[1, 1] + dt_pre_pruning_confusion_matrix[1, 0])
# dt_pre_pruning_specificity = dt_pre_pruning_confusion_matrix[0, 0] / (dt_pre_pruning_confusion_matrix[0, 0] + dt_pre_pruning_confusion_matrix[0, 1])
# dt_pre_pruning_f1_score = f1_score(y_true, y_pred)
# dt_pre_pruning_roc_frp, dt_pre_pruning_roc_tpr, dt_pre_pruning_roc_thresholds = roc_curve(y_true, y_pred)
# dt_pre_pruning_roc_auc = roc_auc_score(y_true, y_pred)
# dt_pre_pruning_k_fold = grid_search.cv_results_


# print("====================================")
# print("Decision Tree")
# print("Post-pruning")
# print("====================================")
# tuned_parameters = [{'ccp_alpha':[0.0,0.1,0.2,0.3,0.4,0.5]}]
#
# grid_search_post_prunning = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=5, scoring='accuracy')
# grid_search_post_prunning.fit(X_train, y_train)
# dt_post_pruning = grid_search.best_estimator_
# print("Best parameters set found on development set:")
# print(grid_search_post_prunning.best_params_)
# print("Grid scores on development set:")
# means = grid_search_post_prunning.cv_results_['mean_test_score']
# stds = grid_search_post_prunning.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, grid_search_post_prunning.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r" %(mean, std*2, params))
# print("Detailed classification report:")
# y_true, y_pred = y_test, dt_post_pruning.predict(X_test)
# print(classification_report(y_true, y_pred))
# print("Accuracy: ", accuracy_score(y_true, y_pred))
# print("Confusion Matrix: ")
# print(confusion_matrix(y_true, y_pred))
