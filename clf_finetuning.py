import pandas as pd
import numpy as np
import pickle
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


from LoreSA.datamanager import prepare_dataset
#from Tabular.xailib.explainers.lore_explainer import LoreTabularExplainer
from LoreSA.sklearn_classifier_wrapper import sklearn_classifier_wrapper
from LoreSA.util import get_df_stats

import matplotlib.pyplot as plt
from tabulate import tabulate

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

# Prepare data
dataset_path = 'Datasets/'
source_file_dict = {'compas' : dataset_path + 'compas_clean.csv',
                    'adult' : dataset_path + 'adult_clean.csv',
                    'german_credit' : dataset_path + 'german_credit.csv',
                    'wine' : dataset_path + 'winequality_white.csv',
                    'student': dataset_path + 'students.csv',
                    'abalone': dataset_path + 'abalone_3_class.csv'
                    }

class_field_dict = {'compas' : 'did_recid',
                    'adult' : 'class',
                    'german_credit' : 'default',
                    'wine' : 'quality',
                    'student': 'Target',
                    'abalone': 'RingsRange'
                    }

# Load and transform dataset 
# Binary
compas_df = pd.read_csv(source_file_dict['compas'], skipinitialspace=True, na_values='?', keep_default_na=True)
adult_df = pd.read_csv(source_file_dict['adult'], skipinitialspace=True, na_values='?', keep_default_na=True)
german_df = pd.read_csv(source_file_dict['german_credit'], skipinitialspace=True, na_values='?', keep_default_na=True)
# Multiclass
wine_df = pd.read_csv(source_file_dict['wine'], sep=';', skipinitialspace=True, na_values='?', keep_default_na=True)
students_df = pd.read_csv(source_file_dict['student'], sep=';', skipinitialspace=True, na_values='?', keep_default_na=True)
abalone_df = pd.read_csv(source_file_dict['abalone'], skipinitialspace=True, na_values='?', keep_default_na=True)


df_dict = {'compas' : compas_df,
            'adult' : adult_df,
            'german_credit' : german_df,
            'wine' : wine_df,
            'student': students_df,
            'abalone': abalone_df
            }

# Collect statistics about datasets
stats_dict = dict()
for k, v in df_dict.items():
    stats = get_df_stats(v, class_field_dict[k])
    stats_dict[k] = stats

stats_dict = pd.DataFrame(stats_dict)
stats_dict.to_csv('dataset_statistics.csv')
#tabulate(s, headers = 'keys', tablefmt = 'plain')

# Trasnform datasets
df_dict_new = dict()
for k, v in df_dict.items():
    df, feature_names, class_values, numeric_columns, categorical_columns, rdf, real_feature_names, features_map = prepare_dataset(v, class_field_dict[k], 'onehot')
    df_dict_new[k] = [df, feature_names, class_values, numeric_columns, categorical_columns, rdf, real_feature_names, features_map]


rf_param_grid = {
    'n_estimators': [25, 50, 100, 150],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [3, 6, 9],
    'max_leaf_nodes': [3, 6, 9],
    'min_samples_split': [2, 4, 6]
}

svm_param_grid = {
    'C': [0.1, 1, 10, 100], 
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear', 'poly']
    } 

mlp_param_grid = {
    'hidden_layer_sizes': [(150,100,50), (120,80,40), (100,50,30)],
    'max_iter': [50, 100, 150],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

tabnet_param_grid = {
    'n_d' : [8, 16, 24], 
    'n_a' : [8, 16, 24],
    'n_steps' : [3, 4, 5], 
    'gamma' : [1.2, 1.3, 1.5],
    'optimizer_params' : [dict(lr=x) for x in [0.015, 0.02, 0.025]],
    'scheduler_params' : [dict(gamma=x, step_size=50) for x in [0.85, 0.9, 0.95]]
}

xgb_param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

def hyperparameter_tuning(clf, params, X, y):
    # define evaluation
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=123)
    # define search
    rnd_search = RandomizedSearchCV(clf, params, n_iter=50, scoring='accuracy', n_jobs=-1, cv=cv, refit=True, verbose= 2, random_state=123)
    # execute search
    result = rnd_search.fit(X, y)
    # summarize result
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)
    return result.best_estimator_, result.best_params_


clf_dict = {
    'RandomForest' : [RandomForestClassifier(), rf_param_grid],
    'SVC' : [SVC(), svm_param_grid],
    'MLPClassifier' : [MLPClassifier(), mlp_param_grid],
    'TabNet' : [TabNetClassifier(), tabnet_param_grid],
    'XGBoost' : [XGBClassifier(), xgb_param_grid]
}


all_d_results = dict()
for d, d_val in df_dict_new.items():
    print('Dataset: ', d)
    # for each dataset, finetune all 5 classifiers
    X = d_val[0].drop(columns=class_field_dict[d])
    y = d_val[0][class_field_dict[d]]
    d_result = dict()
    for clf, clf_val in clf_dict.items():
        print('Dataset: ', d, ' - Classifier: ', clf)
        best_clf, best_params =  hyperparameter_tuning(clf_val[0], # classifier
                                                       clf_val[1], # parameter grid
                                                       X, 
                                                       y)
        d_result[clf] = [best_clf, best_params]
    joblib.dump(d_result, d+'_finetuning_results.pkl')
    all_d_results[d] = d_result
joblib.dump(all_d_results, 'all_datasets_finetuning_results.pkl')

