import pandas as pd
import numpy as np
import pickle
import joblib
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


from MyLoreSA.datamanager import prepare_dataset
#from Tabular.xailib.explainers.lore_explainer import LoreTabularExplainer
#from LoreSA.sklearn_classifier_wrapper import sklearn_classifier_wrapper
from MyLoreSA.util import get_df_stats, hyperparameter_tuning

import matplotlib.pyplot as plt
from tabulate import tabulate

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
#from sklearn.model_selection import RepeatedStratifiedKFold
#from sklearn.model_selection import RandomizedSearchCV

from MyLoreSA.best_params_clf import best_params_dict, grid_dict
import argparse
 
parser = argparse.ArgumentParser(description="",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-clf", "--classifier", default='RandomForest', help="Classifier to be finetuned")
args = parser.parse_args()
config = vars(args)

single_clf = config['classifier']

if torch.cuda.is_available():
    device = 'cuda'
    tree_method_xgb = 'gpu_hist'
else:
    device = 'cpu'
    tree_method_xgb = 'auto'

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


df_dict = { 
            'wine' : wine_df,
            'student': students_df,
            'abalone': abalone_df,
            'compas' : compas_df,
            'adult' : adult_df,
            'german_credit' : german_df,
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


clf_dict = {
    'RandomForest' : [RandomForestClassifier(), grid_dict['RandomForest']],
    'MLPClassifier' : [MLPClassifier(early_stopping=True), grid_dict['MLP']],
    'TabNet' : [TabNetClassifier(device_name=device, verbose=0), grid_dict['TabNet']],
    'XGBoost' : [XGBClassifier(tree_method=tree_method_xgb), grid_dict['XGBoost']],
    'SVC' : [SVC(), grid_dict['SVC']],
}

all_d_results = dict()
for d, d_val in df_dict_new.items():
    print('Dataset: ', d)
    # for each dataset, finetune all 5 classifiers
    X = d_val[0].drop(columns=class_field_dict[d]).values
    y = d_val[0][class_field_dict[d]].values
    d_result = dict()
    for clf, clf_val in clf_dict.items():
        print('Dataset: ', d, ' - Classifier: ', clf)
        if best_params_dict[d][clf] == None:
            best_clf, best_params =  hyperparameter_tuning(clf_val[0], # classifier
                                                       clf_val[1], # parameter grid
                                                       X, 
                                                       y)
            d_result[clf] = [best_clf, best_params]
            joblib.dump(d_result, d+'_temp_finetuning_results.pkl')
    joblib.dump(d_result, d+'_finetuning_results.pkl')
    all_d_results[d] = d_result
joblib.dump(all_d_results, 'all_datasets_finetuning_results.pkl')
 
# one classifier only
# clf_result = dict()
# for d, d_val in df_dict_new.items():
#     if d == 'adult':
#         print('Dataset: ', d)
#         # for each dataset, finetune all 5 classifiers
#         X = d_val[0].drop(columns=class_field_dict[d]).values
#         y = d_val[0][class_field_dict[d]].values    
#         clf, clf_val = clf_dict[single_clf]
#         best_clf, best_params =  hyperparameter_tuning(clf, # classifier
#                                                     clf_val, # parameter grid
#                                                     X, 
#                                                     y)
#         clf_result[d] = [best_clf, best_params]
#         joblib.dump(clf_result, single_clf+'_temp_finetuning_results.pkl')
# #joblib.dump(clf_result, d+'_finetuning_results.pkl')
    
