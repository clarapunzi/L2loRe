import pandas as pd
import numpy as np
import joblib
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from MyLoreSA.datamanager import prepare_dataset
#from Tabular.xailib.explainers.lore_explainer import LoreTabularExplainer
from MyLoreSA.sklearn_classifier_wrapper import sklearn_classifier_wrapper
from MyLoreSA.util import get_df_stats, get_tuned_classifier, get_classification_metrics, load_datasets, transform_datasets

import matplotlib.pyplot as plt
from tabulate import tabulate

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

from best_params_clf import best_params_dict, source_file_dict, class_field_dict

if torch.cuda.is_available():
    device = 'cuda'
    tree_method_xgb = 'gpu_hist'
else:
    device = 'cpu'
    tree_method_xgb = 'auto'


# Load and transform dataset
df_dict = load_datasets(source_file_dict)

# Collect statistics about datasets
stats_dict = dict()
for k, v in df_dict.items():
    stats = get_df_stats(v, class_field_dict[k])
    stats_dict[k] = stats


df_dict_new = transform_datasets(df_dict, class_field_dict)

clf_dict = {
    'RandomForest' : RandomForestClassifier(random_state=123),
    'MLPClassifier' : MLPClassifier(random_state=123, early_stopping=True),
    'TabNet' : TabNetClassifier(device_name=device, verbose=0, seed=123),
    'XGBoost' : XGBClassifier(tree_method=tree_method_xgb, random_state=123),
    'SVC' : SVC(probability=True, random_state=123)
}

     
all_d_results = dict()

for d, d_val in df_dict_new.items():
    
    # for each dataset, evaluate all 5 classifiers
    X = d_val[0].drop(columns=class_field_dict[d])
    y = d_val[0][class_field_dict[d]]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=123,
                                                        stratify=y)
    d_result = dict()
    for clf, clf_val in clf_dict.items():
        print('Dataset: ', d, ' - Classifier: ', clf)
        bb = get_tuned_classifier(d, clf, clf_dict, best_params_dict)
        bb.fit(X_train.values, y_train.values)
        y_pred = bb.predict(X_test.values)
        y_proba = bb.predict_proba(X_test.values)
        binary = True if stats_dict[d]['classification_type'] == 'binary' else False
        metrics = get_classification_metrics(y_test, y_pred, y_proba, binary)
        d_result[clf] = metrics       
    
    # update the stats dict with the average accuracy, f1-score and ROC-AUC score
    stats_dict[d]['avg_accuracy'] = np.mean([m['accuracy_score'] for m in d_result.values()])
    stats_dict[d]['avg_f1_score'] = np.mean([m['f1_score'] for m in d_result.values()])
    stats_dict[d]['avg_roc_auc'] = np.mean([m['roc_auc_score'] for m in d_result.values()])
    stats_dict[d]['max_accuracy'] = np.max([m['accuracy_score'] for m in d_result.values()])
    stats_dict[d]['max_f1_score'] = np.max([m['f1_score'] for m in d_result.values()])
    stats_dict[d]['max_roc_auc'] = np.max([m['roc_auc_score'] for m in d_result.values()])
    
    all_d_results[d] = d_result

# Save results
reformed_results = {(outerKey, innerKey): values for outerKey, innerDict in all_d_results.items() for innerKey, values in innerDict.items()}
all_d_results_df = pd.DataFrame(reformed_results)
all_d_results_df = all_d_results_df.stack().unstack(level=0)
all_d_results_df = all_d_results_df.append(pd.Series(all_d_results_df.max(), name='max'))
all_d_results_df = all_d_results_df.append(pd.Series(all_d_results_df.idxmax(), name='best_clf'))

stats_df = pd.DataFrame(stats_dict)
stats_df.to_csv('dataset_statistics.csv')

# joblib.dump(all_d_results, 'all_datasets_classification_results.pkl')

# for d in stats_df.columns:
#     stats_dict[d]['max_accuracy'] = np.max([all_d_results[d][clf]['accuracy_score'] for clf in all_d_results[d].keys()])
#     stats_dict[d]['max_f1_score'] = np.max([all_d_results[d][clf]['f1_score'] for clf in all_d_results[d].keys()])
#     stats_dict[d]['max_roc_auc'] = np.max([all_d_results[d][clf]['roc_auc_score'] for clf in all_d_results[d].keys()])
    