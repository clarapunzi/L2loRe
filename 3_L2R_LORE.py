import pandas as pd
import numpy as np
import pickle
import torch

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import datetime

#from Tabular.xailib.explainers.lore_explainer import LoreTabularExplainer
from MyLoreSA.lorem_new import LOREU
from MyLoreSA.util import get_tuned_classifier, load_datasets, transform_datasets, compute_rejection_policy, compute_distance_from_counterfactual
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from MyLoreSA.best_params_clf import best_clf_dict, best_params_dict, source_file_dict, class_field_dict
from MyLoreSA.cf_parameters import neigh_kwargs
import argparse
from MyLoreSA.l2r_lore import L2R_LORE
 
parser = argparse.ArgumentParser(description="",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--datasets", nargs='+', type=str, default='german_credit') 
parser.add_argument("-test_size", "--test_size", type=int, default=-1)
parser.add_argument("-u", "--uncertainty_threshold", type=float, default=0.5)
args = parser.parse_args()
config = vars(args)

datasets = config['datasets']

if torch.cuda.is_available():
    device = 'cuda'
    tree_method_xgb = 'gpu_hist'
else:
    device = 'cpu'
    tree_method_xgb = 'auto'

# Load and transform dataset
df_dict = load_datasets(source_file_dict)
df_dict_new = transform_datasets(df_dict, class_field_dict)

clf_dict = {
    #'RandomForest' : RandomForestClassifier(random_state=123),
    'MLPClassifier' : MLPClassifier(random_state=123, early_stopping=True),
    #'TabNet' : TabNetClassifier(device_name=device, verbose=0, seed=123),
    'XGBoost' : XGBClassifier(tree_method=tree_method_xgb, random_state=123),
    'SVC' : SVC(probability=True, random_state=123)
}


neigh_type = 'rndgenp' # the generation you want (random, genetic, geneticp, cfs, rndgen)
binary = 'binary_from_dts' #how to merge the trees (binary from dts, binary from bb are creating a binary tree, nari is creating a n ari tree)
n = 1000 # neighborhood size (i.e., 2n instances in total) --> in realtà dovrebbe essere sample
n_neigh = 150
cxpb = 0.7 # values to set for the genetic generation
mutpb = 0.5 #values to set for the genetic generation
unc_thr = config['uncertainty_threshold'] # TO DO: finetuning
uncertainty_metric = 'max' # max or ratio
test_size = config['test_size'] # TO DO: add option None to train lore on train-test split
extract_counterfactuals_by='min_distance'
optimize_tau_by='nonrejected_accuracy' 
counterfactual_metric  = 'mixed'


df_dict_new_small = {k:v for k,v in df_dict_new.items() if k in datasets}
print('The following datasets will be analyzed: ', df_dict_new_small.keys())

all_d_results = dict()
for d, d_val in df_dict_new_small.items():
    
    # for each dataset, optimize the rejection policy and compute predictions/rejections with explanations
    X = d_val[0].drop(columns=class_field_dict[d])
    y = d_val[0][class_field_dict[d]]

    if test_size == -1:
        test_size == X.shape[0]
    
    numeric_columns, categorical_columns = [], []
    for i, col in enumerate(X.columns):
        if col in d_val[3]:
            numeric_columns.append(i)
        else:
            categorical_columns.append(i)
    
    #X, y = X.head(100), y.head(100)

    # here we create test set using perc_test and calibration set using perc_hold
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=1234,
                                                        stratify=y)
    X_train, X_cal, y_train, y_cal = train_test_split(X_train, y_train,
                                                        test_size=0.125,
                                                        random_state=1234,
                                                        stratify=y_train)
    d_result = dict()

    # Generate counterfactuals for the best classifier only
    clf = best_clf_dict[d]
    print('Dataset: ', d, ' - Classifier: ', clf, ' - Training size: ', X_train.shape[0])
    bb = get_tuned_classifier(d, clf, clf_dict, best_params_dict)


    # Initialize the explainer object
    explainer = LOREU(X_train.values, bb.predict, bb.predict_proba, feature_names=d_val[1], class_name=class_field_dict[d], 
                      class_values=d_val[2], numeric_columns=d_val[3], features_map=d_val[-1],
                      neigh_type=neigh_type, categorical_use_prob=True, continuous_fun_estimation=True, size=n,
                      ocr=0.1, multi_label=False, one_vs_rest=False, random_state=42, verbose=False,
                      Kc=X_train, K_transformed=X_train.values, discretize=True, encdec=None,
                      uncertainty_thr=unc_thr, uncertainty_metric=uncertainty_metric, 
                      binary=binary, extreme_fidelity = True, **neigh_kwargs)
    

    lore = L2R_LORE(bb, explainer, numeric_columns, categorical_columns,
                    n_neighbors=n_neigh, extract_counterfactuals_by=extract_counterfactuals_by, 
                    optimize_tau_by=optimize_tau_by, 
                    counterfactual_metric=counterfactual_metric, n_tau_candidates=100, 
                    max_rej_fraction=0.6, max_error=0.3, tau=0, base_score=0, max_score=0, 
                    rej_rate=0, counterfactual_rate=None)
    
    print('Fitting data')    
    training_results, calibration_results = lore.fit(X_train, y_train, X_cal, y_cal)

    print('Computing predictions')
    #y_preds, l2r_preds, dist_dict, rejected_samples 
    prediction_results = lore.predict(X_test)
    y_preds = [y.prediction for y in prediction_results['y_predictions']]
    n_rejected_best = len(prediction_results['rejected_list'])
    
    print('Evaluating the selective classifier')
    evaluation_results = lore.evaluate_l2r(len(prediction_results['rejected_list']), 
                                           prediction_results['distance_dict'], 
                                           y_test, y_preds)

    results = {'training': training_results, 
               'calibration': calibration_results,
               'predictions': prediction_results,
               'evaluation': evaluation_results
               }

    ct = datetime.datetime.now()
    ct = ct.strftime("%m_%d_%Y_%H_%M")
    with open('Results/l2lore_results_'+d+'_'+ct+'.pickle', "wb") as pickle_file:
        pickle.dump(results, pickle_file)
