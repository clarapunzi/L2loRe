import pandas as pd
import numpy as np
import pickle
import torch

from sklearn.model_selection import train_test_split
from tqdm import tqdm

#from Tabular.xailib.explainers.lore_explainer import LoreTabularExplainer
from MyLoreSA.lorem_new import LOREU
from MyLoreSA.util import get_tuned_classifier, load_datasets, transform_datasets, compute_rejection_policy, compute_distance_from_counterfactual
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from best_params_clf import best_clf_dict, best_params_dict, source_file_dict, class_field_dict
from cf_parameters import neigh_kwargs
import argparse
 
parser = argparse.ArgumentParser(description="",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--datasets", nargs='+', type=str, default='abalone')
parser.add_argument("-test_size", "--test_size", type=int, default=5)
parser.add_argument("-u", "--uncertainty_threshold", type=float, default=0.6)
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


neigh_type = 'geneticp' # the generation you want (random, genetic, geneticp, cfs, rndgen, rndgenp)
binary = 'binary_from_dts' #how to merge the trees (binary from dts, binary from bb are creating a binary tree, nari is creating a n ari tree)
n = 1000 # neighborhood size (i.e., 2n instances in total) --> in realtà dovrebbe essere sample
n_neigh = 150
cxpb = 0.7 # values to set for the genetic generation
mutpb = 0.5 #values to set for the genetic generation
unc_thr = config['uncertainty_threshold']
uncertainty_metric = 'max' # max or ratio
test_size = config['test_size']


df_dict_new_small = {k:v for k,v in df_dict_new.items() if k in datasets}
print('The following datasets will be analyzed: ', df_dict_new_small.keys(), ' on ', test_size, ' test samples.')

all_d_results = dict()
for d, d_val in df_dict_new_small.items():
    # for each dataset, evaluate all 5 classifiers
    X = d_val[0].drop(columns=class_field_dict[d])
    y = d_val[0][class_field_dict[d]]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=123,
                                                        stratify=y)
    d_result = dict()
    # Generate counterfactuals for the best classifier only
    clf = best_clf_dict[d]
    print('Dataset: ', d, ' - Classifier: ', clf)
    bb = get_tuned_classifier(d, clf, clf_dict, best_params_dict)
    bb.fit(X_train.values, y_train.values)
    y_pred = bb.predict(X_test.values)
    y_proba = bb.predict_proba(X_test.values)

    # Initialize the explainer object
    explainer = LOREU(X_train.values, bb.predict, bb.predict_proba, feature_names=d_val[1], class_name=class_field_dict[d], 
                      class_values=d_val[2], numeric_columns=d_val[3], features_map=d_val[-1],
                      neigh_type=neigh_type, categorical_use_prob=True, continuous_fun_estimation=True, size=n,
                      ocr=0.1, multi_label=False, one_vs_rest=False, random_state=42, verbose=False,
                      Kc=X_train, K_transformed=X_train.values, discretize=True, encdec=None,
                      uncertainty_thr=unc_thr, uncertainty_metric=uncertainty_metric, binary=binary, extreme_fidelity = True, **neigh_kwargs)

    # Compute counterfactual over a random sample of 200 test instances (stratified by correct prediction)
    s = pd.Series(y_pred == y_test).to_frame()
    s = s.groupby(class_field_dict[d]).sample(frac=test_size/len(y_pred))
    selected_indices = s.index
    #x_list = X_test.loc[selected_indices] 
    y_list = y_test.loc[selected_indices] 

    id_map = {j:i for i, j in enumerate(y_test.index)}
    expl_list = dict()
    for j in tqdm(selected_indices):
        exp = explainer.explain_instance_stable(X_test.loc[j].values, samples=n_neigh, extract_counterfactuals_by= 'min_distance')
        expl_list[j] = exp

    # compute distance from counterfactuals (minimum over all classes)
    dist_dict = compute_distance_from_counterfactual(X_test, expl_list)
    frac_found_counterfactuals = (test_size - list(dist_dict.values()).count(np.inf) ) / test_size

    # r_list contains the rejection fraction on which to evaluate the selective classifier
    r_list = list(range(test_size+1))
    rejected_samples, an_list, cq_list, rq_list, rej_class_report_list = compute_rejection_policy(r_list, dist_dict, y_test.loc[selected_indices], [y_pred[id_map[i]] for i in selected_indices])

    all_d_results[d] = {
        'tested_samples': selected_indices,
        'explanation': expl_list,
        'min_distances': dist_dict,
        'fraction_found_counterfactuals': frac_found_counterfactuals,
        'uncertainty_threshold': unc_thr,
        'rejected_samples' : rejected_samples,
        'nonrejected_accuracy' : an_list,
        'classification_quality': cq_list,
        'rejection_quality' : rq_list,
        'rejection_classification_report' : rej_class_report_list,
        'class_values_map': {k:v for k,v in enumerate(d_val[2])},
        'original_data': X_test.loc[selected_indices]
    } 

    with open('Results/fake'+d+'_results.pickle', "wb") as pickle_file:
        pickle.dump(all_d_results, pickle_file)
