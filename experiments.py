import pandas as pd
import numpy as np
import pickle

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy.spatial.distance import cdist

from LoreSA.datamanager import prepare_dataset
#from Tabular.xailib.explainers.lore_explainer import LoreTabularExplainer
from LoreSA.sklearn_classifier_wrapper import sklearn_classifier_wrapper

from LoreSA.lorem_new import LOREM
from LoreSA.metrics import evaluate_cf_list, nonrejected_accuracy, classification_quality, rejection_quality, rejection_classification_report
from LoreSA.util import get_df_stats

import matplotlib.pyplot as plt
from tabulate import tabulate

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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

stats_dict = dict()
for k, v in df_dict.items():
    stats = get_df_stats(v, class_field_dict[k])
    stats_dict[k] = stats

stats_dict = pd.DataFrame(stats_dict)
#tabulate(s, headers = 'keys', tablefmt = 'plain')

get_df_stats(students_df, class_field_dict['student'])

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

clf = RandomForestClassifier()
rf, rf_params = hyperparameter_tuning(clf, rf_param_grid, df_dict_new['compas'][0].drop(columns=['did_recid']), df_dict_new['compas'][0]['did_recid'])

# Train RF
test_size = 0.2
random_state = 42
X_train, X_test, Y_train, Y_test = train_test_split(df[feature_names], df[class_field],
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=df[class_field])
bb = RandomForestClassifier(n_estimators=20, random_state=random_state)
bb.fit(X_train.values, Y_train.values)
bbox = sklearn_classifier_wrapper(bb) 
Y_pred = bb.predict(X_test)
print(classification_report(Y_test, Y_pred))


# predict on a new instance
i = 14383
inst = X_test.loc[i].values
print('Instance ',inst)
print('True class ',Y_test.loc[i])
print('Predicted class ',bb.predict(inst.reshape(1, -1)))
print('Predicted class probabilities ', bb.predict_proba(inst.reshape(1, -1))[0])

# LORE_sa
neigh_type = 'geneticp' # the generation you want (random, genetic, geneticp, cfs, rndgen)

binary = 'binary_from_dts' #how to merge the trees (binary from dts, binary from bb are creating a binary tree, nari is creating a n ari tree)
cxpb = 0.7 # values to set for the genetic generation
mutpb = 0.5 #values to set for the genetic generation
alpha1 = 0.5
alpha2 = 0.5
ngen = 2 # number of neighborhood generations to do
runs = 2 #how many neighbours and trees to create and then merge
class_name = 'class' #name of the column used as target
class_values = [0,1] #values that the target may have
unc_thr = 0.7
uncertainty_metric = 'max'
neigh_kwargs = {
        "balance": False,
        "sampling_kind": "gaussian",
        "kind": "gaussian_global",
        "downward_only": True,
        "redo_search": True,
        "forced_balance_ratio": 0.5,
        "cut_radius": True,
        "n": 800,
        "normalize": 'minmax',
        "forced_balance_ratio": 0.5,
        "n_batch": 5,
        "datas": X_train.values,
        "alpha1": alpha1,
        "alpha2": alpha2
    }

# ocr, mutpb, cxpb: values to set for the genetic generation
# discretize: in the surrogate trees, discretize variables to better generalize and have a smaller tree
# encdec: if you want to generate the neighbours in an encoded space (for now you can select onw hot and target encoding)
# dataset: in case you selected an encdec, this dataset is used to fit the encoder
# K_transformed: a piece of dataset in the form you can feed to the BB
# extreme fidelity: if True, it checks that the prediction of the surrogate model is the same as the BB
# filter c rules: it filters the counterfactual rules by checking if they are true conterfactuals or not
explainer = LOREM(X_train.values, bb.predict, bb.predict_proba, 
                  feature_names, class_values, numeric_columns, categorical_columns
                  feature_names, class_name, class_values, numeric_columns, features_map,
                      neigh_type=neigh_type, categorical_use_prob=True, continuous_fun_estimation=True, size=1000,
                      ocr=0.1, multi_label=False, one_vs_rest=False, random_state=42, verbose=True,
                      Kc=X_train, bb_predict_proba=bb.predict_proba, K_transformed=X_train.values, discretize=True,
                      encdec=None, uncertainty_thr=unc_thr, uncertainty_metric=uncertainty_metric, binary=binary, **neigh_kwargs)

# x the instance to explain
# samples the number of samples to generate during the neighbourhood generation
# use weights True or False
# metric default is neuclidean, it is the metric employed to measure the distance between records
# runs number of times the neighbourhood generation is done
# exemplar_num number of examplars to retrieve
# kwargs a dictionary in which add the parameters needed for cfs generation
# kernel and kernel width are for the definition of weights. deafult is None, it automatically select them.

x_list = []
y_true = []
i = 14383
inst = X_test.loc[i].values
y_true.append(Y_test.loc[14383])
x_list.append(inst)
for i in range(100):
    y_true.append(Y_test.iloc[i])
    x_list.append(X_test.iloc[i].values)

expl_list = []
for j in range(len(x_list)):
    print('Iteration ', j+1)
    expl = explainer.explain_instance_stable(x_list[j], 150, runs=runs, n_jobs=2
                                         , extract_counterfactuals_by= 'min_distance'
                                         )
    expl_list.append(expl)


n = len(x_list)
# r_list contains the rejection fraction on which to evaluate the selective classifier
r_list = list(range(n+1))
y_pred = bb.predict(x_list)



plt.plot(r_list, an_list)
plt.show()



unc_list = []
unc_list_c = []
for i in range(len(expl_list)):
    x_pred = bb.predict_proba(x_list[i].reshape(1, -1))[0]
    unc_list.append(x_pred)
    if len(expl_list[i].Xc)>0:
        xc_pred = bb.predict_proba(expl_list[i].Xc[0].reshape(1, -1))[0]
        unc_list_c.append(xc_pred)
    else:
        unc_list_c.append([0,0])
    

for e in expl_list:
    print(len(e.Xc))


# Explanation rule
print('N premises in the rule: ', len(expl.rule))
print('Rule: ', str(expl.rule))

# Counterfactual
print('N counterfactuals: ', len(expl.crules))
# print counterfactual
print(expl.cstr())
for c in expl.crules:
    print(str(c))


# delta contains the single conditions in the counterfactual

n_cols = len(X_train.columns)

# Evaluate counterfactual list
x_eval = evaluate_cf_list(np.array(expl.Xc), 
            expl.crules,   
            inst, 
            bb, 
            expl.bb_pred, 
            3, 
            variable_features=list(range(n_cols)), # We do not account for actionability -> variable_features = all features
            continuous_features_all = list(range(n_cols)), # all features are continuous here 
            categorical_features_all=list(), 
            X_train=X_train.values, X_test=X_test.values,
            ratio_cont = len(numeric_columns) /n_cols, 
            nbr_features = n_cols
            )

for k, v in x_eval.items():
    print(k, ' : ', v)

# Explain test set
n_workers = 6
title = 'first_trial'
set_expl = explainer.explain_set_instances_stable(X_test.values[:100], n_workers, title)


with open('explanations_lorefirst_trial_5.p', 'rb') as pickle_file:
    content = pickle.load(pickle_file)

unc_list = []
unc_list_c = []
for d in content:
    x_pred = bb.predict_proba(d[0].reshape(1, -1))[0]
    xc_pred = bb.predict_proba(d[1].Xc)[0]
    unc_list_c.append(xc_pred)
    unc_list.append(x_pred)

N = 11
colors = 2 * np.pi * np.random.rand(N)
plt.scatter([u[0] for u in unc_list], [u[1] for u in unc_list], c=colors) 
plt.scatter([u[0] for u in unc_list_c], [u[1] for u in unc_list_c], c=colors) 
plt.show()