### Dataset source
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


### Hyperparameter tuning

compas = {
    'RandomForest'  : {'n_estimators': 25, 'min_samples_split': 6, 'max_leaf_nodes': 9, 'max_features': None, 'max_depth': 6},
    'SVC'           : {'kernel': 'rbf', 'gamma': 0.1, 'C': 10},
    'MLPClassifier' : {'solver': 'adam', 'max_iter': 200, 'hidden_layer_sizes': (100, 50, 30), 'alpha': 0.0001, 'activation': 'tanh'},
    'XGBoost'       : {'subsample': 0.7, 'max_depth': 7, 'learning_rate': 0.1, 'colsample_bytree': 0.9, 'objective': 'binary:logistic'},
    'TabNet'        : {'n_d': 16, 'n_a': 8, 'gamma': 1.2} 
}

adult = {
    'RandomForest'  : {'n_estimators': 150, 'min_samples_split': 6, 'max_leaf_nodes': 9, 'max_features': 'log2', 'max_depth': 9},
    'SVC'           : {'kernel': 'rbf', 'gamma': 0.01, 'C': 1},
    'MLPClassifier' : {'solver': 'adam', 'max_iter': 200, 'hidden_layer_sizes': (150, 100, 50), 'alpha': 0.0001, 'activation': 'relu'},
    'XGBoost'       : {'subsample': 0.9, 'max_depth': 7, 'learning_rate': 0.1, 'colsample_bytree': 0.7, 'objective': 'binary:logistic'},
    'TabNet'        : {'n_d': 24, 'n_a': 24, 'gamma': 1.2} 
}

german = {
    'RandomForest'  : {'n_estimators': 25, 'min_samples_split': 2, 'max_leaf_nodes': 9, 'max_features': None, 'max_depth': 9},
    'SVC'           : {'kernel': 'rbf', 'gamma': 0.1, 'C': 0.1},
    'MLPClassifier' : {'solver': 'adam', 'max_iter': 200, 'hidden_layer_sizes': (150, 100, 50), 'alpha': 0.005, 'activation': 'relu'},
    'XGBoost'       : {'subsample': 0.9, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.8, 'objective': 'binary:logistic'},
    'TabNet'        : {'n_d': 8, 'n_a': 8, 'gamma': 1.2} 
}

wine = {
    'RandomForest'  : {'n_estimators': 100, 'min_samples_split': 2, 'max_leaf_nodes': 9, 'max_features': None, 'max_depth': 9},
    'SVC'           : {'kernel': 'rbf', 'gamma': 0.1, 'C': 10},
    'MLPClassifier' : {'solver': 'adam', 'max_iter': 250, 'hidden_layer_sizes': (150, 100, 50), 'alpha': 0.005, 'activation': 'tanh'},
    'XGBoost'       : {'subsample': 0.9, 'max_depth': 7, 'learning_rate': 0.1, 'colsample_bytree': 0.9, 'objective': 'multi:softprob'},
    'TabNet'        : {'n_d': 8, 'n_a': 24, 'gamma': 1.5} 
}

student = {
    'RandomForest'  : {'n_estimators': 100, 'min_samples_split': 2, 'max_leaf_nodes': 9, 'max_features': None, 'max_depth': 9},
    'SVC'           : {'kernel': 'rbf', 'gamma': 0.001, 'C': 1},
    'MLPClassifier' : {'solver': 'adam', 'max_iter': 250, 'hidden_layer_sizes': (100, 50, 30), 'alpha': 0.005, 'activation': 'relu'},
    'XGBoost'       : {'subsample': 0.9, 'max_depth': 5, 'learning_rate': 0.1, 'colsample_bytree': 0.7, 'objective': 'multi:softprob'},
    'TabNet'        : {'n_d': 8, 'n_a': 8, 'gamma': 1.5} 
}

abalone = {
    'RandomForest'  : {'n_estimators': 150, 'min_samples_split': 4, 'max_leaf_nodes': 9, 'max_features': 'log2', 'max_depth': 6},
    'SVC'           : {'kernel': 'rbf', 'gamma': 0.1, 'C': 10},
    'MLPClassifier' : {'solver': 'adam', 'max_iter': 200, 'hidden_layer_sizes': (150, 100, 50), 'alpha': 0.005, 'activation': 'relu'},
    'XGBoost'       : {'subsample': 0.7, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.7, 'objective': 'multi:softprob'},
    'TabNet'        : {'n_d': 24, 'n_a': 24, 'gamma': 1.35} 
}


best_params_dict = {
    'compas'        : compas,
    'adult'         : adult,
    'german_credit' : german,
    'wine'          : wine,
    'student'       : student,
    'abalone'       : abalone
}


#### Hyperarameters for finetuning

rf_param_grid = {
    'n_estimators': [25, 50, 100, 150],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [3, 6, 9],
    'max_leaf_nodes': [3, 6, 9],
    'min_samples_split': [2, 4, 6]
}

svm_param_grid = {
    'C': [0.1, 1, 10], 
    'gamma': [0.1, 0.01, 0.001],
    'kernel': ['rbf']#, 'linear']
    } 

mlp_param_grid = {
    'hidden_layer_sizes': [(150,100,50), (100,50,30)],
    'max_iter': [200, 250],
    'activation': ['tanh', 'relu'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.005]
   # 'learning_rate': ['constant','adaptive'],
}

tabnet_param_grid = {
    'n_d' : [8, 16, 24], 
    'n_a' : [8, 16, 24],
   # 'n_steps' : [3, 4, 5], 
    'gamma' : [1.2, 1.35, 1.5],
   # 'optimizer_params' : [dict(lr=x) for x in [0.015, 0.02, 0.025]],
   # 'scheduler_params' : [dict(gamma=x, step_size=50) for x in [0.85, 0.9, 0.95]]
}

xgb_param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

grid_dict = {
    'RandomForest' :  rf_param_grid,
    'MLPClassifier' : mlp_param_grid,
    'TabNet' :        tabnet_param_grid,
    'XGBoost' :       xgb_param_grid,
    'SVC' :           svm_param_grid,
}

## Best classifier
best_clf_dict = {
    'compas'        : 'SVC',
    'adult'         : 'XGBoost',
    'german_credit' : 'XGBoost',
    'wine'          : 'XGBoost',
    'student'       : 'XGBoost',
    'abalone'       : 'MLPClassifier'
}


