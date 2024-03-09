from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import IsolationForest
from scipy.stats import gamma
from MyLoreSA.lorem_new import LOREU
from MyLoreSA.util import compute_rejection_policy, compute_distance_from_counterfactual
from Lib.LoreSA.util import neuclidean
from tqdm import tqdm
import numpy as np
import pandas as pd
from termcolor import colored


class L2R_LORE_Prediction(object):

    def __init__(self, x=None, prediction=None, explanation=None, is_rejected = False, is_outlier=False):
        self.x = x
        self.prediction = prediction
        self.explanation = explanation
        self.is_rejected = is_rejected
        self.is_outlier = is_outlier

    def __str__(self):
        print(self.explanation)

    def get_explanation(self):
        print(self.explanation)
        return self.explanation
    
    def get_prediction(self):
        return self.prediction
    
    def get_id(self):
        return self.x
    
    def set_prediction(self, y):
        self.prediction = y

    def set_explanation(self, e):
        self.explanation = e

    def is_rejected(self):
        return self.is_rejected
    
    def is_outlier(self):
        return self.is_outlier


class L2R_LORE(BaseEstimator, ClassifierMixin):
     
    def __init__(self, base_clf, explainer, numeric_columns, categorical_columns,
                 n_neighbors=150, extract_counterfactuals_by='min_distance', optimize_tau_by='nonrejected_accuracy',
                 counterfactual_metric = 'mixed',
                 n_tau_candidates=100, max_rej_fraction=0.6, max_error=0.3, tau=0, 
                 base_score=0, max_score=0, rej_rate=0, counterfactual_rate=None):
        
        self.base_clf = base_clf
        self.explainer = explainer
        self.n_neighbors = n_neighbors
        self.extract_counterfactuals_by = extract_counterfactuals_by
        self.optimize_tau_by = optimize_tau_by
        self.n_tau_candidates = n_tau_candidates
        self.max_rej_fraction = max_rej_fraction
        self.max_error = max_error
        self.tau = tau
        self.base_score = base_score
        self.max_score = max_score
        self.rej_rate = rej_rate
        self.counterfactual_rate=counterfactual_rate
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.counterfactual_metric = counterfactual_metric
        self.training_rejection_report = None

    
    def fit(self, X, y, sample=False, n_samples = 200):

        # STEP 1: BASE CLASSIFIER
        # Fit and get prediction of the base classifier
        self.base_clf.fit(X.values, y.values)
        y_pred = self.base_clf.predict(X.values)
        y_proba = self.base_clf.predict_proba(X.values)
        # check if there are misclassified samples
        miscl = []
        for i in range(len(y)):
            if y.values[i] != y_pred[i]:
                miscl.append(i)
        if len(miscl) == 0:
            print('The black box correctly classified all samples')

        # STEP 2: COUNTERFACTUALS
        # Compute counterfactuals for all samples in X (train)
        X_indices = X.index
        sample_size = X.shape[0]

        if sample:
            s = pd.Series(y_pred == y).to_frame()
            s = s.groupby(self.explainer.class_name).sample(frac=n_samples/len(y_pred))
            X_indices = s.index
            sample_size = n_samples
            #x_list = X_test.loc[selected_indices] 
            #y_list = y_test.loc[selected_indices] 
        
        id_map = {j:i for i, j in enumerate(y.index)}
        expl_list = dict()
        for i in tqdm(range(len(X_indices))):
            j = X_indices[i]
            exp = self.explainer.explain_instance_stable(X.loc[j].values, y_pred[i],
                                                         self.numeric_columns, self.categorical_columns,
                                                         samples=self.n_neighbors, 
                                                         extract_counterfactuals_by=self.extract_counterfactuals_by,
                                                         counterfactual_metric = self.counterfactual_metric,
                                                         metric=neuclidean)
            expl_list[j] = exp

        # compute distance from counterfactuals (minimum over all classes)
        #dist_dict = compute_distance_from_counterfactual(X, expl_list)t
        min_dist_dict = dict()
        for k, v in expl_list.items():
            min_dist = np.inf
            for d, v_d in v.dist_dict.items():
                if v_d < min_dist:
                    min_dist = v_d
            min_dist_dict[k] = min_dist
        
        
        frac_found_counterfactuals = (sample_size - list(min_dist_dict.values()).count(np.inf) ) / sample_size
        self.counterfactual_rate = frac_found_counterfactuals

        # check which points are novelties among those at inf distance
        # if x is a novelty -> reject (update dist dict with -inf)
        # else x is far and confident -> keep
        # list of -1 for outliers and 1 for inliers
        outliers_isolation_forest = IsolationForest(random_state=0).fit_predict(X)
        outliers_dict = {v:outliers_isolation_forest[k] for k,v in enumerate(X.index)}

        # compare instances with no counterfactuals with outliers
        for j in outliers_dict.keys():
            if (min_dist_dict[j] == np.inf) and (outliers_dict[j]) == -1 :
                # correct outlier
                min_dist_dict[j] = 0

        # r_list contains the number of rejected samples on which to evaluate the selective classifier
        r_list = list(range(sample_size+1))
        rejected_samples, an_dict, cq_dict, rq_dict, rej_class_report_dict = compute_rejection_policy(r_list, 
                                                                                                      min_dist_dict, 
                                                                                                      y.loc[X_indices], 
                                                                                                      [y_pred[id_map[i]] for i in X_indices]
                                                                                                      )
        metric_dict = {
            'classification_quality': cq_dict,
            'rejection_quality'     : rq_dict,
            'nonrejected_accuracy'  : an_dict,
            'rejected_samples'      : rejected_samples,
            'rej_class_report'      : rej_class_report_dict
        }       
        
        # STEP 3: FIND OPTIMAL TAU
        # fit the rejection threshold tau
        base, tau, m_max, frac_rejected, n_rej_final = self.fit_tau(sample_size, 
                                                               min_dist_dict, 
                                                               list(an_dict.values()), 
                                                               metric_dict[self.optimize_tau_by], 
                                                               self.n_tau_candidates, 
                                                               self.max_rej_fraction, 
                                                               self.max_error
                                                               )
        self.tau = tau
        self.base_score = base
        self.max_score = m_max
        self.rej_rate = frac_rejected
        self.training_rejection_report = metric_dict

        print('Optimal rejection rate: ', frac_rejected)
        print('Optimal rejection threshold: ', tau)

        return self
    
    def predict(self, X):
        # output the class inferred by the classifier
        
        # STEP 1: BASE CLASSIFIER
        # Get prediction of the base classifier
        y_pred = self.base_clf.predict(X.values)
        y_proba = self.base_clf.predict_proba(X.values)

        
        # Check outliers through IsolationForest
        # list of -1 for outliers and 1 for inliers
        outliers_isolation_forest = IsolationForest(random_state=0).fit_predict(X)

        
        # STEP 2: COUNTERFACTUALS
        # Compute counterfactuals for all samples in X (train)  
        expl_list = dict()
        id_map = {j:i for i, j in enumerate(X.index)}
        for j in tqdm(X.index):
            i = id_map[j]
            exp = self.explainer.explain_instance_stable(X.loc[j].values, y_pred[i],
                                                         self.numeric_columns, self.categorical_columns,
                                                         samples=self.n_neighbors, 
                                                         extract_counterfactuals_by=self.extract_counterfactuals_by
                                                         )
            expl_list[j] = exp

        # compute distance from counterfactuals (minimum over all classes)
        #dist_dict = compute_distance_from_counterfactual(X, expl_list)
        min_dist_dict = dict()
        for k, v in expl_list.items():
            min_dist = np.inf
            for d, v_d in v.dist_dict.items():
                if v_d < min_dist:
                    min_dist = v_d
            min_dist_dict[k] = min_dist
        
        sample_size = X.shape[0] 
        frac_found_counterfactuals = (sample_size - list(min_dist_dict.values()).count(np.inf) ) / sample_size
        print('Counterfactual rate: ', frac_found_counterfactuals)


        # check which points are novelties among those at inf distance
        # if x is a novelty -> reject (update dist dict with -inf)
        # else x is far and confident -> keep
        # list of -1 for outliers and 1 for inliers
        outliers_isolation_forest = IsolationForest(random_state=0).fit_predict(X)
        outliers_dict = {v:outliers_isolation_forest[k] for k,v in enumerate(X.index)}

        # compare instances with no counterfactuals with outliers
        for j in outliers_dict.keys():
            if (min_dist_dict[j] == np.inf) and (outliers_dict[j]) == -1 :
                # correct outlier
                min_dist_dict[j] = 0

        
        # STEP 3: REJECTION POLICY
        rejected_samples = [k for k, v in min_dist_dict.items() if v < self.tau]
        n_rejected = len(rejected_samples)
        frac_rej = n_rejected/len(min_dist_dict.keys())
        print('Number of rejected samples: ', len(rejected_samples), ' (', frac_rej*100, '%)')

        # Final prediction list with -1 for all rejected samples        
        id_map = {i:j for i, j in enumerate(X.index)}
        #rejected_samples_ids = [id_map[i] for i in rejected_samples]

         
        # STEP 4: EXPLANATION OF REJECTION
        preds = []
        for i in range(len(y_pred)):
            x = id_map[i]
            if x in rejected_samples:
                # check if outlier
                if outliers_dict[x] == -1:
                    ##
                    xai = 'The instance is an outlier'
                    preds.append(L2R_LORE_Prediction(x, y_pred[i], xai, True, True))
                    
                else:
                    xai = self.explain_rejection(expl_list[x], X, x)
                    preds.append(L2R_LORE_Prediction(x, y_pred[i], xai, True))
            else:
                xai = self.explain_prediction(expl_list[x], X, x)
                preds.append(L2R_LORE_Prediction(x, y_pred[i]))

        prediction_results = {'y_predictions' : preds,
                              'distance_dict' : min_dist_dict,
                              'rejected_list' : rejected_samples,
                              'outliers'      : outliers_isolation_forest,
                              'rejection_rate': frac_rej,
                              'tau'           : self.tau
                              }
        
        
        # y_pred, preds, min_dist_dict, rejected_samples
        return prediction_results
    
    
    
    def misprediction_rate(self, nonrejected_accuracy_list):
        # value of nonrejected accuracy when n_rej samples have been rejected
        n = len(nonrejected_accuracy_list)
        m_list = []
        for i in range(n):
            # (miscl. over all accepted) * (number of accepted samples) / tot samples
            m = (1 - nonrejected_accuracy_list[i]) * (n-i) / n
            m_list.append(m)
        return np.array(m_list)
    
    
    def fit_tau(self, sample_size, dist_dict, nonrejected_accuracy_list, metric_list, n_tau_candidates, max_rej_fraction, max_error):
        """
        metric_list is a list with the results from one of ['nonrejected_accuracy', 'classification_quality', 'rejection_quality']
        """

        # fit gamma distribution and sample data
        alpha, loc, beta = gamma.fit([d for d in dist_dict.values() if d != np.inf])
        tau_candidates = gamma.rvs(alpha, loc, beta, size = n_tau_candidates)
        tau_candidates = np.append(tau_candidates, [0, np.inf])

        miscl_by_r = self.misprediction_rate(nonrejected_accuracy_list)
    
        m_max = 0
        tau = np.inf
        n_rej_final = 0
        #n =  X.shape[0]S
        for t in tau_candidates:
            # check number of instances that will be rejected, i.e., all those with distance < t
            if t == np.inf:
                rej = [x for x, d in dist_dict.items() if d <= t]
            else:
                rej = [x for x, d in dist_dict.items() if d < t]
            n_rej = len(rej)
            rej_rate = n_rej / sample_size
            # if n_rej >= 11:
            #     print('Here')
        
            # constraints: 1) rejection rate below thr and 2) mispredictions below thr
            if (rej_rate < max_rej_fraction) & (miscl_by_r[n_rej] < max_error):
                m = metric_list[n_rej] 
                if m > m_max: # accuracy
                    m_max = m
                    n_rej_final = n_rej
                    tau = t
                elif (m == m_max) & (t < tau):
                    n_rej_final = n_rej
                    tau = t

        base = metric_list[0]
        frac_rejected = n_rej_final / sample_size
        return base, tau, m_max, frac_rejected, n_rej_final
    

    def explain_rejection(self, explanation, X, i):
        #x = res[d]['explanation'][i]
        n_c_classes = len(explanation.crules.keys())
        class_map = {k:v for k,v in enumerate(self.explainer.class_values)}

        #X_test = res[d]['original_data']

        # Rule
        text = 'The instance '+ colored('x = '+str(i), 'green') + ' has been ' + colored('REJECTED', 'red', attrs=['bold']) + '.\n' 
        text += 'In fact, it would have been classified by the black-box as '+ colored(str(explanation.rule.class_name) + '='+ str(explanation.bb_pred), 'green') + ' with ' + colored(str(round(max(explanation.bb_pred_proba), 2)), 'green') +  ' predicted probability, '
        text += '\nbut ' + colored(str(n_c_classes)+' counterfactuals', 'blue', attrs=['bold']) + ' of different classes were found in close proximity:'

        # counterfactuals
        j = 1
        for c in explanation.deltas.keys():
            item = str(j) + ') ' + str(explanation.rule.class_name) + '=' + str(explanation.crules[c][0].cons) + ': '
            text += '\n' + colored(item, 'blue')
            probs = {v: round(explanation.c_pred_proba[c][0][k],2) for k, v in class_map.items()}
            probs[explanation.crules[c][0].cons] = colored(str(probs[explanation.crules[c][0].cons]), 'blue')
            #print(*[str(k) + ':' + str(v) for k,v in probs.items()], sep=',')
            #print('\t\x1B[3mPredicted probabilities:\x1B[0m '+np.array2string(x.c_pred_proba[c][0], precision=2, separator=', '))
            text += '\n\t\x1B[3mPredicted probabilities:\x1B[0m '#+ *[str(k) + ': ' + str(v) + ', ' for k,v in probs.items()]
            for k,v in probs.items():
                text += str(k) + ': ' + str(v) + ', '
            text += '\n\t\x1B[3mChanges needed:\x1B[0m '
            for cond in explanation.deltas[c][0]:
                att = cond.att
                original = X.loc[i][att]
                text +='\n\t\t- '+str(cond) + '(original value:' + str(round(original, 2)) + ')'
            j +=1
        return text


    def explain_prediction(self, explanation, X, i):
        #x = res[d]['explanation'][i]
        n_c_classes = len(explanation.crules.keys())
        class_map = {k:v for k,v in enumerate(self.explainer.class_values)}

        #X_test = res[d]['original_data']

        # Rule
        text  = 'The instance '+ colored('x = '+str(i), 'green') + ' has been ' + colored('PREDICTED', 'green', attrs=['bold']) + '.\n' 
        text += 'It has been classified by the black-box as '+ colored(str(explanation.rule.class_name) + '='+ str(explanation.bb_pred), 'green') + ' with ' + colored(str(round(max(explanation.bb_pred_proba), 2)), 'green') +  ' predicted probability. '
        text += 'This prediction derived from the rule: ' + str(explanation.rule) + '.'
        
        text += '\nFurthermore ' + colored(str(n_c_classes)+' counterfactuals', 'blue', attrs=['bold']) + ' of different classes were found far from the decision boundary:'

        # counterfactuals
        j = 1
        for c in explanation.deltas.keys():
            item = str(j) + ') ' + str(explanation.rule.class_name) + '=' + str(explanation.crules[c][0].cons) + ': '
            text += '\n' + colored(item, 'blue')
            probs = {v: round(explanation.c_pred_proba[c][0][k],2) for k, v in class_map.items()}
            probs[explanation.crules[c][0].cons] = colored(str(probs[explanation.crules[c][0].cons]), 'blue')
            #print(*[str(k) + ':' + str(v) for k,v in probs.items()], sep=',')
            #print('\t\x1B[3mPredicted probabilities:\x1B[0m '+np.array2string(x.c_pred_proba[c][0], precision=2, separator=', '))
            text += '\n\t\x1B[3mPredicted probabilities:\x1B[0m '#+ *[str(k) + ': ' + str(v) + ', ' for k,v in probs.items()]
            for k,v in probs.items():
                text += str(k) + ': ' + str(v) + ', '
            text += '\n\t\x1B[3mChanges needed:\x1B[0m '
            for cond in explanation.deltas[c][0]:
                att = cond.att
                original = X.loc[i][att]
                text +='\n\t\t- '+str(cond) + '(original value:' + str(round(original, 2)) + ')'
            j +=1
        return text



    def evaluate_l2r(self, n_rejected, dist_dict, y_true, y_pred):
        r_list = list(range(len(y_true)+1))
        rejected_samples, an_dict, cq_dict, rq_dict, rej_class_report_dict = compute_rejection_policy(r_list, 
                                                                                                      dist_dict, 
                                                                                                      y_true, 
                                                                                                      y_pred
                                                                                                      )
        metric_dict = {
            'n_rejected'            : n_rejected,
            'classification_quality': cq_dict,
            'rejection_quality'     : rq_dict,
            'nonrejected_accuracy'  : an_dict,
            'rejected_samples'      : rejected_samples,
            'rej_class_report'      : rej_class_report_dict
        }  
    
        return metric_dict