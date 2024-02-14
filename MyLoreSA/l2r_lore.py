from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import gamma
from MyLoreSA.lorem_new import LOREU
from MyLoreSA.util import compute_rejection_policy, compute_distance_from_counterfactual
from tqdm import tqdm
import numpy as np
import pandas as pd
from termcolor import colored


class L2R_LORE_Prediction(object):

    def __init__(self, x=None, prediction=None, explanation=None):
        self.x = x
        self.prediction = prediction
        self.explanation = explanation

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


class L2R_LORE(BaseEstimator, ClassifierMixin):
     
    def __init__(self, base_clf, expl, n_neigh=150, extract_counterfactuals_by='min_distance', optimize_by='nonrejected_accuracy',
                 n_tau_candidates=100, max_rej_fraction=0.6, max_error=0.3, tau=0, base_score=0, max_score=0, rej_rate=0, counterfactual_rate=None):
        
        self.base_clf = base_clf
        self.explainer = expl
        self.n_neighbors = n_neigh
        self.extract_counterfactuals_by = extract_counterfactuals_by
        self.optimize_by = optimize_by
        self.n_tau_candidates = n_tau_candidates
        self.max_rej_fraction = max_rej_fraction
        self.max_error = max_error
        self.tau = tau
        self.base_score = base_score
        self.max_score = max_score
        self.rej_rate = rej_rate
        self.counterfactual_rate=counterfactual_rate

    
    def fit(self, X, y, sample=False, n_samples = 200):

        # STEP 1: BASE CLASSIFIER
        # Fit and get prediction of the base classifier
        self.base_clf.fit(X.values, y.values)
        y_pred = self.base_clf.predict(X.values)
        y_proba = self.base_clf.predict_proba(X.values)

        # STEP 2: COUNTERFACTUALS
        # Compute counterfactuals for all samples in X (train)
        counterfactual_indices = X.index
        sample_size = X.shape[0]

        if sample:
            s = pd.Series(y_pred == y).to_frame()
            s = s.groupby(self.explainer.class_name).sample(frac=n_samples/len(y_pred))
            counterfactual_indices = s.index
            sample_size = n_samples
            #x_list = X_test.loc[selected_indices] 
            #y_list = y_test.loc[selected_indices] 
        
        id_map = {j:i for i, j in enumerate(y.index)}
        expl_list = dict()
        for j in tqdm(counterfactual_indices):
            exp = self.explainer.explain_instance_stable(X.loc[j].values, 
                                                         samples=self.n_neighbors, 
                                                         extract_counterfactuals_by=self.extract_counterfactuals_by
                                                         )
            expl_list[j] = exp

        # compute distance from counterfactuals (minimum over all classes)
        dist_dict = compute_distance_from_counterfactual(X, expl_list)
        frac_found_counterfactuals = (sample_size - list(dist_dict.values()).count(np.inf) ) / sample_size
        self.counterfactual_rate = frac_found_counterfactuals

        # r_list contains the rejection fraction on which to evaluate the selective classifier
        r_list = list(range(sample_size+1))
        rejected_samples, an_list, cq_list, rq_list, rej_class_report_list = compute_rejection_policy(r_list, 
                                                                                                      dist_dict, 
                                                                                                      y.loc[counterfactual_indices], 
                                                                                                      [y_pred[id_map[i]] for i in counterfactual_indices]
                                                                                                      )
        metric_dict = {'classification_quality': cq_list,
                       'rejection_quality': rq_list,
                       'nonrejected_accuracy': an_list
                    }       
        
        # STEP 3: FIND OPTIMAL TAU
        # fit the rejection threshold tau
        base, tau, m_max, frac_rejected, n_rej_final = self.fit_tau(sample_size, 
                                                               dist_dict, 
                                                               an_list, 
                                                               metric_dict[self.optimize_by], 
                                                               self.n_tau_candidates, 
                                                               self.max_rej_fraction, 
                                                               self.max_error
                                                               )
        self.tau = tau
        self.base_score = base
        self.max_score = m_max
        self.rej_rate = frac_rejected

        print('Optimal rejection rate: ', frac_rejected)
        print('Optimal rejection threshold: ', tau)

        return self
    
    def predict(self, X):
        # output the class inferred by the classifier
        
        # STEP 1: BASE CLASSIFIER
        # Get prediction of the base classifier
        y_pred = self.base_clf.predict(X.values)
        y_proba = self.base_clf.predict_proba(X.values)

        # STEP 2: COUNTERFACTUALS
        # Compute counterfactuals for all samples in X (train)
        counterfactual_indices = X.index
        sample_size = X.shape[0]       

        expl_list = dict()
        
        for j in tqdm(counterfactual_indices):
            exp = self.explainer.explain_instance_stable(X.loc[j].values, 
                                                         samples=self.n_neighbors, 
                                                         extract_counterfactuals_by=self.extract_counterfactuals_by
                                                         )
            expl_list[j] = exp

        # compute distance from counterfactuals (minimum over all classes)
        dist_dict = compute_distance_from_counterfactual(X, expl_list)
        frac_found_counterfactuals = (sample_size - list(dist_dict.values()).count(np.inf) ) / sample_size
        print('Counterfactual rate: ', frac_found_counterfactuals)

        # STEP 3: REJECTION POLICY
        rejected_samples = [k for k, v in dist_dict.items() if v < self.tau]
        frac_rej = len(rejected_samples)/len(dist_dict.keys())
        print('Numeber of rejected samples: ', len(rejected_samples), ' (', frac_rej*100, '%)')

        # Final prediction list with -1 for all rejected samples        
        id_map = {i:j for i, j in enumerate(X.index)}
        rejected_samples_ids = [id_map[i] for i in rejected_samples]
         
        # STEP 4: EXPLANATION OF REJECTION
        preds = []
        for i in range(len(y_pred)):
            x = id_map[i]
            if x in rejected_samples:
                xai = self.explain_rejection(self, expl_list[j], X, x)
                preds.append(L2R_LORE_Prediction(x, -1, xai))
            else:
                preds.append(L2R_LORE_Prediction(x, y_pred[i]))

        return preds
    
    
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

        miscl_by_r = self.misprediction_rate(nonrejected_accuracy_list)
    
        m_max = 0
        tau = np.inf
        n_rej_final = 0
        #n =  X.shape[0]
        for t in tau_candidates:
            # check number of instances that will be rejected, i.e., all those with distance < t
            rej = [x for x, d in dist_dict.items() if d < t]
            n_rej = len(rej)
            rej_rate = n_rej / sample_size
        
            # constraints: 1) rejection rate below thr and 2) mispredictions below thr
            if (rej_rate < max_rej_fraction) & (miscl_by_r[n_rej] < max_error):
                m = metric_list[n_rej] 
                if m > m_max:
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