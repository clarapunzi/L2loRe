import numpy as np

from joblib import Parallel, delayed
from Lib.LoreSA.surrogate import *

from sklearn.metrics import accuracy_score

from .explanation import SuperExplanation
from .rule import get_rule, get_counterfactual_rules, get_counterfactual_rules_supert, get_rule_supert
from .util import neuclidean
from Lib.LoreSA.lorem_new import LOREM
from Lib.LoreSA.discretizer import RMEPDiscretizer


# L2R- LOcal Rule-based Explanation Method
class LOREU(LOREM):

    def __init__(self, K, bb_predict, predict_proba, feature_names, class_name, class_values, numeric_columns, features_map,
                 neigh_type='genetic', K_transformed=None, categorical_use_prob=True, continuous_fun_estimation=False,
                 size=1000, ocr=0.1, multi_label=False, one_vs_rest=False, filter_crules=True, init_ngb_fn=True,
                 kernel_width=None, kernel=None, random_state=None, encdec = None, dataset = None, binary=False, discretize=True, verbose=False,
                 extreme_fidelity = False, uncertainty_thr = 0.5, uncertainty_metric = 'max', constraints = None, **kwargs):

        super(LOREU, self).__init__(K, bb_predict, predict_proba, feature_names, class_name, class_values, numeric_columns, features_map,
                 neigh_type=neigh_type, K_transformed=K_transformed, categorical_use_prob=categorical_use_prob, continuous_fun_estimation=continuous_fun_estimation,
                 size=size, ocr=ocr, multi_label=multi_label, one_vs_rest=one_vs_rest, filter_crules=filter_crules, init_ngb_fn=init_ngb_fn,
                 kernel_width=kernel_width, kernel=kernel, random_state=random_state, encdec=encdec, dataset=dataset, binary=binary, discretize=discretize, verbose=verbose,
                 extreme_fidelity=extreme_fidelity, constraints=constraints, **kwargs)

        self.uncertainty_metric = uncertainty_metric
        self.uncertainty_thr = uncertainty_thr


    # qui l'istanza arriva originale
    def explain_instance_stable(self, x, samples=100, use_weights=True, metric=neuclidean, runs=3, exemplar_num=5,
                                n_jobs=-1, prune_tree=False, single=False, extract_counterfactuals_by= 'min_distance', kwargs=None):

        if self.multi_label:
            print('Not yet implemented')
            raise Exception

        if self.encdec is not None:
            y = self.bb_predict(x.reshape(1, -1))
            x = self.encdec.enc(x, y)

        if isinstance(samples, int):
            if self.neigh_type == 'cfs':
                Z_list = self.multi_neighgen_fn_parallel(x, runs, samples, n_jobs)
            else:
                Z_list = self.multi_neighgen_fn(x, runs, samples, kwargs)
        else:
            Z_list = list()
            for z in samples:
                Z_list.append(np.array(z))

        Yb_list = list()
        Yb_proba_list = list()
        #print('la Z creata ', len(Z_list), Z_list[0])
        if self.encdec is not None:
            for Z in Z_list:
                Z = self.encdec.dec(Z)
                #print('Z decodificata ', Z)
                Z = np.nan_to_num(Z)
                Yb = self.bb_predict(Z)
                #print('la yb ', Counter(Yb))
                Yb_list.append(Yb)
        else:
            if single:
                Yb = self.bb_predict(Z_list)
                Yb_list.append(Yb)
                Yb_proba_list.append(self.bb_predict_proba(Z_list))
            else:
                for Z in Z_list:
                    Yb = self.bb_predict(Z)
                    Yb_list.append(Yb)
                    Yb_proba_list.append(self.bb_predict_proba(Z))

        if self.verbose:
            neigh_class_counts_list = list()
            for Yb in Yb_list:
                neigh_class, neigh_counts = np.unique(Yb, return_counts=True)
                neigh_class_counts = {self.class_values[k]: v for k, v in zip(neigh_class, neigh_counts)}
                neigh_class_counts_list.append(neigh_class_counts)

            for neigh_class_counts in neigh_class_counts_list:
                print('Synthetic neighborhood class counts %s' % neigh_class_counts)

        weights_list = list()
        if single:
            weights = None if not use_weights else self.__calculate_weights__(Z_list, metric)
            weights_list.append(weights)
        #print('la shape di z e come e fatta ', len(Z_list), Z[0].dtype)
        else:
            for Z in Z_list:
                # print('nel calcolo del peso', Z.dtype, Z.shape)
                weights = None if not use_weights else self.__calculate_weights__(Z, metric)
                weights_list.append(weights)

        if self.verbose:
            print('Learning local decision trees')

        # discretize the data employed for learning decision tree
        if self.discretize:
            if single:
                discr = RMEPDiscretizer()
                discr.fit(Z, Yb)
                Z_list = discr.transform(Z_list)
            else:
                Z = np.concatenate(Z_list)
                Yb = np.concatenate(Yb_list)
                Yb_proba = np.concatenate(Yb_proba_list)

                discr = RMEPDiscretizer()
                discr.fit(Z, Yb)
                temp = list()
                for Zl in Z_list:
                    temp.append(discr.transform(Zl))
                Z_list = temp
        # caso binario da Z e Y da bb
        if self.binary == 'binary_from_bb':
            surr = DecTree()
            weights = None if not use_weights else self.__calculate_weights__(Z, metric)
            superT = surr.learn_local_decision_tree(Z, Yb, weights, self.class_values)
            fidelity = superT.score(Z, Yb, sample_weight=weights)

        # caso binario da Z e Yb
        # caso n ario
        # caso binario da albero n ario
        else:
            #qui prima creo tutti i dt, che servono sia per unirli con metodo classico o altri
            dt_list = [DecTree() for i in range(runs)]
            dt_list = Parallel(n_jobs=n_jobs, verbose=self.verbose,prefer='threads')(
                delayed(t.learn_local_decision_tree)(Zl, Yb, weights, self.class_values, prune_tree=prune_tree)
                for Zl, Yb, weights, t in zip(Z_list, Yb_list, weights_list, dt_list))

            Z = np.concatenate(Z_list)
            Z = np.nan_to_num(Z)
            Yb = np.concatenate(Yb_list)
            Yb_proba = np.concatenate(Yb_proba_list)

            # caso binario da Z e Yb dei vari dt
            if self.binary == 'binary_from_dts':
                weights = None if not use_weights else self.__calculate_weights__(Z, metric)
                surr = DecTree()
                superT = surr.learn_local_decision_tree(Z, Yb, weights, self.class_values)
                fidelity = superT.score(Z, Yb, sample_weight=weights)

            # caso n ario
            # caso binario da albero n ario
            else:
                if self.verbose:
                    print('Pruning decision trees')
                surr = SuperTree()
                for t in dt_list:
                    surr.prune_duplicate_leaves(t)
                if self.verbose:
                    print('Merging decision trees')

                weights_list = list()
                for Zl in Z_list:
                    weights = None if not use_weights else self.__calculate_weights__(Zl, metric)
                    weights_list.append(weights)
                weights = np.concatenate(weights_list)
                n_features = list()
                for d in dt_list:
                    n_features.append(list(range(0, len(self.feature_names))))
                roots = np.array([surr.rec_buildTree(t, FI_used) for t, FI_used in zip(dt_list, n_features)])

                superT = surr.mergeDecisionTrees(roots, num_classes=np.unique(Yb).shape[0], verbose=False)

                if self.binary == 'binary_from_nari':
                    superT = surr.supert2b(superT, Z)
                    Yb = superT.predict(Z)
                    fidelity = superT.score(Z, Yb, sample_weight=weights)
                else:
                    Yz = superT.predict(Z)
                    fidelity = accuracy_score(Yb, Yz)

                if self.extreme_fidelity:
                    res = superT.predict(x)
                    if res != y:
                        raise Exception('The prediction of the surrogate model is different wrt the black box')

                if self.verbose:
                    print('Retrieving explanation')
        x = x.flatten()
        Yc = superT.predict(X=Z)
        if self.binary == 'binary_from_nari' or self.binary == 'binary_from_dts' or self.binary == 'binary_from_bb':
            rule = get_rule(x, self.bb_predict(x.reshape(1, -1)), superT, self.feature_names, self.class_name, self.class_values,
                                self.numeric_columns, encdec=self.encdec,
                                multi_label=self.multi_label)
        else:
            rule = get_rule_supert(x, superT, self.feature_names, self.class_name, self.class_values,
                                       self.numeric_columns,
                                       self.multi_label, encdec=self.encdec)
        if self.binary == 'binary_from_nari' or self.binary == 'binary_from_dts' or self.binary == 'binary_from_bb':
            #print('la shape di x che arriva fino alla get counter ', x, x.shape)
            Xc_final, crules, deltas, pred_proba_list = get_counterfactual_rules(x, Yc[0], superT, Z, Yc, self.feature_names,
                                                          self.class_name, self.class_values, self.numeric_columns,
                                                          self.features_map, self.features_map_inv, encdec=self.encdec,
                                                          filter_crules = self.filter_crules, 
                                                          bb_predict_proba = self.bb_predict_proba, uncertainty_thr = self.uncertainty_thr, uncertainty_metric = self.uncertainty_metric,
                                                          constraints= self.constraints, extract_counterfactuals_by=extract_counterfactuals_by, metric = metric)
        else:
            #print('la shaoe di x che arriva a get counter con super t', x, x.shape)
            Xc_final, crules, deltas, pred_proba_list = get_counterfactual_rules_supert(x, Yc[0], superT, Z, Yc, self.feature_names,
                                                                 self.class_name, self.class_values, self.numeric_columns,
                                                                 self.features_map, self.features_map_inv,
                                                                 filter_crules = self.filter_crules)

        # Feature Importance
        if self.binary:
            feature_importance, feature_importance_all = self.get_feature_importance_binary(superT, x)
            #exemplars_rec, cexemplars_rec = self.get_exemplars_cexemplars_binary(superT, x, exemplar_num)
        else:
            feature_importance, feature_importance_all = self.get_feature_importance_supert(superT, x, len(Yb))
            #exemplars_rec, cexemplars_rec = self.get_exemplars_cexemplars_supert(superT, x, exemplar_num)
        # Exemplar and Counter-exemplar

        # if exemplars_rec is not None:
        #     #print('entro con exemplars ', exemplars_rec, self.feature_names)
        #     exemplars = self.get_exemplars_str(exemplars_rec)
        # else:
        #     exemplars = 'None'
        # if cexemplars_rec is not None:
        #     cexemplars = self.get_exemplars_str(cexemplars_rec)
        # else:
        #     cexemplars = 'None'


        exp = SuperExplanation()
        exp.bb_pred = Yb[0]
        exp.dt_pred = Yc[0]
        exp.bb_pred_proba = Yb_proba[0]
        exp.rule = rule
        exp.crules = crules
        exp.deltas = deltas
        exp.Xc = Xc_final
        exp.c_pred_proba = pred_proba_list
        exp.dt = superT
        exp.fidelity = fidelity
        exp.feature_importance = feature_importance
        exp.feature_importance_all = feature_importance_all
        # exp.exemplars = exemplars
        # exp.cexemplars = cexemplars

        return exp
