import numpy as np
from Lib.LoreSA.surrogate import *
from .util import vector2dict, multilabel2str, neuclidean
from .metrics import distance_cont_cat
from Lib.LoreSA.rule import *



def filter_uncertainty(Z, unc_thr, bb_predict_proba, uncertainty_metric):
    Z1 = Z
    Y = bb_predict_proba(Z)
    if uncertainty_metric=='max':
        Ymax = np.array(list(max(i) for i in Y))
        Z1 = Z[np.where(Ymax > unc_thr)[0]]
    elif uncertainty_metric == 'delta':
        probs = np.array(list(np.sort(y)[::-1] for y in Y))
        delta = np.apply_along_axis(lambda x: x[0]-x[1], 1, probs)
        Z1 = Z[np.where(delta > unc_thr)[0]]
    return Z1


def update_crules_by_min_sc(qlen, clen, xc, crule, delta, Xc_final, crule_list, delta_list, pred_proba_list, bb_predict_proba):
    if qlen < clen:
        clen = qlen
        Xc_final = [xc]
        crule_list = [crule]
        delta_list = [delta]
        pred_proba_list = [bb_predict_proba(xc.reshape(1, -1))[0]]
    elif qlen == clen:
        if delta not in delta_list:
            Xc_final.append(xc)
            crule_list.append(crule)
            delta_list.append(delta)
            pred_proba_list.append(bb_predict_proba(xc.reshape(1, -1))[0])
    return qlen, clen, xc, crule, delta, Xc_final, crule_list, delta_list, pred_proba_list
    
def update_crules_by_min_feature_distance(cfdist, x, xc, crule, delta, Xc_final, crule_list, delta_list, 
                                          metric, pred_proba_list, bb_predict_proba, numeric_columns, categorical_columns):    
    
    if metric == 'euclidean':
        # compute euclidean distance
        qdist = cdist(x.reshape(1, -1), xc.reshape(1, -1), metric=metric).ravel()[0]
    else:
        # compute metric differently on continuous and categorical variables, then combine
        qdist = distance_cont_cat(x, xc, numeric_columns, categorical_columns)
        
    if qdist < cfdist:
        cfdist = qdist
        Xc_final = [xc]
        crule_list = [crule]
        delta_list = [delta]
        pred_proba_list = [bb_predict_proba(xc.reshape(1, -1))[0]]
    elif qdist == cfdist:
        if delta not in delta_list:
            Xc_final.append(xc)
            crule_list.append(crule)
            delta_list.append(delta)
            pred_proba_list.append(bb_predict_proba(xc.reshape(1, -1))[0])
    return cfdist, xc, crule, delta, Xc_final, crule_list, delta_list, pred_proba_list


def get_counterfactual_rules(x, y, dt, Z, Y, feature_names, class_name, class_values, numeric_columns, categorical_columns, features_map,
                             features_map_inv, multi_label=False, encdec=None, filter_crules = None, bb_predict_proba = None,
                             uncertainty_thr = 0.5, uncertainty_metric = 'max', constraints=None, unadmittible_features=None, 
                             extract_counterfactuals_by='min_sc', metric = neuclidean):
    
    xd = vector2dict(x, feature_names)

    # keep only instancies with different class
    Z_diff = Z[np.where(Y != y)[0]]
    Y_diff = Y[np.where(Y != y)[0]]
    # keep track of counterfactual classes

    # keep only instances with high predicted probability
    # Z1 = filter_uncertainty(Z1, unc_thr, bb_predict_proba, uncertainty_metric)
    Xc_final_dict = dict()
    crule_dict = dict()
    delta_dict = dict()
    pred_proba_dict = dict()
    clen_dict, cfdist_dict = dict(), dict()

    for i in range(len(Z_diff)):
        z = Z_diff[i]
        y_z = Y_diff[i]
        if y_z == y:
            print('Error: same y!!')
            continue
        crule = get_rule(z, y, dt, feature_names, class_name, class_values, numeric_columns,encdec, multi_label)
        delta, qlen = get_falsified_conditions(xd, crule)
        if unadmittible_features != None:
            is_feasible = check_feasibility_of_falsified_conditions(delta, unadmittible_features)
            if not is_feasible:
                continue
        if constraints is not None:
            to_remove = list()
            for p in crule.premises:
                if p.att in constraints.keys():
                    if p.op == constraints[p.att]['op']:
                        if p.thr > constraints[p.att]['thr']:
                            break
                            #caso corretto
                        else:
                            to_remove.append()


        if filter_crules is not None:
            xc = apply_counterfactual(x, delta, feature_names, features_map, features_map_inv, numeric_columns)
            # first check: prediction probabilities of the counterfactual
            bb_outcomec_probs = bb_predict_proba(xc.reshape(1,-1))[0]
            if uncertainty_metric=='max':
                if np.max(bb_outcomec_probs) < uncertainty_thr:
                    continue
            elif uncertainty_metric == 'ratio':
                vals = np.sort(bb_outcomec_probs)[::-1]
                if vals[1] > 0:
                    ratio = vals[0] / vals[1]
                    # small values of top-2 class ratio correspond to high uncertainty
                    if ratio < uncertainty_thr: 
                        continue
            else:
                print('Uncertainty metric not yet implemented')

            bb_outcomec = filter_crules(xc.reshape(1, -1))[0]
            bb_outcomec = class_values[bb_outcomec] if isinstance(class_name, str) else multilabel2str(bb_outcomec,
                                                                                                       class_values)
            dt_outcomec = crule.cons
            if bb_outcomec == dt_outcomec:
                # keep track of different counterfactual classes
                if y_z in Xc_final_dict.keys():
                    Xc_final = Xc_final_dict[y_z]
                    crule_list = crule_dict[y_z]
                    delta_list = delta_dict[y_z]
                    pred_proba_list = pred_proba_dict[y_z]
                else:
                    Xc_final = []
                    crule_list = []
                    delta_list = []
                    pred_proba_list = []
                if extract_counterfactuals_by == 'min_sc':
                    if y_z in clen_dict.keys():
                        clen = clen_dict[y_z]
                    else:
                        clen = np.inf
                    qlen, clen, xc, crule, delta, Xc_final, crule_list, delta_list, pred_proba_list = update_crules_by_min_sc(qlen, clen, xc, crule, delta, Xc_final, crule_list, delta_list, pred_proba_list, bb_predict_proba)
                    clen_dict[y_z] = clen
                elif extract_counterfactuals_by == 'min_distance':
                    if y_z in cfdist_dict.keys():
                        cfdist = cfdist_dict[y_z]
                    else:
                        cfdist = np.inf
                    cfdist, xc, crule, delta, Xc_final, crule_list, delta_list, pred_proba_list = update_crules_by_min_feature_distance(cfdist, x, xc, crule, delta, Xc_final, crule_list, delta_list, metric, pred_proba_list, bb_predict_proba, numeric_columns, categorical_columns)
                    cfdist_dict[y_z] = cfdist
                else:
                    print('Type of counterfactual not yet implemented')
                
                Xc_final_dict[y_z] = Xc_final
                crule_dict[y_z] = crule_list
                delta_dict[y_z] = delta_list
                pred_proba_dict[y_z] = pred_proba_list
                
        
        else:
            # keep track of different counterfactual classes
            if y_z in Xc_final_dict.keys():
                Xc_final = Xc_final_dict[y_z]
                crule_list = crule_dict[y_z]
                delta_list = delta_dict[y_z]
                pred_proba_list = pred_proba_dict[y_z]
            else:
                Xc_final = []
                crule_list = []
                delta_list = []
                pred_proba_list = []
            if extract_counterfactuals_by == 'min_sc':
                if y_z in clen_dict.keys():
                    clen = clen_dict[y_z]
                else:
                    clen = np.inf
                qlen, clen, xc, crule, delta, Xc_final, crule_list, delta_list = update_crules_by_min_sc(qlen, clen, np.array([]), crule, delta, Xc_final, crule_list, delta_list)
                clen_dict[y_z] = clen
            elif extract_counterfactuals_by == 'min_distance':
                if y_z in cfdist_dict.keys():
                    cfdist = cfdist_dict[y_z]
                else:
                    cfdist = np.inf
                cfdist, xc, crule, delta, Xc_final, crule_list, delta_list = update_crules_by_min_feature_distance(cfdist, x, np.array([]), crule, delta, Xc_final, crule_list, delta_list, metric, numeric_columns, categorical_columns)
                cfdist_dict[y_z] = cfdist
            else:
                print('Type of counterfactual not yet implemented')

            Xc_final_dict[y_z] = Xc_final
            crule_dict[y_z] = crule_list
            delta_dict[y_z] = delta_list
            pred_proba_dict[y_z] = pred_proba_list
        

    return Xc_final_dict, crule_dict, delta_dict, pred_proba_dict


def get_counterfactual_rules_supert(x, y, dt, Z, Y, feature_names, class_name, class_values, numeric_columns, features_map,
                             features_map_inv, multi_label=False, filter_crules = None, encdec = None, unadmittible_features=None):
    clen = np.inf
    crule_list = list()
    delta_list = list()
    Z1 = Z[np.where(Y != y)[0]]
    xd = vector2dict(x, feature_names)
    Z1_final = []
    for z in Z1:
        crule = get_rule_supert(z, dt, feature_names, class_name, class_values, numeric_columns, multi_label, encdec=encdec)
        delta, qlen = get_falsified_conditions(xd, crule)
        if unadmittible_features != None:
            is_feasible = check_feasibility_of_falsified_conditions(delta, unadmittible_features)
            if not is_feasible:
                continue

        if filter_crules is not None:
            xc = apply_counterfactual_supert(x, delta, feature_names, features_map, features_map_inv, numeric_columns)
            bb_outcomec = filter_crules(xc.reshape(1, -1))[0]
            bb_outcomec = class_values[bb_outcomec] if isinstance(class_name, str) else multilabel2str(bb_outcomec,
                                                                                                       class_values)
            dt_outcomec = crule.cons

            if bb_outcomec == dt_outcomec:
                if qlen < clen:
                    clen = qlen
                    Z1_final = [z]
                    crule_list = [crule]
                    delta_list = [delta]
                elif qlen == clen:
                    if delta not in delta_list:
                        Z1_final.append(z)
                        crule_list.append(crule)
                        delta_list.append(delta)
        else:
            if qlen < clen:
                clen = qlen
                Z1_final = [z]
                crule_list = [crule]
                delta_list = [delta]
            elif qlen == clen:
                if delta not in delta_list:
                    Z1_final.append(z)
                    crule_list.append(crule)
                    delta_list.append(delta)
    return Z1_final, crule_list, delta_list, None


