import numpy as np
import pandas as pd
import scipy.stats as st

#import encdec
from Lib.LoreSA.encdec import *
from scipy.spatial.distance import jaccard
import warnings

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from MyLoreSA.metrics import nonrejected_accuracy, classification_quality, rejection_quality, rejection_classification_report

from MyLoreSA.datamanager import prepare_dataset

def vector2dict(x, feature_names):
    return {k: v for k, v in zip(feature_names, x)}

def neuclidean(x, y):
    return 0.5 * np.var(x - y) / (np.var(x) + np.var(y))

def record2str(x, feature_names, numeric_columns, encdec=None):
    xd = vector2dict(x, feature_names)
    if encdec:
        x_dec = encdec.dec(x)
    s = '{ '
    for att, val in xd.items():
        #print('att ', att)
        #print('att val ', val)
        if att not in numeric_columns and val == 0.0:
            continue
        elif att in numeric_columns:
            s += '%s = %s, ' % (att, val)
        elif encdec is None:
            s += '%s = %s' % (att, val)
        else:
            if isinstance(encdec, OneHotEnc):
            #if type(encdec) is encdec.OneHotEnc:
                att_split = att.split('=')
                s += '%s = %s, ' % (att_split[0], att_split[1])
            if isinstance(encdec, MyTargetEnc):
                #caso in cui devo riprendere il valore originale
                ind = x.index(val)
                s += '%s = %s' % x_dec[ind]
            '''
            if encdec:
                att_split = [None]*2
                att_split[0] = att
                try:
                    ind = feature_names.tolist().index(att)
                except:
                    print('in except', feature_names)
                    ind = feature_names.index(att)
                att_split[1] = x_dec[0][ind]
            else:
                att_split = att.split('=')
            #print('att split ', att_split)
            s += '%s = %s, ' % (att_split[0], att_split[1])'''

    s = s[:-2] + ' }'
    return s


def multilabel2str(y, class_name):
    mstr = ', '.join([class_name[i] for i in range(len(y)) if y[i] == 1.0])
    return mstr


def multi_dt_predict(X, dt_list):
    nbr_labels = len(dt_list)
    Y = np.zeros((X.shape[0], nbr_labels))
    for l in range(nbr_labels):
        Y[:, l] = dt_list[l].predict(X)
    return Y

def mixed_distance_idx(x, y, idx, ddist=jaccard, cdist=neuclidean):

    dim = len(x)
    xc, xd = x[:idx], x[idx:]
    yc, yd = y[:idx], y[idx:]

    wc = 1.0 * len(xc) / dim
    cd = cdist(xc, yc)

    wd = 1.0 * len(xd) / dim
    dd = ddist(xd, yd)

    return wd * dd + wc * cd
def calculate_feature_values(X, numeric_columns_index, categorical_use_prob=False, continuous_fun_estimation=False,
                             size=1000):

    feature_values = list()
    for i in range(X.shape[1]):
        values = X[:, i]
        unique_values = np.unique(values)
        if len(unique_values) == 1:
            new_values = np.array([unique_values[0]] * size)
        else:
            if i in numeric_columns_index:
                values = values.astype(float)
                if continuous_fun_estimation:
                    new_values = get_distr_values(values, size)
                else:  # suppose is gaussian
                    mu = float(np.mean(values))
                    sigma = float(np.std(values))
                    new_values = np.random.normal(mu, sigma, size)
                new_values = np.concatenate((values, new_values), axis=0)
            else:
                if categorical_use_prob:
                    diff_values, counts = np.unique(values, return_counts=True)
                    prob = 1.0 * counts / np.sum(counts)
                    new_values = np.random.choice(diff_values, size=size, p=prob)
                else:  # uniform distribution
                    diff_values = unique_values
                    new_values = diff_values

        feature_values.append(new_values)
    return feature_values


def get_distr_values(x, size=1000):
    nbr_bins = int(np.round(estimate_nbr_bins(x)))
    name, params = best_fit_distribution(x, nbr_bins)
    # print(name, params)
    dist = getattr(st, name)

    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    distr_values = np.linspace(start, end, size)

    return distr_values


# Distributions to check
DISTRIBUTIONS = [st.uniform, st.exponweib, st.expon, st.expon, st.gamma, st.beta, st.alpha,
                 st.chi, st.chi2, st.laplace, st.lognorm, st.norm, st.powerlaw] #st.dweibull,


def freedman_diaconis(x):
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    n = len(x)
    h = max(2.0 * iqr / n**(1.0/3.0), 1)
    k = np.ceil((np.max(x) - np.min(x))/h)
    return k


def struges(x):
    n = len(x)
    k = np.ceil(np.log2(n)) + 1
    return k


def estimate_nbr_bins(x):
    if len(x) == 1:
        return 1
    k_fd = freedman_diaconis(x) if len(x) > 2 else 1
    k_struges = struges(x)
    if k_fd == float('inf') or np.isnan(k_fd):
        k_fd = np.sqrt(len(x))
    k = max(k_fd, k_struges)
    return k


# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
                #print 'aaa'
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                except Exception:
                    pass

                # identify if this distribution is better
                # print distribution.name, sse
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return best_distribution.name, best_params


# def amp2math(c):
#     if '&le;' in c:
#         idx = c.find('&le;')
#         cnew = '%s%s%s' % (c[:idx], '<=', c[idx + 4:])
#         return cnew
#     elif '&lt;' in c:
#         idx = c.find('&lt;')
#         cnew = '%s%s%s' % (c[:idx], '<', c[idx + 4:])
#         return cnew
#     elif '&gl;' in c:
#         idx = c.find('&gl;')
#         cnew = '%s%s%s' % (c[:idx], '>=', c[idx + 4:])
#         return cnew
#     elif '&gt;' in c:
#         idx = c.find('&gt;')
#         cnew = '%s%s%s' % (c[:idx], '>', c[idx + 4:])
#         return cnew
#     return c
#
#
# def math2amp(c):
#     if '<=' in c:
#         idx = c.find('<=')
#         cnew = '%s%s%s' % (c[:idx], '&le;', c[idx + 2:])
#         return cnew
#     elif '<' in c:
#         idx = c.find('<')
#         cnew = '%s%s%s' % (c[:idx], '&lt;', c[idx + 1:])
#         return cnew
#     elif '>=' in c:
#         idx = c.find('>=')
#         cnew = '%s%s%s' % (c[:idx], '&gl;', c[idx + 2:])
#         return cnew
#     elif '>' in c:
#         idx = c.find('>')
#         cnew = '%s%s%s' % (c[:idx], '&gt;', c[idx + 1:])
#         return cnew
#     return c


def sigmoid(x, x0=0.5, k=10.0, L=1.0):
    """
    A logistic function or logistic curve is a common "S" shape (sigmoid curve

    :param x: value to transform
    :param x0: the x-value of the sigmoid's midpoint
    :param k: the curve's maximum value
    :param L: the steepness of the curve
    :return: sigmoid of x
    """
    return L / (1.0 + np.exp(-k * (x - x0)))


def neuclidean(x, y):
    return 0.5 * np.var(x - y) / (np.var(x) + np.var(y))


def nmeandev(x, y):  # normalized mean deviation
    return np.mean(np.abs(x-y)/np.max([np.abs(x), np.abs(y)], axis=0))


def get_df_stats(df, target):
    stats = dict()
    stats['n_samples'] = df.shape[0]
    stats['n_features'] = df.shape[1]
    n_labels = df[target].nunique()
    stats['n_target_classes'] = n_labels
    if n_labels > 2:
        stats['classification_type'] = 'multiclass'
    else:
        stats['classification_type'] = 'binary'
    df_no_y = df.copy()
    df_no_y.drop(columns=[target], inplace=True)
    stats['n_numerical_features'] =  len(df_no_y.select_dtypes(include='number').columns)
    stats['n_categorical_features'] = len(df_no_y.select_dtypes(exclude='number').columns)

    return stats

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

def get_tuned_classifier(d, clf, clf_dict, params_dict):
    best_clf = clf_dict[clf]
    best_clf.set_params(**params_dict[d][clf])
    return best_clf

def get_classification_metrics(y_true, y_pred, y_proba, binary):
    metrics = dict()
    if binary:
          metrics['accuracy_score'] = accuracy_score(y_true, y_pred)
          metrics['f1_score'] = f1_score(y_true, y_pred)
          metrics['roc_auc_score'] = roc_auc_score(y_true, y_proba[:,1])
    else:
          # multiclass classification
          metrics['accuracy_score'] = accuracy_score(y_true, y_pred)
          metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
          metrics['roc_auc_score'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
    return metrics

# Load and transform dataset 
def load_datasets(source_file_dict):

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
    return df_dict
          

def transform_datasets(df_dict, class_field_dict):

    # Transform datasets
    df_dict_new = dict()
    for k, v in df_dict.items():
        df, feature_names, class_values, numeric_columns, categorical_columns, rdf, real_feature_names, features_map = prepare_dataset(v, class_field_dict[k], 'onehot')
        df_dict_new[k] = [df, feature_names, class_values, numeric_columns, categorical_columns, rdf, real_feature_names, features_map]

    return df_dict_new

def compute_rejection_policy(r_list, dist_dict, y_true, y_pred):

    rej_class_report_list = []
    an_list = []
    cq_list = []
    rq_list = []
    rejected_samples = dict()
    n = len(y_true)
    id_map = {i:j for i, j in enumerate(dist_dict.keys())}
    for r_num in r_list:
        r_frac = r_num / n     # fraction of rejected samples over total samples
        # define rejection list
        rejection_list = [0]*n 
        rejected_x = [] 
        if r_num > 0:
            dist_values = np.array(list(dist_dict.values()))
            #id_to_rej = [k for k, v in sorted(dist_dict.items(), key=lambda x:x[1])][:r_num]
            id_to_rej = np.argpartition(dist_values, r_num - 1)[:r_num]  # 0 to k indeces of min values are at the front of the array 
            for i in id_to_rej:
                rejection_list[i] = 1 
                rejected_x.append(id_map[i])

        # get the rejection-classification report
        correct_nonrejected, correct_rejected, miscl_nonrejected, miscl_rejected, df = rejection_classification_report(y_true, y_pred, rejection_list)

        # get performance metrics
        #AN = nonrejected_accuracy(correct_nonrejected, n, r_frac)
        AN = nonrejected_accuracy(correct_nonrejected, miscl_nonrejected)
        #CQ = classification_quality(correct_nonrejected, n, r_frac)
        CQ = classification_quality(correct_nonrejected, miscl_rejected, n)
        #RQ = rejection_quality_by_r(correct_nonrejected, n, r_frac)
        RQ = rejection_quality(correct_rejected, correct_nonrejected, miscl_rejected, miscl_nonrejected)

        # save results
        rej_class_report_list.append(df)
        rejected_samples[r_num] = rejected_x
        an_list.append(AN)
        cq_list.append(CQ)
        rq_list.append(RQ)

    return rejected_samples, an_list, cq_list, rq_list, rej_class_report_list


def compute_distance_from_counterfactual(X_test, expl_list):

    # create a list of [(x, xc)]
    # compute the distance for each of them
    # sort by distance and reject based on a rejection fraction
    dist_dict = dict()
    for i in expl_list.keys():
        d = np.inf
        # iterate over different classes
        for c in expl_list[i].Xc.keys():
            # iterate over all counterfactuals per class 
            for countf in expl_list[i].Xc[c]:
                dist = cdist(X_test.loc[i].values.reshape(1, -1) , countf.reshape(1, -1))[0]
                if dist < d:
                    d = dist
        dist_dict[i] = float(d)
    return dist_dict