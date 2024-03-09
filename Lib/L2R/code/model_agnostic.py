# standard modules
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, train_test_split
import sklearn.metrics as skm
import copy
import pandas as pd

# pluginrule
class PlugInRule(ClassifierMixin, BaseEstimator):
    """
    Class for PlugIn algorithm.
    It takes as input a probabilistic classifier and it constructs the PlugIn estimator of Herbei and Wegkamp
    References

    Example
    >>> X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
    >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
    >>> clf = PlugInRule(model=LogisticRegression())
    >>> clf.fit(X_tr, y_tr)
    >>> preds = clf.predict(X_te)
    """

    def __init__(
        self,
        model,
        coverages: list = [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, .7],
        seed: int = 42,
    ):
        """

        :param model:
        :param coverages:
        :param seed:
        """
        self.model = model
        self.coverages = sorted(coverages, reverse=True)
        self.seed = seed
        self.thetas = None

    def fit(self, X, y, sample_weight=None):
        """

        :param X:
        :param y:
        :param sample_weight:
        :return:
        """

        # here we store the classes
        self.classes = np.unique(y)
        # here we split train and holdout
        self.model.fit(X, y)
        # quantiles

    def calibrate(self, X, confidence_function: str = "softmax", target_coverages = None):
        """
        :param X:
        :param confidence_function:
        :return:
        """
        if target_coverages is not None:
            self.coverages = target_coverages
        probas = self.model.predict_proba(X)
        if confidence_function == "softmax":
            self.confidence = "softmax"
            confs = np.max(probas, axis=1)
        else:
            raise NotImplementedError("Confidence function not yet implemented")
        self.quantiles = [1 - c for c in self.coverages]
        self.thetas = [np.quantile(confs, q) for q in self.quantiles]

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise AttributeError(
                "The original model does not have predict_proba method."
            )

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def qband(self, X, confidence_function: str = "softmax"):
        if self.thetas is not None:
            if self.confidence == "softmax":
                confs = np.max(self.predict_proba(X), axis=1)
            else:
                raise AttributeError(
                    "The original model does not have predict_proba method."
                )
            return np.digitize(confs, self.thetas)
        else:
            raise ValueError(
                "The model is not fitted yet. Please call the fit method before."
            )


class SCRoss(ClassifierMixin, BaseEstimator):
    """
    Class for SCRoss algorithm.
    It takes as input a probabilistic classifier and it constructs the SCross estimator of Pugnana and Ruggieri (2023)
    References

    Example
    >>> X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
    >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
    >>> clf = SCRoss(model=LogisticRegression())
    >>> clf.fit(X_tr, y_tr)
    >>> preds = clf.predict(X_te)
    """

    def __init__(
        self,
        model,
        cv: int = 5,
        coverages: list = [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, .7],
        seed: int = 42,
    ):
        self.cv = cv
        self.seed = seed
        self.coverages = sorted(coverages, reverse=True)
        self.thetas = None
        self.kmodels = [copy.deepcopy(model) for _ in range(self.cv)]
        self.model = copy.deepcopy(model)

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        confidence_function: str = "softmax",
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        self.classes = np.unique(y)
        z = []
        localthetas = []
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.seed)
        if n_jobs == 1:
            for i, (train_index, test_index) in enumerate(skf.split(X, y)):
                if isinstance(X, pd.DataFrame):
                    X_train = X.iloc[train_index]
                    X_test = X.iloc[test_index]
                else:
                    X_train = X[train_index]
                    X_test = X[test_index]
                if isinstance(y, pd.Series):
                    y_train = y.iloc[train_index]
                    # y_test = y.iloc[test_index]
                else:
                    y_train = y[train_index]
                    # y_test = y[test_index]
                self.kmodels[i].fit(X_train, y_train)
                if verbose:
                    print("{} out of {} models fitted".format(i, self.cv))
                # quantiles
                probas = self.kmodels[i].predict_proba(X_test)
                if confidence_function == "softmax":
                    self.confidence = "softmax"
                    confs = np.max(probas, axis=1)
                else:
                    raise NotImplementedError(
                        "Confidence function not yet implemented."
                    )
                z.append(confs)
        elif n_jobs == -1:
            raise NotImplementedError("Parallelization not yet implemented.")
        elif n_jobs > 1:
            raise NotImplementedError("Parallelization not yet implemented.")
        self.confs = np.concatenate(z).ravel()
        self.model.fit(X, y)

    def calibrate(self, X=None, quantile_est: str = "knight", target_coverages = None):
        if target_coverages is not None:
            self.coverages = target_coverages
        self.quantiles = [1 - cov for cov in self.coverages]
        if quantile_est == "knight":
            sub_confs_1, sub_confs_2 = train_test_split(
                self.confs, test_size=0.5, random_state=42
            )
            tau = 1 / np.sqrt(2)
            self.thetas = [
                (
                    tau * np.quantile(self.confs, q)
                    + (1 - tau)
                    * (
                        0.5 * np.quantile(sub_confs_1, q)
                        + 0.5 * np.quantile(sub_confs_2, q)
                    )
                )
                for q in self.quantiles
            ]
        elif quantile_est == "standard":
            self.thetas = [np.quantile(self.confs, q) for q in self.quantiles]
        else:
            raise NotImplementedError("Quantile estimator not yet implemented")

    def predict_proba(self, X, ensembling=False):
        if ensembling:
            if hasattr(self.kmodels[0], "predict_proba"):
                return np.mean([clf.predict_proba(X) for clf in self.kmodels], axis=0)
            else:
                raise AttributeError(
                    "The original model does not have predict_proba method."
                )
        else:
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(X)
            else:
                raise AttributeError(
                    "The original model does not have predict_proba method."
                )

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def qband(self, X):
        if self.thetas is not None:
            if self.confidence == "softmax":
                confs = np.max(self.predict_proba(X), axis=1)
            else:
                raise AttributeError(
                    "The original model does not have predict_proba method."
                )
            return np.digitize(confs, self.thetas)
        else:
            raise ValueError(
                "The model is not fitted yet. Please call the fit method before."
            )


# pluginruleAUC based
class PlugInRuleAUC(ClassifierMixin, BaseEstimator):
    """
    Class for PlugInAUC algorithm.
    It takes as input a probabilistic classifier and it constructs the PlugInAUC estimator of Pugnana and Ruggieri (2023)
    References

    Example
    >>> X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
    >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
    >>> clf = PlugInRuleAUC(model=LogisticRegression())
    >>> clf.fit(X_tr, y_tr)
    >>> preds = clf.predict(X_te)
    """

    def __init__(
        self,
        model,
        coverages: list = [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, .7],
        seed: int = 42,
    ):
        """

        :param model:
        :param coverages:
        :param seed:
        """
        self.model = model
        self.coverages = sorted(coverages, reverse=True)
        self.seed = seed
        self.thetas = None

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        test_perc: float = 0.1,
    ):
        self.classes = np.unique(y)
        if len(self.classes) > 2:
            raise NotImplementedError(
                "PlugInAUC for multiclass classification is not implemented yet."
            )
        self.model.fit(X, y)
        # quantiles

    def calibrate(self, X, y, target_coverages = None):
        if target_coverages is not None:
            self.coverages = target_coverages
        y_scores = self.model.predict_proba(X)[:, 1]
        auc_roc = skm.roc_auc_score(y, y_scores)
        n, npos = len(y), np.sum(y)
        pneg = 1 - np.mean(y)
        u_pos = int(auc_roc * pneg * n)
        pos_sorted = np.argsort(y_scores)
        if isinstance(y, pd.Series):
            tp = np.cumsum(y.iloc[pos_sorted[::-1]])
        else:
            tp = np.cumsum(y[pos_sorted[::-1]])
        l_pos = n - np.searchsorted(tp, auc_roc * npos + 1, side="right")
        self.quantiles = [1 - cov for cov in self.coverages]
        pos = (u_pos + l_pos) / 2
        self.thetas = []
        for q in self.quantiles:
            delta = int(n * q / 2)
            t1 = y_scores[pos_sorted[max(0, round(pos - delta))]]
            t2 = y_scores[pos_sorted[min(round(pos + delta), n - 1)]]
            self.thetas.append([t1, t2])
            # print('Local thetas:', [t1, t2])

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise AttributeError(
                "The original model does not have predict_proba method."
            )

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def qband(self, X):
        confs = self.predict_proba(X)[:, 1]
        m = len(self.quantiles)
        res = np.zeros(len(confs)) + m
        for i, t in enumerate(reversed(self.thetas)):
            t1, t2 = t[0], t[1]
            res[((t1 <= confs) & (confs <= t2))] = m - i - 1
        return res


