from densratio import densratio
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.optimize import *
from functools import partial
from cvxopt import matrix, solvers
from .pykliep import DensityRatioEstimator


solvers.options['show_progress'] = False

"""
All reweighting functions have a likelihoods argument request. However, for importance learning and
baselines, it is not used. All methods which perform reweighting have an option for weights normalization and
weight clipping. However, some of these methods already perform regularization by themselves.
"""


# old name, it is referred to as PROB in the paper
def adversarial_reweighting(likelihoods, df_train, normalize=True, model=RandomForestClassifier(), oob=False):
    adv_label = [1 if df_train["batch"].iloc[i] == df_train["batch"].values[-1] else 0 for i in df_train.index]
    try:

        model.fit(X=df_train.loc[:, df_train.columns != "batch"], y=adv_label, oob_score=oob)

        if not oob:
            weights = model.predict_proba(df_train.iloc[:, :-1])[:, 1]
        else:
            weights = model.oob_decision_function_[:, 1]

        if normalize:
            weights = weights / np.linalg.norm(weights, 1) * len(weights)

        return weights

    except:
        print("Error in adversarial reweighting")
        return np.ones(df_train.shape[0])


def compute_rbf(X, Z, sigma=1.0):
    K = np.zeros((X.shape[0], Z.shape[0]), dtype=float)
    for i, vx in enumerate(X):
        K[i, :] = np.exp(-np.sum((vx - Z) ** 2, axis=1) / (2.0 * sigma))
    return K


def kmm_reweighting(likelihoods, df_train, kern='rbf', B=1.0, eps=None, normalize=False, clipper=None):

    X = np.array(df_train[df_train.batch == df_train.batch.values[-1]].loc[:, df_train.columns != "batch"])  #X=new data
    Z = np.array(df_train[df_train.batch != df_train.batch.values[-1]].loc[:, df_train.columns != "batch"])  #Z=old data
    nx = X.shape[0]
    nz = Z.shape[0]
    if eps == None:
        eps = B / nz**0.5
    if kern == 'lin':
        K = np.dot(Z, Z.T)
        kappa = np.sum(np.dot(Z, X.T) * float(nz) / float(nx), axis=1)
    elif kern == 'rbf':
        K = compute_rbf(Z, Z)
        kappa = np.sum(compute_rbf(Z, X), axis=1) * float(nz) / float(nx)
    else:
        raise ValueError('unknown kernel')

    K = matrix(K)
    kappa = matrix(kappa)
    G = matrix(np.r_[np.ones((1, nz)), -np.ones((1, nz)), np.eye(nz), -np.eye(nz)])
    h = matrix(np.r_[nz * (1 + eps), nz * (eps - 1), B * np.ones((nz,)), np.zeros((nz,))])

    sol = solvers.qp(K, -kappa, G, h)
    weights = np.array(sol['x'])

    #print(np.max(weights), np.mean(weights))

    weights = weights.reshape(1, -1)[0]
    weights = np.append(np.array(weights), np.ones(X.shape[0]))

    if clipper:
        weights = np.clip(weights, 0, clipper)

    if normalize:
        weights = weights / np.linalg.norm(weights, 1) * len(weights)

    return weights


def kliep_reweighting(likelihoods, df_train, normalize=False, clipper=None):
    X_train = np.array(
        df_train[df_train.batch == df_train.batch.values[-1]].loc[:, df_train.columns != "batch"])
    X_test = np.array(
        df_train[df_train.batch != df_train.batch.values[-1]].loc[:, df_train.columns != "batch"])

    kliep = DensityRatioEstimator(max_iter=1000, num_params=[.1, .2], epsilon=1e-4, cv=3,
                                  sigmas=[.01, .25, .5, .75], random_state=None, verbose=0) #.1, 1
    kliep.fit(X_train, X_test)  # keyword arguments are X_train and X_test
    weights = kliep.predict(np.array(df_train.loc[:, df_train.columns != "batch"]))

    #print(np.max(weights), np.mean(weights))

    weights = np.array(weights.reshape(1, -1)[0])

    if clipper:
        weights = np.clip(weights, 0, clipper)

    if normalize:
        weights = weights / np.linalg.norm(weights, 1) * len(weights)

    return weights


def multiple_reweighting(likelihoods, df_train, clipper=100, normalize=True):
    # here we simply reweight considering a balance heuristics
    denominator_array = np.array(
        [np.exp(likelihoods[i - 1].score_samples(df_train.loc[:, df_train.columns != "batch"]))
         for i in range(0, len(likelihoods))])

    denominator = np.mean(denominator_array, axis=0)

    weights = np.exp(likelihoods[-1].score_samples(df_train.loc[:, df_train.columns != "batch"])) / denominator

    if clipper:
        weights = np.clip(weights, 0, clipper)

    if normalize:
        weights = weights / np.linalg.norm(weights, 1) * len(weights)

    return weights


def window_multiple_reweighting(k, likelihoods, df_train, clipper=100, normalize=True):
    denominator = np.exp(
        likelihoods[0].score_samples((df_train[df_train.batch == 0].loc[:, df_train.columns != "batch"])))

    for i in range(1, len(likelihoods)):
        denominator = np.append(denominator,
                                np.mean([np.exp(likelihoods[j].score_samples(df_train[df_train.batch == i].loc[:,
                                                                             df_train.columns != "batch"]))
                                         for j in range(max(i - k, 0), min(i + k, len(likelihoods)))], axis=0))

    weights = np.exp(likelihoods[-1].score_samples(df_train.loc[:, df_train.columns != "batch"])) / (denominator + 0.01)

    if clipper:
        weights = np.clip(weights, 0, clipper)
    if normalize:
        weights = weights / np.linalg.norm(weights, 1) * len(weights)

    return weights


def direct_reweighting(likelihoods, df_train, clipper=100, normalize=True):
    weights = np.exp(likelihoods[-1].score_samples(df_train.loc[:, df_train.columns != "batch"]))
    weights = weights / np.linalg.norm(weights)

    if clipper:
        weights = np.clip(weights, 0, clipper)

    if normalize:
        weights = weights / np.linalg.norm(weights, 1) * len(weights)

    return weights


def reweighting(likelihoods, df_train, clipper=100, normalize=True, epsilon=1e-16):
    # normalize = self normalising importance weighting
    weights = []
    # it is possible it can be made faster
    for i in np.unique(df_train["batch"].values):
        local_samples = df_train[df_train.batch == i].loc[:, df_train.columns != "batch"]
        weights = np.append(weights, np.exp(likelihoods[-1].score_samples(local_samples)) /
                            (np.exp(likelihoods[i].score_samples(local_samples))+epsilon))

    if clipper:
        weights = np.clip(weights, 0, clipper)

    if normalize:
        weights = weights / np.linalg.norm(weights, 1) * len(weights)

    return weights


def optimal_lambda(weights, s, delta, l_):
    if l_ < 0 or l_ > 1:
        return -10000000
    sum_ = sum(((((1 - l_) * weights) ** s + l_) ** (1 / s)) ** 2)
    result = l_ ** 2 / len(weights) * sum_ - 2 * np.log10(1 / delta) / (3 * len(weights))
    return result


def power_law_reweighting(likelihoods, df_train, lambda_=0.9, s=-1, clipper=False, normalize=False,
                          tune_lambda=True, epsilon=1e-16):
    weights = []
    for i in np.unique(df_train["batch"].values):
        local_samples = df_train[df_train.batch == i].loc[:, df_train.columns != "batch"]
        weights = np.append(weights, np.exp(likelihoods[-1].score_samples(local_samples)) /
                            (np.exp(likelihoods[i].score_samples(local_samples))+epsilon))

    if clipper:
        weights = np.clip(weights, 0, clipper)

    if normalize:
        weights = weights / np.linalg.norm(weights, 1) * len(weights)

    if tune_lambda:
        delta = 0.1
        l_partial = partial(optimal_lambda, weights, s, delta)
        try:
            lambda_ = newton_krylov(l_partial, lambda_)
        except:
            pass

    weights = ((1 - lambda_) * weights ** s + lambda_) ** (1 / s)

    return weights


def retraining_all(likelihoods, df_train, **kwargs):
    return np.ones(df_train.shape[0])


# problem in some cases (train=batch + small to be assessed)
def retraining_last_k(k, batch_size, likelihoods, df_train, **kwargs):
    try:
        weights = np.concatenate([np.zeros(df_train.shape[0] - batch_size * k), np.ones(batch_size * k)])
        weights = weights / np.linalg.norm(weights, 1) * len(weights)
    except ValueError:
        weights = np.ones(df_train.shape[0])
    return weights



def rulsif_reweighting(likelihoods, df_train, alpha=0, sigma_range=[1e-1, 1e1], #1e-3, 1e-2,
                       lambda_range=[1e-1, 1e1], normalize=False, clipper=None): #1e-3, 1e-2,
    try:
        local = np.array(df_train[df_train.batch == df_train.batch.values[-1]].loc[:, df_train.columns != "batch"])
        rest = np.array(df_train[df_train.batch != df_train.batch.values[-1]].loc[:, df_train.columns != "batch"])
        densratio_obj = densratio(local, rest, alpha=alpha, sigma_range=sigma_range,
                                  lambda_range=lambda_range, verbose=False)
        weights = densratio_obj.compute_density_ratio(np.array(df_train.loc[:, df_train.columns != "batch"]))

        if clipper:
            weights = np.clip(weights, 0, clipper)

        if normalize:
            weights = weights / np.linalg.norm(weights, 1) * len(weights)
        return weights
    except ValueError:
        print("error in rulsif")
        return np.ones(df_train.shape[0])
