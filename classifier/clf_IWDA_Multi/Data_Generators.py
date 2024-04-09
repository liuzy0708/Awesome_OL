from numpy.random import default_rng
import numpy as np
import emcee
import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
import copy
from scipy.stats import norm, ortho_group
import random
import math
import scipy.stats as ss


"""
A collection of synthetic data generators, including multivariate normal data, data generated with archimedean copulas,
data generated with arbitrary marginals and gaussian copula and data from already existing drift generators.
"""

rng = default_rng()

# three available archimedean copulas
def clayton(theta, n):
    v = random.gammavariate(1/theta, 1)
    uf = [random.expovariate(1)/v for _ in range(n)]
    return [(k+1)**(-1.0/theta) for k in uf]


def amh(theta, n):
    # NOTE: Use SciPy RNG for convenience here
    v = ss.geom(1-theta).rvs()
    uf = [random.expovariate(1)/v for _ in range(n)]
    return [(1-theta)/(math.exp(k)-theta) for k in uf]


def frank(theta, n):
    v = ss.logser(1-math.exp(-theta)).rvs()
    uf = [random.expovariate(1)/v for _ in range(n)]
    return [-math.log(1-(1-math.exp(-theta))*(math.exp(-k))/theta) for k in uf]


def new_distribution_cholesky(pre_mean, ch_mean, perturbation=0.1):
    pre_mean[ch_mean] = pre_mean[ch_mean] + np.random.random(sum(ch_mean)) - perturbation
    cond = 10000
    var = None
    while cond > 1000:
        chol = ortho_group.rvs(len(pre_mean))
        var = chol@chol.T
        cond = np.linalg.cond(var)
    return pre_mean, var


def new_similar_distribution_cholesky(pre_mean, pre_chol, ch_mean, perturbation=0.1):
    """Problematic, as the resulting cov matrix is almost diagonal!"""
    pre_mean[ch_mean] = pre_mean[ch_mean] + np.random.random(sum(ch_mean)) - perturbation  # not to change the mean too much
    cond = 10000
    var = None
    while cond > 1000:
        chol = pre_chol + np.random.uniform(0, perturbation, (len(pre_mean), len(pre_mean)))
        chol = nearest_orthogonal_matrix(chol)
        var = chol@chol.T
        cond = np.linalg.cond(var)
    return pre_mean, var


def new_distribution_svd(pre_mean, ch_mean, perturbation=0.1, conditioning=1000):
    pre_mean[ch_mean] = pre_mean[ch_mean] + np.random.random(sum(ch_mean)) - perturbation
    cond = conditioning*100*len(pre_mean)
    var = None
    while cond > conditioning*10*len(pre_mean) or cond < conditioning*len(pre_mean):
        nums = np.random.uniform(0, 1, len(pre_mean))  # change eigenvalues distribution
        corr = ss.random_correlation.rvs(nums/sum(nums)*len(pre_mean), random_state=rng)
        S = np.diag(np.random.uniform(0, 1, len(pre_mean)))
        var = S.T@corr@S
        cond = np.linalg.cond(var)
    return pre_mean, var


def new_similar_distribution_svd(pre_mean, pre_nums, pre_S, ch_mean, perturbation=0.02):
    pre_mean[ch_mean] = pre_mean[ch_mean] + np.random.random(sum(ch_mean)) - perturbation
    cond = 10000*len(pre_mean)
    var = None
    while cond > 1000*len(pre_mean) or cond < 10*len(pre_mean):
        nums = pre_nums + np.random.uniform(0, perturbation, len(pre_mean))
        corr = ss.random_correlation.rvs(nums/sum(nums)*len(pre_mean), random_state=rng)
        S = pre_S + np.diag(np.random.uniform(0, perturbation/2, len(pre_mean)))
        var = S.T@corr@S
        cond = np.linalg.cond(var)
    return pre_mean, var


def new_distribution(pre_mean, pre_cov, ch_mean, ch_cov, change_X=True, change_y=True):
    # ch_mean and ch_cov are masks with where to change mean and cov (localised drift)
    # important! A complete mask for cov has to be passed, but only the upper triangular part will be considered
    if change_y and change_X:
        pre_mean[ch_mean] = pre_mean[ch_mean] + np.random.random(sum(ch_mean)) - 0.5  # not to change the mean too much
        pre_cov[ch_cov] = np.random.normal(size=sum(sum(ch_cov)))
        pre_cov = np.tril(pre_cov.T) + np.triu(pre_cov, 1)
        if not np.all(np.linalg.eigvals(pre_cov) > 0):
            pre_cov = nearestPD(pre_cov)
    elif change_X:
        pre_mean_old = pre_mean
        pre_mean[ch_mean] = pre_mean[ch_mean] + np.random.random(sum(ch_mean)) - 0.5  # not to change the mean too much
        pre_mean[-1] = pre_mean_old[-1]
        pre_cov_old = pre_cov
        pre_cov[ch_cov] = np.random.normal(size=sum(sum(ch_cov)))
        pre_cov = np.tril(pre_cov.T) + np.triu(pre_cov, 1)
        pre_cov[-1][-1] = pre_cov_old[-1][-1]
        while np.any(np.linalg.eigvals(pre_cov) <= 0):
            pre_cov_ = nearestPD(pre_cov)
            pre_cov_[-1][-1] = pre_cov_old[-1][-1]
            pre_cov = pre_cov_
    else:   # non mi serve ora fare caso solo y cambia
        n_dim = len(pre_cov)
        ch_cov = np.array([[False] * int(n_dim)] * int(n_dim), dtype=bool)
        ch_cov[:, -1] = [True] * (n_dim-1) + [False]
        pre_cov[ch_cov] = np.random.normal(size=sum(sum(ch_cov)))
        pre_cov = np.tril(pre_cov.T) + np.triu(pre_cov, 1)
        pre_cov_old = pre_cov
        while np.any(np.linalg.eigvals(pre_cov) <= 0):
            pre_cov_ = nearestPD(pre_cov)
            # i add a small perturbation to P(X) too, if not I cannot change P(Y|X) without singularity in the cov matrix
            pre_cov_[np.invert(ch_cov)] = pre_cov_old[np.invert(ch_cov)]+np.random.normal(size=sum(sum(np.invert(ch_cov))))/20
            pre_cov_ = np.tril(pre_cov_.T) + np.triu(pre_cov_, 1)
            pre_cov = pre_cov_


    return pre_mean, pre_cov


def new_similar_distribution(pre_mean, pre_cov, ch_mean, ch_cov,  change_X=True, change_y=True):
    # ch_mean and ch_cov are masks with where to change mean and cov (localised drift)
    # important! A complete mask for cov has to be passed, but only the upper triangular part will be considered

    # new similar distribution, as of now, only permits data drift + covariate drift, unlike abrupt where
    # the two can be separated and simulated independently

    if change_y:
        pre_mean[ch_mean] = pre_mean[ch_mean] + rng.uniform(-0.1, 0.1, size=sum(ch_mean))
        pre_cov[ch_cov] = np.reshape(pre_cov[ch_cov], -1) + rng.uniform(-0.1, 0.1, sum(sum(ch_cov)))
        pre_cov = np.tril(pre_cov.T) + np.triu(pre_cov, 1)
        if not np.all(np.linalg.eigvals(pre_cov) > 0):
            pre_cov = nearestPD(pre_cov)
    else:
        pre_mean_old = pre_mean
        pre_mean[ch_mean] = pre_mean[ch_mean] + rng.uniform(-0.1, 0.1, size=sum(ch_mean))
        pre_mean[-1] = pre_mean_old[-1]
        pre_cov_old = pre_cov
        pre_cov[ch_cov] = np.reshape(pre_cov[ch_cov], -1) + rng.uniform(-0.1, 0.1, sum(sum(ch_cov)))
        pre_cov = np.tril(pre_cov.T) + np.triu(pre_cov, 1)
        pre_cov[-1][-1] = pre_cov_old[-1][-1]
        while np.any(np.linalg.eigvals(pre_cov) <= 0):
            pre_cov_ = nearestPD(pre_cov)
            pre_cov_[-1][-1] = pre_cov_old[-1][-1]
            pre_cov = pre_cov_


    return pre_mean, pre_cov


def new_distribution_deprecated(pre_mean, pre_cov, ch_mean, ch_cov):
    # ch_mean and ch_cov are masks with where to change mean and cov (localised drift)
    pre_mean[ch_mean] = pre_mean[ch_mean] + np.random.random(sum(ch_mean)) - 0.5
    pre_cov[ch_cov] = np.random.random((sum(ch_cov),len(pre_mean)))
    pre_cov = nearestPD(pre_cov)
    return pre_mean, pre_cov


def lnprob_trunc_norm(x, mean, n_dim, C):
    if sum(x) > 0 *n_dim:
        return -np.inf
    else:
        return -0.5 *( x -mean).dot(np.linalg.inv(C)).dot( x -mean)


def truncated_normal_sampling(pre_mean, pre_cov, size, n_dim):
    if size <= 0:
        return None
    if size >= n_dim*2:
        pos = emcee.utils.sample_ball(pre_mean, np.sqrt(np.diag(pre_cov)), size=size)
    else:
        pos = rng.multivariate_normal(pre_mean, pre_cov, size=size)
    S = emcee.EnsembleSampler(size, n_dim, lnprob_trunc_norm, args=(pre_mean, n_dim, pre_cov))
    pos, prob, state = S.run_mcmc(pos, 100)
    # print(np.max(pos))
    return pos


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def nearest_orthogonal_matrix(A):
        '''
        Find closest orthogonal matrix to *A* using iterative method.

        Bases on the code from REMOVE_SOURCE_LEAKAGE function from OSL Matlab package.
        Args:
            A (numpy.array): array shaped k, n, where k is number of channels, n - data points

        Returns:
            L (numpy.array): orthogonalized matrix with amplitudes preserved
        Reading:
            Colclough GL et al., A symmetric multivariate leakage correction for MEG connectomes.,
                        Neuroimage. 2015 Aug 15;117:439-48. doi: 10.1016/j.neuroimage.2015.03.071

        '''
        #
        MAX_ITER = 2000

        TOLERANCE = np.max((1, np.max(A.shape) * np.linalg.svd(A.T, False, False)[0])) * np.finfo(A.dtype).eps  # TODO
        reldiff = lambda a, b: 2 * abs(a - b) / (abs(a) + abs(b))
        convergence = lambda rho, prev_rho: reldiff(rho, prev_rho) <= TOLERANCE

        A_b = A.conj()
        d = np.sqrt(np.sum(A * A_b, axis=1))

        rhos = np.zeros(MAX_ITER)

        for i in range(MAX_ITER):
            scA = A.T * d

            u, s, vh = np.linalg.svd(scA, False)

            V = np.dot(u, vh)

            # TODO check is rank is full
            d = np.sum(A_b * V.T, axis=1)

            L = (V * d).T
            E = A - L
            rhos[i] = np.sqrt(np.sum(E * E.conj()))
            if i > 0 and convergence(rhos[i], rhos[i - 1]):
                break
        return L


def generate_normal_drift_data(batch_size, train_size, length, pre_mean_, pre_cov_, ch_mean, ch_cov,
                               change, n_dim, scale=False, gradual_drift=False, oracle=False, change_X=True,
                               change_y=True, verbose=False):
    """Generates multivariate normal drifting data"""
    if scale:
        scaler = StandardScaler()
    pre_mean = pre_mean_.copy()
    pre_cov = pre_cov_.copy()
    df = pd.DataFrame()
    means = []
    covs = []
    if verbose:
        disable = False
    else:
        disable = True

    for i in tqdm(range(length), disable=disable):
        if i % change == 0:
            pre_mean, pre_cov = new_distribution(pre_mean, pre_cov, ch_mean, ch_cov,
                                                 change_X=change_X, change_y=change_y)
        if gradual_drift:
            pre_mean, pre_cov = new_similar_distribution(np.zeros(n_dim), pre_cov, [False] * n_dim, ch_cov,
                                                         change_X=change_X, change_y=change_y)
        if i == 0:
            data = rng.multivariate_normal(pre_mean, pre_cov, size=train_size)
        else:
            data = rng.multivariate_normal(pre_mean, pre_cov, size=batch_size)
        prov = pd.DataFrame(data)
        if i == 0 and scale:
            scaled_features = scaler.fit_transform(prov.values)
            prov = pd.DataFrame(scaled_features, index=prov.index, columns=prov.columns)
        elif i != 0 and scale:
            scaled_features = scaler.transform(prov.values)
            prov = pd.DataFrame(scaled_features, index=prov.index, columns=prov.columns)
        prov["batch"] = i
        df = df.append(prov, ignore_index=True)
        means.append(list(pre_mean))
        covs.append(copy.deepcopy(pre_cov))
    df.rename(columns={n_dim - 1: 'label'}, inplace=True)
    if oracle:
        return df, means, covs
    else:
        return df


def generate_normal_drift_data_cholesky(batch_size, train_size, length, pre_mean_, pre_chol_, ch_mean,
                               change, n_dim, scale=False, gradual_drift=False, oracle=False, verbose=False):
    """Generates multivariate normal drifting data -> no correlation! Do not use!!!"""
    if scale:
        scaler = StandardScaler()
    pre_mean = pre_mean_.copy()
    pre_chol = pre_chol_.copy()
    df = pd.DataFrame()
    means = []
    covs = []
    if verbose:
        disable = False
    else:
        disable = True

    for i in tqdm(range(length), disable=disable):
        if i % change == 0:
            pre_mean, cov = new_distribution_cholesky(pre_mean, ch_mean)
        if gradual_drift:
            pre_mean, cov = new_similar_distribution_cholesky(pre_mean, pre_chol, ch_mean)
        if i == 0:
            data = rng.multivariate_normal(pre_mean, cov, size=train_size)
        else:
            data = rng.multivariate_normal(pre_mean, cov, size=batch_size)
        prov = pd.DataFrame(data)
        if i == 0 and scale:
            scaled_features = scaler.fit_transform(prov.values)
            prov = pd.DataFrame(scaled_features, index=prov.index, columns=prov.columns)
        elif i != 0 and scale:
            scaled_features = scaler.transform(prov.values)
            prov = pd.DataFrame(scaled_features, index=prov.index, columns=prov.columns)
        prov["batch"] = i
        df = df.append(prov, ignore_index=True)
        means.append(list(pre_mean))
        covs.append(copy.deepcopy(cov))
    df.rename(columns={n_dim - 1: 'label'}, inplace=True)
    if oracle:
        return df, means, covs
    else:
        return df



def generate_normal_drift_data_svd(batch_size, train_size, length, pre_mean_, pre_eigs_, pre_S_, ch_mean,
                               change, n_dim, scale=False, gradual_drift=False, oracle=False, verbose=False):
    """Generates multivariate normal drifting data"""
    if scale:
        scaler = StandardScaler()
    pre_mean = pre_mean_.copy()
    pre_eigs = pre_eigs_.copy()
    pre_S = pre_S_.copy()
    df = pd.DataFrame()
    means = []
    covs = []

    if verbose:
        disable = False
    else:
        disable = True

    for i in tqdm(range(length), disable=disable):
        if i % change == 0:
            pre_mean, cov = new_distribution_svd(pre_mean, ch_mean)
        if gradual_drift:
            pre_mean, cov = new_similar_distribution_svd(pre_mean, pre_eigs, pre_S, ch_mean)
        if i == 0:
            data = rng.multivariate_normal(pre_mean, cov, size=train_size)
        else:
            data = rng.multivariate_normal(pre_mean, cov, size=batch_size)
        prov = pd.DataFrame(data)
        if i == 0 and scale:
            scaled_features = scaler.fit_transform(prov.values)
            prov = pd.DataFrame(scaled_features, index=prov.index, columns=prov.columns)
        elif i != 0 and scale:
            scaled_features = scaler.transform(prov.values)
            prov = pd.DataFrame(scaled_features, index=prov.index, columns=prov.columns)
        prov["batch"] = i
        df = df.append(prov, ignore_index=True)
        means.append(list(pre_mean))
        covs.append(copy.deepcopy(cov))
    df.rename(columns={n_dim - 1: 'label'}, inplace=True)
    if oracle:
        return df, means, covs
    else:
        return df


def generate_normal_localised_drift_data(batch_size, train_size, length, pre_mean, pre_cov, ch_mean, ch_cov,
                                         change, n_dim, scale=False, oracle=False, verbose=False):
    """Generates multivariate normal drifting data with drift localised in space with truncated normal sampling
    with shifting covariance in the desired part of the space"""
    if scale:
        scaler = StandardScaler()
    df = pd.DataFrame()
    means = []
    covs = []
    pre_mean_2 = pre_mean.copy()
    pre_cov_2 = pre_cov.copy()
    if verbose:
        disable = False
    else:
        disable = True

    for i in tqdm(range(length), disable=disable):
        if i == 0:
            data = np.random.multivariate_normal(pre_mean, pre_cov, size=train_size)
        else:
            data = np.random.multivariate_normal(pre_mean, pre_cov, size=batch_size)
        # se in una zona del piano -> change distribution
        data = data[data.sum(axis=1) < 0]
        if i == 0:
            data2 = truncated_normal_sampling(pre_mean_2, pre_cov_2, train_size - len(data), n_dim)
        else:
            data2 = truncated_normal_sampling(pre_mean_2, pre_cov_2, batch_size - len(data), n_dim)
        data2 = data2.clip(-4, 4)  # there are some problems in the sampling from the truncated normal
        data = np.concatenate((data, data2))
        prov = pd.DataFrame(data)
        if i == 0 and scale:
            scaled_features = scaler.fit_transform(prov.values)
            prov = pd.DataFrame(scaled_features, index=prov.index, columns=prov.columns)
        elif i != 0 and scale:
            scaled_features = scaler.transform(prov.values)
            prov = pd.DataFrame(scaled_features, index=prov.index, columns=prov.columns)
        prov["batch"] = i
        df = df.append(prov, ignore_index=True)
        means.append(list(pre_mean_2))
        covs.append(pre_cov_2)
        if i % change == 0:
            pre_mean_2, pre_cov_2 = new_distribution(pre_mean_2, pre_cov_2, ch_mean, ch_cov)

    df.rename(columns={n_dim - 1: 'label'}, inplace=True)
    if oracle:
        return df, means, covs
    else:
        return df


def generate_gaussian_copula_drift_data(batch_size, train_size, length, marginals, pre_cov_, ch_cov,
                                        change, n_dim, scale=False, gradual_drift=False, oracle=False, verbose=False):
    """Generate data with the desired marginal distributions and a gaussian copula with drifting cov. matrix"""
    if scale:
        scaler = StandardScaler()
    pre_cov = pre_cov_.copy()
    df = pd.DataFrame()
    covs = []
    if verbose:
        disable = False
    else:
        disable = True

    for i in tqdm(range(length), disable=disable):
        if i % change == 0:
            _, pre_cov = new_distribution(np.zeros(n_dim), pre_cov, [False]*n_dim, ch_cov)
        if gradual_drift:
            _, pre_cov = new_similar_distribution(np.zeros(n_dim), pre_cov, [False] * n_dim, ch_cov)
        if i == 0:
            data = rng.multivariate_normal(np.zeros(n_dim), pre_cov, size=train_size)
        else:
            data = rng.multivariate_normal(np.zeros(n_dim), pre_cov, size=batch_size)
        prov_pre = pd.DataFrame(data)
        prov = pd.DataFrame()
        for j in range(n_dim):
            prov = prov.append(pd.Series(marginals[j].ppf(norm.cdf(prov_pre.iloc[:, j]))), ignore_index=True)
        prov = prov.T
        if i == 0 and scale:
            scaled_features = scaler.fit_transform(prov.values)
            prov = pd.DataFrame(scaled_features, index=prov.index, columns=prov.columns)
        elif i != 0 and scale:
            scaled_features = scaler.transform(prov.values)
            prov = pd.DataFrame(scaled_features, index=prov.index, columns=prov.columns)
        prov["batch"] = i
        df = df.append(prov, ignore_index=True)
        covs.append(copy.deepcopy(pre_cov))
    df.rename(columns={n_dim - 1: 'label'}, inplace=True)
    if oracle:
        return df, covs
    else:
        return df


def generate_archimedean_copula_drift_data(batch_size, train_size, length, change, n_dim, gradual_drift=False, scale=True,
                                           verbose=False):
    """generator which continuously switches the dependence structure from a clayton to a frank copula, while
    also changing the parameter theta which governs the dependence structure."""
    if scale:
        scaler = StandardScaler()
    df = pd.DataFrame()
    # here, I am changing the covariance structure of the copula
    if verbose:
        disable = False
    else:
        disable = True

    for i in tqdm(range(length), disable=disable):
        if i % change == 0:
            theta = rng.uniform(0.2, 10)
            distribution = random.choice([clayton, frank])
        if gradual_drift:
            theta = max(theta + rng.uniform(-1, 1), 0.01)
        if i == 0:
            data = [distribution(theta, n_dim) for _ in range(train_size)]
        else:
            data = [distribution(theta, n_dim) for _ in range(batch_size)]

        prov= pd.DataFrame(data)

        if i == 0 and scale:
            scaled_features = scaler.fit_transform(prov.values)
            prov = pd.DataFrame(scaled_features, index=prov.index, columns=prov.columns)
        elif i != 0 and scale:
            scaled_features = scaler.transform(prov.values)
            prov = pd.DataFrame(scaled_features, index=prov.index, columns=prov.columns)
        prov["batch"] = i
        df = df.append(prov, ignore_index=True)
    df.rename(columns={n_dim - 1: 'label'}, inplace=True)
    return df


def generate_skmultiflow_data(batch_size, train_size, length, change, generator, scale=True, verbose=False):
    """Supports AGRAWAL, SEA, Sine, STAGGER, Mixed. It gets the generator in input and produces a dataframe
     which is compatible with the environment for fast batch-based experimentation"""
    if scale:
        scaler = StandardScaler()
    df = pd.DataFrame()

    # here, the data is generated in batches
    if verbose:
        disable = False
    else:
        disable = True

    for i in tqdm(range(length), disable=disable):
        if i % change == 0:
            generator.generate_drift()
        if i == 0:
            data_loc = generator.next_sample(train_size)
        else:
            data_loc = generator.next_sample(batch_size)

        data = pd.DataFrame(data_loc[0])
        n_dim = data.shape[1]+1
        if i == 0 and scale:
            scaled_features = scaler.fit_transform(data.values)
            data = pd.DataFrame(scaled_features, index=data.index, columns=data.columns)
        elif i != 0 and scale:
            scaled_features = scaler.transform(data.values)
            data = pd.DataFrame(scaled_features, index=data.index, columns=data.columns)
        data["label"] = pd.DataFrame(data_loc[1])
        data["batch"] = i
        df = df.append(data, ignore_index=True)
        df.rename(columns={n_dim - 1: 'label'}, inplace=True)
    return df


# for streaming only?
def generate_river_data(batch_size, train_size, length_data, change, generator, scale=True, verbose=False):
    if scale:
        scaler = StandardScaler()
    df = pd.DataFrame()

    data = pd.DataFrame([i for i, _ in generator.take(length_data)])
    data["label"] = pd.Series([j for _, j in generator.take(length_data)])

    if scale:
        scaler.fit(data[:100].values)
        scaled_features = scaler.transform(data.values)
        data = pd.DataFrame(scaled_features, index=data.index, columns=data.columns)
        data["batch"] = [0]*train_size + [(i//batch_size) for i in range(train_size, length_data)]
    return data


def shift_dataset(data):
    df = data.copy()
    temp_old = df.iloc[:, 0]
    for idx in range(df.shape[1]):
        index = (idx + 1) % df.shape[1]
        temp = df.iloc[:, index]
        df.iloc[:, index] = temp_old
        temp_old = temp
    df["label"] = data["label"]
    return df


def drift_dataset(df):
    #df.loc[:, df.columns != "label"].iloc[:, np.random.permutation(df.shape[1]-1)] = \
    #    df.loc[:, df.columns != "label"].iloc[:, np.random.permutation(df.shape[1]-1)]

    da = df.loc[:, df.columns != "label"].iloc[:, np.random.permutation(df.shape[1] - 1)]

    da.rename(columns={x: y for x, y in zip(da.columns, df.columns)}, inplace=True)

    return da


def swap_two_features(df):
    a, b = 10, 10
    while a == b:
        a, b = np.random.randint(0, df.shape[1]-1, 2)
    col_list = list(df)
    col_list[a], col_list[b] = col_list[b], col_list[a]
    df.columns = col_list
    return df


def generate_drift_ucl_data(batch_size, train_size, length, change, data, gradual_drift=True, verbose=False):
# pass data already scaled or create helper function to prepare dataframe
    if verbose:
        disable = False
    else:
        disable = True

    data = data[:train_size + length*batch_size]
    df = pd.DataFrame()
    for i in tqdm(range(length), disable=disable):
        if not gradual_drift:
            if i % change == 0:
                if i == 0:
                    data_loc = data[:train_size]
                else:
                    data = drift_dataset(data.copy())
                    data_loc = data[train_size+i*batch_size:train_size+(i+1)*batch_size]
            else:
                data_loc = data[train_size + i * batch_size:train_size + (i+1) * batch_size]

        else:
            if i == 0:
                data_loc = data[:train_size]
            else:
                data = swap_two_features(data.copy())
                data_loc = data[train_size + i * batch_size: train_size + (i + 1) * batch_size]

            #  possibilità: o si inserisce mistura dei due casi o si simula drift per mezzo di cambiamento in P(X)

        data_loc["batch"] = i
        df = df.append(data_loc, ignore_index=True)

    return df
