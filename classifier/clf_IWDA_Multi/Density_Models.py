import scipy.stats as ss
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.covariance import EmpiricalCovariance, OAS
from sklearn.neighbors import KernelDensity
from lightgbm import LGBMRegressor
import torch
from torch import nn
from torch import optim, relu

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

# available density models: MultivariateNormal, KDE, GMM, Bayesian GMM, MAR Normalising Flows


"""
Any Density Model with a fit and score_samples methods can be used for modeling the density for IW.
"""


class KernelDensity2(KernelDensity):

    def __init__(self):
        super(KernelDensity2, self).__init__()

    def score(self, X):
        return self.score_samples(X).mean()


class MultivariateNormal(OAS):

    def __init__(self):
        super(MultivariateNormal, self).__init__()

    def score_samples(self, X):
        if hasattr(self, "location_"):
            mu = self.location_
            sigma = self.covariance_
            return np.array(ss.multivariate_normal.logpdf(X, mu, sigma))
        else:
            return np.ones(X.shape[0])


class MARFlow(BaseEstimator):
    """Masked autoregressive normalising flow"""

    def __init__(self, n_dim, num_layers=4, base_dist=StandardNormal, hidden_features=4, transform=None, num_iter=2000,
                 optimizer=optim.Adam, batch_size=64, verbose=False, train_percent=0.9):
        super(MARFlow, self).__init__()
        self.n_dim = n_dim
        self.num_layers = num_layers
        self.base_dist = base_dist(shape=[n_dim])
        if transform is None:
            transforms = []
            for _ in range(num_layers):
                transforms.append(ReversePermutation(features=n_dim))
                transforms.append(MaskedAffineAutoregressiveTransform(features=n_dim, hidden_features=hidden_features))
            self.transform = CompositeTransform(transforms)

        else:
            self.transform = transform
        self.flow = Flow(self.transform, self.base_dist)
        self.optimizer = optimizer(self.flow.parameters())
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self.train_percent = train_percent
# learning rate?
    def reset(self, num_layers=4, base_dist=StandardNormal, hidden_features=4,
              transform=None, num_iter=2000, optimizer=optim.Adam, verbose=False, train_percent=0.9):
        self.__init__(n_dim=self.n_dim, num_layers=num_layers, base_dist=base_dist, hidden_features=hidden_features,
                      transform=transform, num_iter=num_iter, optimizer=optimizer,
                      verbose=verbose, train_percent=train_percent)

    def fit(self, X):
        X = np.array(X, dtype=np.float32)
        X = torch.from_numpy(X)
        permutation = torch.randperm(X.size()[0])
        indices = permutation[:int(self.train_percent*X.size()[0])]
        X_train = X[indices]
        X_val = X[-indices]
        min_val_loss = 100000
        min_val_epoch = 0
        delta_stop = self.num_iter/20

        for i in range(self.num_iter):
            permutation = torch.randperm(X_train.size()[0])
            indices = permutation[:self.batch_size]
            batch_x = X_train[indices]
            self.optimizer.zero_grad()
            loss = -self.flow.log_prob(inputs=batch_x).mean()
            loss.backward()
            self.optimizer.step()

            if i % int(self.num_iter/50) == 0:
                val_loss = -self.flow.log_prob(inputs=X_val).mean()
                if self.verbose:
                    print(f"{i}, train_loss: {loss}, val_loss: {val_loss}")

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    min_val_epoch = i

                elif i - min_val_epoch > delta_stop:  # no decrease in min_val_loss for "delta_stop epochs"
                    break

        return self

    def score_samples(self, X):
        return self.flow.log_prob(torch.tensor(np.array(X), dtype=torch.float32)).clone().detach().numpy()

    def score(self, X):
        return self.score_samples(X).mean()

    def sample(self, n):
        return self.flow.sample(n).clone().detach().numpy()


class ClairvoyantNormal(BaseEstimator):
    """only for testing once, this class then destroys its internal information"""

    def __init__(self, means, covs, batch=0):
        super(ClairvoyantNormal, self).__init__()
        self.means = means
        self.covs = covs
        self.batch = batch


    def fit(self, X):
        return self

    def score_samples(self, X):
        mean = self.means[self.batch]
        var = self.covs[self.batch]
        return np.array(ss.multivariate_normal.logpdf(X, mean, var, allow_singular=True))

    def score(self, X):
        return self.score_samples(X).mean()


def weighted_mse_loss(inputs, target, sample_weight):
    if sample_weight is not None:
        return (sample_weight * (inputs - target) ** 2).mean()
    else:
        return ((inputs - target) ** 2).mean()


# regressor, not density model
class RegressorNet(nn.Module):
    def __init__(self, n_dim, num_iter=100, optimizer=optim.Adam):
        super(RegressorNet, self).__init__()
        self.hid1 = nn.Linear(n_dim, 64)  # 8-(10-10)-1
        self.drop1 = nn.Dropout(0.2)
        self.hid2 = nn.Linear(64, 32)
        self.drop2 = nn.Dropout(0.2)
        self.oupt = nn.Linear(32, 1)

        self.num_iter = num_iter
        self.optimizer = optimizer(self.parameters())

    def forward(self, x):
        z = relu(self.hid1(x))
        z = self.drop1(z)
        z = relu(self.hid2(z))
        z = self.drop2(z)
        z = self.oupt(z)  # no activation
        return z

    def fit(self, X, y, sample_weight=None):

        nn.init.xavier_uniform_(self.hid1.weight)
        nn.init.zeros_(self.hid1.bias)
        nn.init.xavier_uniform_(self.hid2.weight)
        nn.init.zeros_(self.hid2.bias)
        nn.init.xavier_uniform_(self.oupt.weight)
        nn.init.zeros_(self.oupt.bias)

        X = np.array(X, dtype=np.float32)
        X = torch.from_numpy(X)
        y = np.array(y, dtype=np.float32)
        y = torch.from_numpy(y)
        if sample_weight is not None:
            weights = np.array(sample_weight, dtype=np.float32)
            weights = torch.from_numpy(weights)
        else:
            weights = None
        for _ in range(self.num_iter):
            self.optimizer.zero_grad()
            output = self.forward(X)
            loss = weighted_mse_loss(inputs=output, target=y, sample_weight=weights)
            loss.backward()
            self.optimizer.step()
        return self

    def predict(self, X):
        X = np.array(X, dtype=np.float32)
        X = torch.from_numpy(X)
        return self.forward(X).detach().numpy()
