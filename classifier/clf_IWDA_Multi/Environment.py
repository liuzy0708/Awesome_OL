from sklearn.model_selection import GridSearchCV, KFold
import optuna
from sklearn.metrics import r2_score, mean_squared_error
from .Data_Generators import *
from river.drift import *
from .Density_Models import *
from .Reweighting_Models import *
from sklearn.naive_bayes import GaussianNB
import time
from river import metrics

# Mettere attributes
# commentare funzioni

class WeightModel:
    """ General WeightModel class.
        The WeightModel class wraps a density estimator (likelihood model) and a reweighting function, in order to provide
        importance weights. For this, it keeps a list of fit density models to use them for reweighting.
        It also works with schemes which do not require a density model (pass None as argument).
        It also provides limited support for hyperparameter tuning for the density models, with the possibility of
        performing grid search (for sklearn-like model) or bayesian optimization.

        Parameters
        ----------
        likelihood_model : density model instance, default=GaussianMixture
        This parameter specifies the likelihood model to use for reweighting. It needs to provide a fit and
        score_samples (returning the log-likelihoods of the observations) methods, coherent with sklearn notation.
        Apart from sklearn models (GaussianMixture, KDE...) the density models script contains MultivariateNormal and
        Masked Autoregressive Normalising Flows.

        diagnostic : bool, default=True
        If in diagnostic mode, weights and average log-likelihoods (of the newly fitted models on the new data) in time
        are saved inside the model for later inspection.

        reweighting_function: a reweighting function, default = reweighting
        This parameter specifies the reweighting function to use. A reweighting function takes in input a likelihoods
        argument and the training data + optional parameters which have to be set before passing the function to the
        WeightModel class, for instance, with functools.partial

        cv: bool, default = False
        Number of cross-validation folds to tune the density model. Typically is used in conjunction with a
        parameters kwarg dictionary with the parameters to be tuned.


        Attributes
        ----------
        likelihood_model: the density model used for reweighting.
        likelihoods: the fit density models.
        reweighting_function: the reweighting function used for reweighting.
        cv: # of folds for tuning.

    """

    def __init__(self, likelihood_model=GaussianMixture(n_components=2, covariance_type='full'), diagnostic=True,
                 reweighting_function=reweighting, cv=False, **kwargs):
        self.likelihood_model = likelihood_model
        self.likelihoods = []
        self.reweighting_function = reweighting_function
        if diagnostic:
            self.weights = []
            self.loglik = []  # this is the loglikelihood of the new (used for fit) data under the likelihood model
        self.cv = cv
        if "params" in kwargs:
            self.cv_params = kwargs["params"]

    def initial_fit(self, data):
        """ Initial fit of the weight model.
        Parameters
        ----------
        data: array-like of shape (n_samples, n_features+n_targets+1)
        Feature vectors of the training data.
        """
        if self.likelihood_model is not None:
            local_likelihood = copy.deepcopy(self.likelihood_model)
            self.likelihoods.append(local_likelihood.fit(data.loc[:, data.columns != "batch"]))

    def get_weights(self, data, batch):
        """ Get importance weights for retraining.

        This function updates the internal state of the weight model, creating the newly fitted likelihood model
        and, then, using it to perform reweighting.
        Parameters
        ----------
        data: array-like of shape (n_samples, n_features+n_targets)
        Feature vectors of all the data. The last feature is the batch, which indicates which fit density model
        to use for reweighting that particular chunk of data.

        batch: int
        Batch identifying number for the last data.
        """
        if self.likelihood_model is not None:
            local_likelihood = copy.deepcopy(self.likelihood_model)
            if isinstance(self.likelihood_model, ClairvoyantNormal):
                local_likelihood.batch = batch
            if self.cv:
                estimator = self.tuning_cv_density(lk_model=local_likelihood, data=data, batch=batch,
                                                   verbose=0, n_jobs=-1, refit=True)
                self.likelihoods.append(estimator)
            else:
                self.likelihoods.append(local_likelihood.fit(
                    data[data.batch == batch].loc[:, data.columns != "batch"]))
        weights_local = self.reweighting_function(self.likelihoods, data)
        if hasattr(self, "weights") and self.likelihoods:
            self.weights.append(weights_local)
            self.loglik.append(self.likelihoods[-1].score(data[data.batch == batch].loc[:, data.columns != "batch"]))
        elif self.reweighting_function == adversarial_reweighting or self.reweighting_function == rulsif_reweighting:
            self.weights.append(weights_local)
        return weights_local

    def tuning_cv_density(self, lk_model, data, batch, verbose=0, n_jobs=-1, refit=True):
        """ Tune density in CV

        For now, supports sklearn-style models and MARFlow
        """
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if "sklearn" in str(type(lk_model)) or isinstance(self.likelihood_model, KernelDensity2):
            gs_cv = GridSearchCV(lk_model, self.cv_params, cv=self.cv, verbose=verbose, n_jobs=n_jobs, refit=refit)
            gs_cv.fit((data[data.batch == batch].loc[:, data.columns != "batch"]))
            return gs_cv.best_estimator_
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="maximize")
            if "batch" in data.columns:
                self.local_data = data[data.batch == batch].loc[:, data.columns != "batch"]
            else:
                self.local_data = data
            study.optimize(self.objective, n_trials=25)
            lk_model.reset(**study.best_trial.params)
            return lk_model.fit(self.local_data)

    def objective(self, trial):
        if isinstance(self.likelihood_model, MARFlow):
            data = self.local_data
            num_layers = trial.suggest_int("num_layers", min(self.cv_params["num_layers"]),
                                           max(self.cv_params["num_layers"]))
            hidden_features = trial.suggest_int("hidden_features", min(self.cv_params["hidden_features"]),
                                                max(self.cv_params["hidden_features"]))
            num_iter = trial.suggest_int("num_iter", min(self.cv_params["num_iter"]),
                                         max(self.cv_params["num_iter"]))
            local = copy.deepcopy(self.likelihood_model)
            local.reset(num_layers=num_layers, hidden_features=hidden_features, num_iter=num_iter)
            cv = KFold(self.cv, shuffle=True, random_state=42)
            scores = []
            for train_index, test_index in cv.split(data):
                X_train, X_test = data.iloc[train_index], data.iloc[test_index]
                local.fit(X_train)
                scores.append(local.score(X_test))
            return np.mean(scores)

        else:
            raise NotImplementedError


class Model:
    """ Batch Model class.
       The Model class is a general configuration for the framework in a batch-based context, both for regression and
       classification. It stores the underlying learner and a WeightModel instance and uses them for reweighting and,
        then, model retraining with IWERM.
        Attention: the datasets used for fit need to have a batch column defining the batches.

        Parameters
        ----------
        weight_model : a WeightModel instance
        A WeightModel instance, which stores reweighing schemes and density model.

        ml_model : sklearn-like estimator instance
        A machine learning model which complies with Scikit-learn's conventions.

        name : str, default="None"
        The name identifier for the model.

        Attributes
        ----------
        weight_model: density model + reweighting scheme used for reweighting.
        ml_model: underlying learner.

       """

    def __init__(self, weight_model, ml_model, name="None"):
        self.name = name
        self.weight_model = weight_model
        self.ml_model = ml_model
        self.elapsed_time = 0

    def initial_fit(self, X, y):
        df = pd.concat([X, y], axis=1)
        self.weight_model.initial_fit(df)
        self.ml_model.fit(X.loc[:, X.columns != "batch"], y)

    # i have to pass the batch number for the reweighting -> see if needs to be changed
    def fit(self, X, y, batch):
        start = time.time()
        weights = self.weight_model.get_weights(pd.concat([X, y], axis=1), batch)
        #print(np.mean(weights), np.max(weights), self.name)
        try:
            self.ml_model.fit(X.loc[:, X.columns != "batch"], y, sample_weight=weights)
        except TypeError:
            self.ml_model.fit(X.loc[:, X.columns != "batch"], y)
            print("Error in the structure of the model", self.name)
        except ValueError:
            self.ml_model.fit(X.loc[:, X.columns != "batch"], y)
            print("Error in the weights", self.name, X.index)
        self.elapsed_time += (time.time()-start)

    def predict_proba(self, X):
        return self.ml_model.predict_proba(X.loc[:, X.columns != "batch"].values)

    def predict(self, X):
        return self.ml_model.predict(X.loc[:, X.columns != "batch"].values)


class SamplingModel(Model):
    """ Batch SamplingModel class.
        The SamplingModel class is a general configuration for the framework in a batch-based context, both for
        regression and classification. It stores the underlying learner and a WeightModel instance and uses
        them for reweighting and, then, model retraining by using a bag of models retrained on n replicas of the
        training extracted with importance sampling.
        Attention: the datasets used for fit need to have a batch column defining the batches.

        Parameters
        ----------
        weight_model : a WeightModel instance
        A WeightModel instance, which stores reweighing schemes and density model.

        ml_model : sklearn-like estimator instance
        A machine learning model which complies with Scikit-learn's conventions.

        name : str, default="None"
        The name identifier for the model.

        num_models : int, default=10
        The number of dataset replicas to sample and, thus, the number of models in the bag.

        size : int, default=200
        The dimension of the dataset replicas to sample.

       """
    def __init__(self, weight_model, ml_model, name="None", num_models=10, size=200):
        super().__init__(weight_model, ml_model, name)
        self.models = [copy.copy(ml_model) for _ in range(num_models)]
        self.size = size

    def fit(self, X, y, batch):
        start = time.time()
        weights = self.weight_model.get_weights(pd.concat([X, y], axis=1), batch)
        try:
            for model in self.models:
                idxs = np.random.choice(len(weights), self.size, p=weights / sum(weights))
                model.fit(X.loc[:, X.columns != "batch"].iloc[idxs], y.iloc[idxs])
        except TypeError:
            for model in self.models:
                idxs = np.random.choice(len(weights), self.size, p=np.ones(len(weights))/len(weights))
                model.fit(X.loc[:, X.columns != "batch"].iloc[idxs], y.iloc[idxs])
            print("Error in the structure of the model", self.name)
        except ValueError:
            for model in self.models:
                idxs = np.random.choice(len(weights), self.size, p=np.ones(len(weights))/len(weights))
                model.fit(X.loc[:, X.columns != "batch"].iloc[idxs], y.iloc[idxs])
            print("Error in the weights", self.name)

        self.elapsed_time += (time.time()-start)


    def predict_proba(self, X):
        return np.mean([model.predict_proba(X.loc[:, X.columns != "batch"]) for model in self.models], axis=0)

    def predict(self, X):
        return np.mean([model.predict(X.loc[:, X.columns != "batch"]) for model in self.models], axis=0)


class DensityDriftModel(Model):
    """ Online Model class with density drift detection.
       The Model class is a general configuration for the framework in a online-based context, both for regression and
       classification. It stores the underlying learner, WeightModel instance and a drift detector using the
        likelihood of the WeightModel for drift detection. At every instance,  the new output is predicted and
        the drift detector is updated. Based on its state, retraining can be performed.

        Parameters
        ----------
        weight_model : a WeightModel instance
        A WeightModel instance, which stores reweighing schemes and density model.

        train_size : int
        Up to which index of the data, it's pre-training data.

        ml_model : sklearn-like estimator instance
        A machine learning model which complies with Scikit-learn's conventions.

        drift_detector : River Drift Detector instance, default = PageHinkley()
        The chosen drift detector. Only ADWIN and PageHinkley are supported.

        name : str, default="None"
        The name identifier for the model.

        old_to_use : int, default=100
        Old to use hyperparameter. It regulates how many of the last samples are used for fitting the new density model.

        update_wm : int, default=200
        Update hyperparameter. It is the number of instances it waits before refitting the density model and, thus,
        creating a new batch, even if drift is not detected.

    """

    def __init__(self, weight_model, train_size, ml_model,
                 drift_detector=PageHinkley(), name="None", old_to_use=100,
                 update_wm=200, verbose=True):
        super().__init__(weight_model, ml_model, name)
        self.drift_detector = drift_detector
        self.batch = 0
        self.old_to_use = old_to_use
        self.last = train_size
        self.update_wm = update_wm
        self.batch_data = []
        self.detected_drifts = []
        self.verbose = verbose

    def initial_fit(self, X, y):
        start = time.time()
        df = pd.concat([X, y], axis=1)
        self.weight_model.initial_fit(df)
        self.ml_model.fit(X.loc[:, X.columns != "batch"].values, y.values)
        scores = self.weight_model.likelihoods[0].score_samples(df.loc[:, df.columns != "batch"])
        self.mean = np.mean(scores)  # This might be changed with a CV estimate
        self.std = np.std(scores)    # This might be changed with a CV estimate
        self.batch_data = [0] * X.shape[0]  # we have to create internally batches and store them
        self.elapsed_time += (time.time()-start)

    def partial_fit(self, X, y):
        start = time.time()
        self.batch_data.append(self.batch)
        df_last = pd.concat([X[-1:], y[-1:]], axis=1)
        # here we are computing the samples score after whitening with the mean and std of the training set scores
        sample_score = (self.weight_model.likelihoods[self.batch].score(df_last.loc[:, df_last.columns != "batch"]) -
                        self.mean) / self.std
        # print("sample_score", sample_score)
        in_drift, in_warning = self.drift_detector.update(sample_score)
        # print(in_drift, in_warning)
        if not in_warning and not in_drift and (X[-1:].index[0] - self.last) > self.update_wm:
            _, df = self.update_batch(X, y, self.update_wm)
            self.last = X[-1:].index[0]
            scores = self.weight_model.likelihoods[self.batch].score_samples(df[-self.update_wm:]
                                                                             .loc[:, df.columns != "batch"])
            self.mean = np.mean(scores)
            self.std = np.std(scores)  # we update our density models in time
        if in_drift:
            weights, df = self.update_batch(X, y, self.old_to_use)
            try:
                self.ml_model.fit(X.loc[:, X.columns != "batch"], y, weights)
            except ValueError:
                print("Error in the weights!!", self.name, ", index:", X[-1:].index[0])
                self.ml_model.fit(X.loc[:, X.columns != "batch"], y)
            scores = self.weight_model.likelihoods[self.batch].score_samples(df[-self.old_to_use:]
                                                                             .loc[:, df.columns != "batch"])
            self.mean = np.mean(scores)
            self.std = np.std(scores)
            self.last = X[:-1].index[0]
            if self.verbose:
                print("drift detected", self.name)
            self.detected_drifts.append([X[-1:].index[0], self.batch])
        self.elapsed_time += (time.time()-start)

    def update_batch(self, X, y, last):
        self.batch += 1
        self.batch_data[-last:] = [self.batch] * last
        df = pd.concat([X, y], axis=1)
        df.batch = self.batch_data[:-1]  # but check bug!!!!!
        weights = self.weight_model.get_weights(df, self.batch)
        return weights, df


class ErrorDriftModel(DensityDriftModel):
    """ Online Model class with error-based drift detection.
       The Model class is a general configuration for the framework in a online-based context, both for regression and
       classification. It stores the underlying learner, WeightModel instance and a drift detector using the
        error of the learner. At every instance,  the new output is predicted and the drift detector is updated.
        Based on its state, retraining can be performed.

        Parameters
        ----------
        weight_model : a WeightModel instance
        A WeightModel instance, which stores reweighting schemes and density model.

        train_size : int
        Up to which index of the data, it's pre-training data.

        ml_model : sklearn-like estimator instance
        A machine learning model which complies with Scikit-learn's conventions.

        drift_detector : River Drift Detector instance, default = PageHinkley()
        The chosen drift detector. For regression, only ADWIN and PageHinkley are supported.

        name : str, default="None"
        The name identifier for the model.

        old_to_use : int, default=100
        Old to use hyperparameter. It regulates how many of the last samples are used for fitting the new density model.

        update_wm : int, default=200
        Update hyperparameter. It is the number of instances it waits before refitting the density model and, thus,
        creating a new batch, even if drift is not detected.

        whiten : bool, default=True
        Whether the error has to be whitened before giving it to drift detectors. Recommended for regression,
        not recommended for classification.

    """
    def __init__(self, weight_model, train_size=1000, ml_model=GaussianNB(),
                 drift_detector=PageHinkley,  # lambda_option=0.1),
                 name="None", old_to_use=50, update_wm=150, verbose=True, whiten=True):
        super().__init__(weight_model, train_size, ml_model, drift_detector, name, old_to_use, update_wm, verbose)
        self.whiten = whiten

    def initial_fit(self, X, y):
        start = time.time()
        df = pd.concat([X, y], axis=1)
        self.weight_model.initial_fit(df)
        self.ml_model.fit(X.loc[:, X.columns != "batch"].values, y.values)
        self.batch_data = [0] * X.shape[0]
        self.elapsed_time += (time.time() - start)
        # preds = self.ml_model.predict(X.loc[:, X.columns != "batch"])
        # if self.whiten:  # here, whitening does not make sense in classification
        #     scores = [mean_squared_error(y[i:i+1], preds[i:i+1], squared=False) for i in range(len(preds))]
        #     self.mean = np.mean(scores)
        #     self.std = np.std(scores)

    def partial_fit(self, X, y, error):
        start = time.time()
        self.batch_data.append(self.batch)
        if self.whiten:
            error = (error - self.mean) / self.std
        if not self.whiten:
            in_drift, in_warning = self.drift_detector.update(error.values[0])
        else:
            in_drift, in_warning = self.drift_detector.update(error)
        if not in_warning and not in_drift and (X[-1:].index[0] - self.last) > self.update_wm:
            if self.weight_model.likelihoods:
                _, df = self.update_batch(X, y, self.update_wm)
            else:
                _, df = self.update_batch(X, y, self.update_wm, return_weights=False)
            self.last = X[-1:].index[0]
            preds = self.ml_model.predict(X[-self.old_to_use:].loc[:, X.columns != "batch"].values)
            scores = [mean_squared_error(y[i:i+1], preds[i:i+1], squared=False) for i in range(len(preds))]
            if self.whiten:
                self.mean = np.mean(scores)
                self.std = np.std(scores)

        if in_drift:
            if self.verbose:
                print("drift detected", X[-1:].index[0], self.name)
            weights, df = self.update_batch(X, y, self.old_to_use)
            try:
                self.ml_model.fit(X.loc[:, X.columns != "batch"].values, y.values, weights)
            except ValueError:
                print("Error in the weights!!", self.name)
                print(y.values)
                print(X.loc[:, X.columns != "batch"])
                self.ml_model.fit((X.loc[:, X.columns != "batch"]).values, y.values)
            self.last = X[:-1].index[0]

            self.detected_drifts.append([X[-1:].index[0], self.batch])
            preds = self.ml_model.predict(X[-self.update_wm:].loc[:, X.columns != "batch"].values)
            if self.whiten:
                scores = [mean_squared_error(y[i:i+1], preds[i:i+1], squared=False) for i in range(len(preds))]
                self.mean = np.mean(scores)
                self.std = np.std(scores)
        self.elapsed_time += (time.time() - start)

    def update_batch(self, X, y, last, return_weights=True):
        self.batch += 1
        self.batch_data[-last:] = [self.batch] * last
        df = pd.concat([X, y], axis=1)
        # df.batch = self.batch_data[:-1]
        df.batch = self.batch_data
        # here, i only call get weights to update the internal state of the wm with new likelihoods
        if return_weights:
            weights = self.weight_model.get_weights(df, self.batch)
            return weights, df  # i do not need, in theory to return weights when only updating
        else:
            return None, df


class EWMAMetaModel:
    """ Expert MetaModel which takes in input different Models' configurations and, online, decides
        which configuration to query. This class is to be used in order to use the expert online.
        However, for time efficiency, it makes sense to compute the experts' predictions offline
        based on the learners' predictions, as this class is still not optimized in order to
        store the underlying learners' metrics.
        Warning: For now, it does not support classification! The results are obtained by saving the
        pointwise aucs approximations and running the expert outside the code.

        Parameters
        ----------
        models : a dictionary of ErrorDriftModels/DensityDriftModels (with name as key)
        The models from which the expert should choose from.

        alpha: float, default = 0.1
        Exponential moving average forgetting factor.

        name : str, default="None"
        The name identifier for the model.

    """

    def __init__(self, models, alpha=0.1, name="EWMAMM"):
        self.name = name
        self.models = models
        self.alpha = alpha
        self.running_rmse = dict((k, 0) for k in models.keys())  # tracks the running rmse for the models
        self.models_used = []  # tracks the used models in time
        self.elapsed_time = 0

    def initial_fit(self, X, y):
        for _, model in self.models.items():
            model.initial_fit(X, y)

    def partial_fit(self, X, y):
        start = time.time()
        for model_name, model in self.models.items():
            preds = model.predict(X[-1:].loc[:, X.columns != "batch"])
            rmse_local = mean_squared_error(y[-1:], preds, squared=False)
            self.running_rmse[model_name] = self.running_rmse[model_name]*(1-self.alpha) + rmse_local*self.alpha

            if isinstance(model, ErrorDriftModel):
                model.partial_fit(X, y, rmse_local)  # data i have up to that point
            else:
                model.partial_fit(X, y)
        self.elapsed_time += (time.time() - start)

    def predict(self, X):
        model_name = min(self.running_rmse, key=self.running_rmse.get)
        self.models_used.append(model_name)
        return self.models[model_name].predict(X.loc[:, X.columns != "batch"])



    """ Batch-based Environment for simulations. It is setup by passing the entire datastream and, then, used
    by calling the run_experiment method. It stores all the models' metrics and predictions for analysis.

        Parameters
        ----------
        dataset: array-like of shape (n_samples, n_features+n_targets+1)
        Dataset representing the whole data stream. The target to be predicted MUST be called "label". Moreover,
        an additional "batch" column is needed.

        train_size : int, default = 1000
        How much of the dataset to use for pre-training.

        batch_size : int, default = 100
        Size of each batch. For now, only supports batches with the same number of samples.

        length : int, default = 50
        Length (in batches!) of the experiments.

        problem : str, default = "Regression"
        Problem type: only support classification and regression.

    """

class Environment:
    def __init__(self, dataset, train_size=1000, batch_size=100, length=50, problem="Regression", verbose=True):
        self.dataset = dataset
        self.X = dataset.loc[:, dataset.columns != 'label']
        self.y = dataset["label"]
        self.train_size = train_size
        self.batch_size = batch_size
        self.length = length
        self.current_index = train_size
        self.aucs = None  # only for classification
        self.r2 = None  # only for regression
        self.rmse = None  # only for regression
        self.elapsed_times = None
        self.predictions = None
        self.verbose = verbose
        self.problem = problem

    # models = dictionary of models
    def run_experiment(self, models):
        """ Runs the experiment for this environment instance.
        Parameters
        ----------
        models:  a dictionary of Models (with name as key)
        The models for which to run the experiment and log results.

        """
        self.elapsed_times = dict((k, []) for k in models.keys())
        self.predictions = dict((k, []) for k in models.keys())
        if self.problem == "Classification":
            self.aucs = dict((k, []) for k in models.keys())
        elif self.problem == "Regression":
            self.r2 = dict((k, []) for k in models.keys())
            self.rmse = dict((k, []) for k in models.keys())
        else:
            raise ValueError('Only supports Classification and Regression problems')

        for _, model in models.items():
            model.initial_fit(self.X[:self.train_size], self.y[:self.train_size])
        if self.verbose:
            pbar = tqdm(range(self.length - 2))

        for i in np.arange(1, self.length - 1):

            for model_name, model in models.items():
                model.fit(self.X[:self.train_size + i * self.batch_size],  # data i have up to that point
                          self.y[:self.train_size + i * self.batch_size], i)

                if self.problem == "Classification":

                    raise NotImplementedError
                    #probas = model.predict_proba(
                    #    self.X[self.train_size + i * self.batch_size:self.train_size + (i + 1) * self.batch_size])
                    #auc_local = metric = metric.update(yt, yp)
                    #self.aucs[model_name].append(auc_local)

                    # for now, batch classification is not supported!

                elif self.problem == "Regression":
                    # predict, then score
                    preds = model.predict(self.X[self.train_size + i * self.batch_size:self.train_size
                                                                                       + (i + 1) * self.batch_size])
                    r2_local = r2_score(
                        self.y[self.train_size + i * self.batch_size:self.train_size + (i + 1) * self.batch_size],
                        preds)
                    self.predictions[model_name].append(preds)
                    rmse_local = mean_squared_error(
                        self.y[self.train_size + i * self.batch_size:self.train_size + (i + 1) * self.batch_size],
                        preds, squared=False)
                    self.r2[model_name].append(r2_local)
                    self.rmse[model_name].append(rmse_local)

                # here you could build other ds to store what you want
            if self.verbose:
                pbar.update(1)

        for model_name, model in models.items():
            self.elapsed_times[model_name].append(model.elapsed_time)


class OnlineEnvironment(Environment):
    """ Online-based Environment for simulations. It is setup by passing the entire datastream and, then, used
    by calling the run_experiment method. It stores all the models' metrics and predictions for analysis.

        Parameters
        ----------
        dataset: array-like of shape (n_samples, n_features+n_targets+1)
        Dataset representing the whole data stream. The target to be predicted MUST be called "label". Moreover,
        an additional "batch" column is needed.

        train_size : int, default = 100
        How much of the dataset to use for pre-training.

        length : int, default = 4000
        Length (in samples) for the experiment.

        problem : str, default = "Regression"
        Problem type: only support classification and regression.

    """
    def __init__(self, dataset, train_size=100, length=4000, problem="Regression",
                 verbose=False):

        super().__init__(dataset=dataset, train_size=train_size, batch_size=None, length=length,
                         problem=problem, verbose=verbose)
        self.track = None

    def run_experiment(self, models):
        self.elapsed_times = dict((k, []) for k in models.keys())
        self.predictions = dict((k, []) for k in models.keys())
        if self.problem == "Classification":
            self.aucs = dict((k, []) for k in models.keys())
            self.river_aucs = dict((k, metrics.ROCAUC(n_thresholds=20)) for k in models.keys())
            # warning : for now, for classification, it is only going to save the cumulative auc!!!
        else:
            raise ValueError('Only supports Classification problems')

        for _, model in models.items():
            model.initial_fit(self.X[:self.train_size], self.y[:self.train_size])
        if self.verbose:
            pbar = tqdm(range(self.length - self.train_size))
        for i in np.arange(self.train_size, self.length):
            for model_name, model in models.items():

                probas = model.predict_proba(
                    self.X[i:i + 1])
                self.river_aucs[model_name] = self.river_aucs[model_name].\
                    update(int(list(self.y[i:i + 1].values)[0]), float(list(probas[:, 1])[0]))
                error = (self.y[i:i + 1] != (probas[:, 1] > 0.5))
                self.aucs[model_name].append(self.river_aucs[model_name].get())
                self.predictions[model_name].append(probas[:, 1])
                if isinstance(model, ErrorDriftModel):
                    model.partial_fit(self.X[:i], self.y[:i], error)  # 1-0 error is better for detection
                else:
                    model.partial_fit(self.X[:i], self.y[:i])



                # here you could build other ds to store what you want
            if self.verbose:
                pbar.update(1)

            for model_name, model in models.items():
                self.elapsed_times[model_name].append(model.elapsed_time)
        print(self.predictions)
