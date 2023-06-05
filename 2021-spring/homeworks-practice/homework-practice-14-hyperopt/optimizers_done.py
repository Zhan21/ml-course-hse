import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Union, Optional
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold
from distributions import BaseDistribution, NumericalDistribution, CategoricalDistribution

from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RBF
from sklearn.neighbors import KernelDensity, NearestNeighbors


class BaseOptimizer(BaseEstimator, ABC):
    '''
    A base class for all hyperparameter optimizers
    '''
    def __init__(self, estimator: BaseEstimator, param_distributions: Dict[str, BaseDistribution],
                 scoring: Optional[Callable[[np.array, np.array], float]] = None, cv: int = 3, num_runs: int = 100,
                 num_dry_runs: int = 5, num_samples_per_run: int = 20, n_jobs: Optional[int] = None,
                 verbose: bool = False, random_state: Optional[int] = None):
        '''
        Params:
          - estimator: sklearn model instance
          - param_distributions: a dictionary of parameter distributions,
            e.g. param_distributions['num_epochs'] = IntUniformDistribution(100, 200)
          - scoring: sklearn scoring object, see
            https://scikit-learn.org/stable/modules/model_evaluation.html
            # scoring-parameter if left None estimator must have 'score' attribute
          - cv: number of folds to cross-validate
          - num_runs: number of iterations to fit hyperparameters
          - num_dry_runs: number of dry runs (i.e. random strategy steps) to gather initial statistics
          - num_samples_per_run: number of hyperparameters set to sample each iteration
          - n_jobs: number of parallel processes to fit algorithms
          - verbose: whether to print debugging information (you can configure debug as you wish)
          - random_state: RNG seed to control reproducibility
        '''
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.scoring = scoring
        self.cv = cv
        self.num_runs = num_runs
        self.num_samples_per_run = num_samples_per_run
        self.num_dry_runs = num_dry_runs
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.reset()

    def reset(self):
        '''
        Reset fields used for fitting
        '''
        self.splitter = None
        self.best_score = None
        self.best_params = None
        self.best_estimator = None
        self.params_history = {
            name: np.array([]) for name in self.param_distributions
        }
        self.scores_history = np.array([])
        
    def sample_params(self) -> Dict[str, np.array]:
        '''
        Sample self.num_samples_per_run set of hyperparameters
        Returns:
          - sampled_params: dict of arrays of parameter samples,
            e.g. sampled_params['num_epochs'] = np.array([178, 112, 155])
        '''
        sampled_params = {}
        # Your code here (⊃｡•́‿•̀｡)⊃
        for param in self.param_distributions.keys():
            sampled_params[param] = self.param_distributions[param].sample(self.num_samples_per_run)
            
        return sampled_params

    @abstractmethod
    def select_params(self, params_history: Dict[str, np.array], scores_history: np.array,
                      sampled_params: Dict[str, np.array]) -> Dict[str, Any]:
        '''
        Select new set of parameters according to a specific search strategy
        Params:
          - params_history: list of hyperparameter values from previous interations
          - scores_history: corresponding array of CV scores
          - sampled_params: dict of arrays of parameter samples to select from
        Returns:
          - new_params: a dict of new hyperparameter values
        '''
        msg = f'method \"select_params\" not implemented for {self.__class__.__name__}'
        raise NotImplementedError(msg)

    def cross_validate(self, X: np.array, y: Optional[np.array],
                       params: Dict[str, Any]) -> float:
        '''
        Calculate cross-validation score for a set of params
        Consider using estimator.set_params() and sklearn.model_selection.cross_validate()
        Also use self.splitter as a cv parameter in cross_validate
        Params:
          - X: object features
          - y: object labels
          - params: a set of params to score
        Returns:
          - score: mean cross-validation score
        '''
        score = 0.0
        # Your code here (⊃｡•́‿•̀｡)⊃
        try:
            self.estimator.set_params(random_state=self.random_state, **params)
        except:
            self.estimator.set_params(**params)
        cv_results = cross_validate(self.estimator, X, y, 
                                    cv=self.splitter,
                                    scoring=self.scoring)
        score = cv_results['test_score'].mean()
        
        return score

    def fit(self, X_train: np.array, y_train: Optional[np.array] = None) -> BaseEstimator:
        '''
        Find the best set of hyperparameters with a specific search strategy
        using cross-validation and fit self.best_estimator on whole training set
        Params:
          - X_train: array of train features of shape (num_samples, num_features)
          - y_train: array of train labels of shape (num_samples, )
            if left None task is unsupervised
        Returns:
          - self: (sklearn standard convention)
        '''
        np.random.seed(self.random_state)
        self.reset()
        
        if y_train is not None and np.issubdtype(y_train.dtype, np.integer):
            self.splitter = StratifiedKFold(n_splits=self.cv, shuffle=True,
                                            random_state=self.random_state)
        else:
            self.splitter = KFold(n_splits=self.cv, shuffle=True,
                                  random_state=self.random_state)
        # Your code here (⊃｡•́‿•̀｡)⊃
        for _ in range(self.num_runs): 
            params = self.select_params(self.params_history, 
                                        self.scores_history, 
                                        self.sample_params())
            score = self.cross_validate(X_train, y_train, params)

            for param in params.keys():
                self.params_history[param] = np.append(self.params_history[param], params[param])
            self.scores_history = np.append(self.scores_history, score)
            
            if (self.best_score is None) or (score > self.best_score):
                self.best_score = score
                self.best_params = params
        try:
            best_est = self.estimator.set_params(random_state=self.random_state, **self.best_params)
        except:
            best_est = self.estimator.set_params(**self.best_params)
        self.best_estimator = best_est
        
        if y_train is not None:
            self.best_estimator.fit(X_train, y_train)
        else:
            self.best_estimator.fit(X_train)
            
        return self

    def predict(self, X_test: np.array) -> np.array:
        '''
        Generate a prediction using self.best_estimator
        Params:
          - X_test: array of test features of shape (num_samples, num_features)
        Returns:
          - y_pred: array of test predictions of shape (num_samples, )
        '''
        if self.best_estimator is None:
            raise ValueError('Optimizer not fitted yet')
            
        # y_pred = np.zeros(X_test.shape[0])
        # your code here ┐(シ)┌
        y_pred = self.best_estimator.predict(X_test)
        
        return y_pred

    def predict_proba(self, X_test: np.array) -> np.array:
        '''
        Generate a probability prediction using self.best_estimator
        Params:
          - X_test: array of test features of shape (num_samples, num_features)
        Returns:
          - y_pred: array of test probabilities of shape (num_samples, num_classes)
        '''
        if self.best_estimator is None:
            raise ValueError('Optimizer not fitted yet')
            
        if not hasattr(self.best_estimator, 'predict_proba'):
            raise ValueError('Estimator does not support predict_proba')

        # y_pred = np.zeros(X_test.shape[0], num_classes)
        # your code here ┐(シ)┌
        y_pred = self.best_estimator.predict_proba(X_test)
        
        return y_pred


class RandomSearchOptimizer(BaseOptimizer):
    '''
    An optimizer implementing random search strategy
    '''
    def __init__(self, estimator: BaseEstimator, param_distributions: Dict[str, BaseDistribution],
                 scoring: Optional[Callable[[np.array, np.array], float]] = None, cv: int = 3, num_runs: int = 100,
                 n_jobs: Optional[int] = None, verbose: bool = False, random_state: Optional[int] = None):
        super().__init__(
            estimator, param_distributions, scoring, cv=cv,
            num_runs=num_runs, num_dry_runs=0, num_samples_per_run=1,
            n_jobs=n_jobs, verbose=verbose, random_state=random_state
        )
        
    def select_params(self, params_history: Dict[str, np.array], scores_history: np.array,
                      sampled_params: Dict[str, np.array]) -> Dict[str, Any]:
        '''
        Select new set of parameters according to a specific search strategy
        Params:
          - params_history: list of hyperparameter values from previous interations
          - scores_history: corresponding array of CV scores
          - sampled_params: dict of arrays of parameter samples to select from
        Returns:
          - new_params: a dict of new hyperparameter values
        '''
        new_params = {}
        # Your code here (⊃｡•́‿•̀｡)⊃
        for param in params_history.keys():
            new_params[param] = np.random.choice(sampled_params[param])
            
        return new_params


class GPOptimizer(BaseOptimizer):
    '''
    An optimizer implementing gaussian process strategy
    '''
    @staticmethod
    def calculate_expected_improvement(y_star: float, mu: np.array,
                                       sigma: np.array) -> np.array:
        '''
        Calculate EI values for passed parameters of normal distribution
        hint: consider using scipy.stats.norm
        Params:
          - y_star: optimal (minimal) score value
          - mu: array of mean values of normal distribution of size (num_samples_per_run, )
          - sigma: array of std values of normal distribution of size (num_samples_per_run, )
        Retuns:
          - ei: array of EI values of size (num_samples_per_run, )
        '''
        ei = np.zeros_like(mu)
        # Your code here (⊃｡•́‿•̀｡)⊃
        u_star = (y_star - mu) / sigma
        ei = - sigma * u_star * (1 - norm.cdf(u_star)) + sigma * norm.pdf(u_star)
        
        return ei

    def select_params(self, params_history: Dict[str, np.array], scores_history: np.array,
                      sampled_params: Dict[str, np.array]) -> Dict[str, Any]:
        new_params = {}
        # Your code here (⊃｡•́‿•̀｡)⊃
        
        # Набираем статистику
        if len(scores_history) < self.num_dry_runs:
            for param in params_history.keys():
                new_params[param] = np.random.choice(sampled_params[param])
                
            return new_params
        else:
            # Разделяем гиперпараметры на числовые и категориальные
            cat_params, num_params = [], []
            for i, param in enumerate(params_history.keys()):
                if self.param_distributions[param].__class__.__name__ == "CategoricalDistribution":
                    cat_params.append(param)
                elif self.param_distributions[param].__class__.__name__ != "CategoricalDistribution":
                    num_params.append(param)
                    
            # Скейлим и собираем history в матрицу
            X_num = np.zeros([len(scores_history), len(num_params)])  # [num_iters, params]
            for i, param in enumerate(num_params):
                scaled_param = self.param_distributions[param].scale(params_history[param])
                X_num[:, i] = scaled_param
                
            # Для числовых гиперпараметров
            if len(num_params) > 0:
                # Обучаем Гауссовский Процесс на числовых гиперпараметрах
                kernel = ConstantKernel() + WhiteKernel() + RBF()
                gpr = GaussianProcessRegressor(kernel=kernel, 
                                               random_state=self.random_state)
                gpr.fit(X_num, scores_history)
                
                # Скейлим и собираем sample в матрицу
                sample_size = len(sampled_params[num_params[0]])
                X_num_sample = np.zeros([sample_size, len(num_params)])  # [sample_size, params]
                for i, param in enumerate(num_params):
                    scaled_param_sample = self.param_distributions[param].scale(sampled_params[param])
                    X_num_sample[:, i] = scaled_param_sample
                    
                # Ищем лучший гиперпараметр через индекс максимального Exepcted Improvement
                mu, sigma = gpr.predict(X_num_sample, return_std=True)
                y_star = np.max(scores_history)
                
                ei = self.calculate_expected_improvement(y_star, mu, sigma)
                best_param_idx = np.argmax(ei)
                
                for param in num_params:
                    new_params[param] = sampled_params[param][best_param_idx]
                
            # Для категориальных гиперпараметров
            if len(cat_params) > 0:
                for param in cat_params:
                    mu, sigma = np.array([]), np.array([])
                    f_vec = params_history[param]
                    y_star = np.max(scores_history)
                    
                    for c in self.param_distributions[param].categories:
                        # mu
                        mu_c = np.sum((f_vec == c) * scores_history) / np.sum(f_vec == c)
                        if np.sum(f_vec == c) != 0:
                            mu = np.append(mu, mu_c)
                        else:
                            mu = np.append(mu, y_star)
                            
                        # sigma
                        if np.sum(f_vec == c) != 0:
                            sigma_sq_c = (1 + np.sum((f_vec == c) * (scores_history - mu_c) ** 2)) / (1 + np.sum(f_vec == c))
                        else:
                            sigma_sq_c = 0
                        sigma = np.append(sigma, np.sqrt(sigma_sq_c))
                    
                    # Ищем лучший гиперпараметр через индекс максимального Exepcted Improvement
                    ei = self.calculate_expected_improvement(y_star, mu, sigma)
                    best_param_idx = np.argmax(ei)
                    
                    new_params[param] = self.param_distributions[param].categories[best_param_idx]
                    
        return new_params
    
    

class TPEOptimizer(BaseOptimizer):
    '''
    An optimizer implementing tree-structured Parzen estimator strategy
    '''
    def __init__(self, estimator: BaseEstimator, param_distributions: Dict[str, BaseDistribution],
                 scoring: Optional[Callable[[np.array, np.array], float]] = None, cv: int = 3, num_runs: int = 100,
                 num_dry_runs: int = 5, num_samples_per_run: int = 20, gamma: float = 0.75,
                 n_jobs: Optional[int] = None, verbose: bool = False, random_state: Optional[int] = None):
        '''
        Params:
          - gamma: scores quantile used for history splitting
        '''
        super().__init__(
            estimator, param_distributions, scoring, cv=cv, num_runs=num_runs,
            num_dry_runs=num_dry_runs, num_samples_per_run=num_samples_per_run,
            n_jobs=n_jobs, verbose=verbose, random_state=random_state
        )
        self.gamma = gamma

    @staticmethod
    def estimate_log_density(scaled_params_history: np.array,
                             scaled_sampled_params: np.array, bandwidth: float):
        '''
        Estimate log density of sampled numerical hyperparameters based on
        numerical hyperparameters history subset
        Params:
          - scaled_params_history: array of scaled numerical hyperparameters history subset
            of size (subset_size, num_numerical_params)
          - scaled_sampled_params: array of scaled sampled numerical hyperparameters
            of size (num_samples_per_run, num_numerical_params)
          - bandwidth: bandwidth for KDE
        Returns:
          - log_density: array of estimated log probabilities of size (num_samples_per_run, )
        '''
        # log_density = np.zeros(self.num_samples_per_run)
        # Your code here (⊃｡•́‿•̀｡)⊃
        
        kernel_dens = KernelDensity(bandwidth=bandwidth)
        kernel_dens.fit(scaled_params_history)
        
        log_density = kernel_dens.score_samples(scaled_sampled_params)
        return log_density
    
    def select_params(self, params_history: Dict[str, np.array], scores_history: np.array,
                      sampled_params: Dict[str, np.array]) -> Dict[str, Any]:
        np.random.seed(self.random_state)
        new_params = {}
        
        # Набираем статистику
        if len(scores_history) < self.num_dry_runs:
            for param in params_history.keys():
                new_params[param] = np.random.choice(sampled_params[param])
                
            return new_params
        else:
            # Разделяем гиперпараметры на числовые и категориальные
            cat_params, num_params = [], []
            for i, param in enumerate(params_history.keys()):
                if self.param_distributions[param].__class__.__name__ == "CategoricalDistribution":
                    cat_params.append(param)
                elif self.param_distributions[param].__class__.__name__ != "CategoricalDistribution":
                    num_params.append(param)
                    
            # Скейлим и собираем history в матрицу
            X_num = np.zeros([len(scores_history), len(num_params)])  # [num_iters, params]
            X_cat = np.zeros([len(scores_history), len(cat_params)])  # [num_iters, params]
            
            for i, param in enumerate(num_params):
                distr = self.param_distributions[param]
                X_num[:, i] = distr.scale(params_history[param])
                
            for i, param in enumerate(cat_params):
                distr = self.param_distributions[param]
                X_cat[:, i] = distr.scale(params_history[param])
                
            # Для числовых гиперпараметров
            if len(num_params) > 0:
                # Разделяем выборку по квантилю
                quant_level = np.quantile(scores_history, self.gamma)
                X_great = X_num[np.where(scores_history >= quant_level)]
                X_lower = X_num[np.where(scores_history < quant_level)]
                
                # Скейлим и собираем sample в матрицу
                sample_size = len(sampled_params[num_params[0]])
                X_num_sample = np.zeros([sample_size, len(num_params)])  # [sample_size, params]
                for i, param in enumerate(num_params):
                    scaled_param_sample = self.param_distributions[param].scale(sampled_params[param])
                    X_num_sample[:, i] = scaled_param_sample
                
                # Ищем лучший гиперпараметр
                nn = NearestNeighbors(n_neighbors=1).fit(X_num)
                bandwidth = np.median(nn.kneighbors()[0])
                
                g_log_dens = self.estimate_log_density(X_great, X_num_sample, bandwidth)
                l_log_dens = self.estimate_log_density(X_lower, X_num_sample, bandwidth)
                
                best_param_idx = np.argmax(g_log_dens - l_log_dens)
                for param in num_params:
                    new_params[param] = sampled_params[param][best_param_idx]
                
            # Для категориальных гиперпараметров
            if len(cat_params) > 0:
                mask_great = np.where(scores_history >= np.quantile(scores_history, self.gamma))
                mask_lower = np.where(scores_history < np.quantile(scores_history, self.gamma))
                X_great = X_cat[mask_great]
                X_lower = X_cat[mask_lower]
                
                for param in cat_params:
                    cats = self.param_distributions[cat].categories
                    prob_cat = 1 / len(cats)
                    
                    mx = np.full((len(cats), len(params_history[param])),
                                fill_value=params_history[param])
                    mx = (mx == cats.reshape(-1, 1)).T
                    
                    g_cat = X_great.shape[0] * prob_cat + np.sum(mx[mask_great], axis=0)
                    l_cat = X_lower.shape[0] * prob_cat + np.sum(mx[mask_lower], axis=0)
                    
                    best_param_idx = np.argmax(g_cat / l_cat)
                    new_params[cat] = cats[best_param_idx]

        return new_params
        
        
        
        
        
        
        
        return new_params
