import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel

from ..preprocessing import normalization
from ..metrics.pairwise.vector import linear_kernel
from ..metrics.pairwise import homogeneous_polynomial_kernel as hpk
from ..generators import HPK_generator, RBF_generator

from ..algorithms import AverageMKL, EasyMKL

class KernelTransformer(BaseEstimator, TransformerMixin):
    """sklearn-transformer-style wrapper to handle MKLpy-based kernel computation.
    Can be used as a transformer (such as StandardScaler) inside a pipeline.
    input:
        N: list: contains number of features per data modality. len=1 for unimodal modeling.
        kernel_types: list: kernels to be used for each modality. len=1 for unimodal modeling.
        norm_data: bool: whether to normalize the data using MKLpy's normalization.
        degrees: list: hyperparameters for the hpk kernel
        gamma: list: hyperparameters for the rbf kernel
        """
        
    def __init__(self, N: list, kernel_types: list, norm_data: bool=True, degrees: list=[1,2,3], gamma: list=[.001, .01, .1]) -> None:
        # Initialize the degree of the polynomial kernel
        self.reference_data_ = None
        self.N = N
        self.kernel_types = kernel_types
        self.norm_data = norm_data
        self.degrees = degrees
        self.gamma = gamma
        assert len(self.N) == len(self.kernel_types)
        
    def fit(self, X, y=None):       
        # Compute the kernel matrix for the training data per modality
        assert np.sum(self.N) == X.shape[1]
        self.K_train_ = []
        self.reference_data_ = self.normalize_data_(X) if self.norm_data else X
        
        sum = 0
        for i, n in enumerate(self.N):
            self.K_train_.extend(self.compute_kernel_(self.reference_data_[:, sum:sum+n],
                                k_type=self.kernel_types[i]))
            sum += n

        return self

    def transform(self, X):
        # Apply the kernel transformation to input data
        assert np.sum(self.N) == X.shape[1]
        if self.reference_data_ is None:
            raise ValueError("Transformer has not been fitted")
        
        # If X is the training data, return the precomputed kernel matrix
        normalized_data = self.normalize_data_(X) if self.norm_data else X
        if normalized_data is self.reference_data_:
            return self.K_train_

        K = []
        sum = 0
        for i, n in enumerate(self.N):
            K.extend(self.compute_kernel_(normalized_data[:, sum:sum+n],
                                          self.reference_data_[:, sum:sum+n], 
                                        k_type=self.kernel_types[i]))
            sum += n

        return K
            
    def compute_kernel_(self, X, y=None, k_type="linear"):
        if k_type == "hpk":
            if isinstance(self.degrees, float) or isinstance(self.degrees, int):
                return [hpk(X, y, degree=self.degrees)]
            else:
                return HPK_generator(X, y, degrees=self.degrees)
        elif k_type == "rbf":
            if isinstance(self.gamma, float):
                return [rbf_kernel(X, y, gamma=self.gamma)]
            else:
                return RBF_generator(X, y, gamma=self.gamma)
        elif k_type == "linear":
            return [linear_kernel(X, y)] 
        else: 
            raise NotImplementedError("Kernel type not implemented.")

    def normalize_data_(self, X):
        normalized_data = np.empty_like(X)
        sum = 0
        for n in self.N:
            normalized_data[:, sum:sum+n] = normalization(X[:, sum:sum+n])
            sum += n
        return normalized_data
    
class EasyMKLWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, classification=True, *args, **kwargs):
        self.model = EasyMKL(*args, **kwargs)
        self.classification = classification
        assert self.classification == True

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        if self.classification:
            from sklearn.metrics import balanced_accuracy_score
            return balanced_accuracy_score(y, self.predict(X))
        else:
            from sklearn.metrics import r2_score
            return r2_score(y, self.predict(X))
    
    def predict_proba(self, X):
        return self.model.learner.predict_proba(self.model.func_form(X, self.model.solution.weights))

    def set_params(self, **params):
        for param, value in params.items():
            if '__' in param:
                # Nested parameter
                param_list = param.split('__')
                target = self.model
                for p in param_list[:-1]:
                    target = getattr(target, p)
                setattr(target, param_list[-1], value)
            else:
                # Top-level parameter
                setattr(self.model, param, value)
        return self

    def get_params(self, deep=True):
        params = {}
        if deep:
            for key, value in self.model.get_params(deep=True).items():
                params[key] = value
                if hasattr(value, 'get_params'):
                    for sub_key, sub_value in value.get_params(deep=True).items():
                        params[f'{key}__{sub_key}'] = sub_value
        else:
            params = self.model.get_params(deep=False)
        return params
    
    def get_weights(self):
        return self.model.solution.weights.numpy()
    
    
class AverageMKLWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, classification=True, *args, **kwargs):
        self.model = AverageMKL(classification=classification, *args, **kwargs)
        self.classification = classification

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        if self.classification:
            from sklearn.metrics import balanced_accuracy_score
            return balanced_accuracy_score(y, self.predict(X))
        else:
            from sklearn.metrics import r2_score
            return r2_score(y, self.predict(X))
    
    def predict_proba(self, X):
        assert self.classification == True
        return self.model.learner.predict_proba(self.model.func_form(X, self.model.solution.weights))

    def set_params(self, **params):
        for param, value in params.items():
            if '__' in param:
                # Nested parameter
                param_list = param.split('__')
                target = self.model
                for p in param_list[:-1]:
                    target = getattr(target, p)
                setattr(target, param_list[-1], value)
            else:
                # Top-level parameter
                if param == "classification":
                    self.classification = value
                setattr(self.model, param, value)
        return self

    def get_params(self, deep=True):
        params = {"classification": self.classification}
        if deep:
            for key, value in self.model.get_params(deep=True).items():
                params[key] = value
                if hasattr(value, 'get_params'):
                    for sub_key, sub_value in value.get_params(deep=True).items():
                        params[f'{key}__{sub_key}'] = sub_value
        else:
            params = self.model.get_params(deep=False)
        return params