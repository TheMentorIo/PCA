import numpy as np

class PCA_SVD():
    def __init__(self,  new_dim:int) -> None:
        # hyperparameter representing the number of dimensions after reduction
        self.new_dim = new_dim
        # for standardization
        self.mean:np.ndarray
        self.std:np.ndarray
        # for PCA
        self.U:np.ndarray
        self.S:np.ndarray 
        self.V:np.ndarray  

    # x_train is (m,n) matrix where each row is an n-dimensional vector of features
    def fit(self, x_train):
        # TODO 1: Find mean and std of each feature in x_train
        self.mean = np.mean(x_train, axis=0)
        self.std = np.std(x_train, axis=0, ddof=1)
        # TODO 2: Standardize the training data
        z_train = (x_train - self.mean) / self.std

        if isinstance(self.new_dim, float) and 0 < self.new_dim < 1:
            eigenvalues = self.S ** 2 / (len(x_train) - 1)
            explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
            cumulative_variance = np.cumsum(explained_variance_ratio)
            n_components = np.argmax(cumulative_variance >= self.new_dim) + 1
        else:
            n_components = self.new_dim
                
        self.U, self.S, Vt = np.linalg.svd(z_train, full_matrices=False)
        self.V = Vt.T[:, :n_components]
        self.U = self.U[:, :n_components]
        self.S = self.S[:n_components]

        return self
    
    # x_val is (m,n) matrix where each row is an n-dimensional vector of features
    def transform(self, x_val):
        z_val = (x_val - self.mean) / self.std
        # TODO 7: Apply the transformation equation
        return z_val @ self.V
    
    def inverse_transform(self, z_val):
        # TODO 8: Apply the inverse transformation equation (including destandardization)
        z_train = z_val @ self.V.T
        return z_train * self.std + self.mean

    def fit_transform(self, x_train):
        return self.fit(x_train).transform(x_train)