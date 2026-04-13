import numpy as np

class PCA():
    def __init__(self,  new_dim:int) -> None:
        # hyperparameter representing the number of dimensions after reduction
        self.new_dim = new_dim
        # for standardization
        self.mean:np.ndarray
        self.std:np.ndarray
        # for PCA
        self.A:np.ndarray 

    # x_train is (m,n) matrix where each row is an n-dimensional vector of features
    def fit(self, x_train):
        # TODO 1: Find mean and std of each feature in x_train
        self.mean = np.mean(x_train, axis=0)
        self.std = np.std(x_train, axis=0, ddof=1)
        # TODO 2: Standardize the training data
        z_train = x_train - self.mean
                
        # TODO 3: Compute the covariance matrix
        cov_matrix = np.cov(z_train,rowvar=False)
        
        # TODO 4: Compute eigenvalues and eigenvectors using Numpy
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real           # sometimes a zero imaginary part can appear due to approximations
        
        # TODO 5: Sort eigenvalues and eigenvectors
        # TODO 5.1: Find the sequence of indices that sort eigenvalues in descending order
        sorting_inds = np.argsort(eigenvalues)[::-1]
        # TODO 5.2: Use it to sort eigenvalues and U
        eigenvalues = eigenvalues[sorting_inds]
        eigenvectors = eigenvectors[:, sorting_inds]
        
        # TODO 6: Select the top L eigenvectors and set A accordingly
        L = self.new_dim
        self.A = eigenvectors[:, :L]
        
        return self
    
    # x_val is (m,n) matrix where each row is an n-dimensional vector of features
    def transform(self, x_val):
        z_val = x_val - self.mean
        # TODO 7: Apply the transformation equation
        return np.dot(z_val, self.A)
    
    def inverse_transform(self, z_val):
        # TODO 8: Apply the inverse transformation equation (including destandardization)
        z_train = np.dot(z_val, self.A.T)
        return z_train + self.mean

    def fit_transform(self, x_train):
        return self.fit(x_train).transform(x_train)