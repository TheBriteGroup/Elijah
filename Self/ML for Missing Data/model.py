import numpy as np

class KNNImputer:
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X):
        self.X_train = X
    
    def impute(self, X_missing):
        X_imputed = X_missing.copy()
        
        for i in range(X_missing.shape[0]):
            for j in range(X_missing.shape[1]):
                if np.isnan(X_missing[i, j]):
                    distances = self.calculate_distances(X_missing[i], self.X_train)
                    nearest_indices = np.argsort(distances)[:self.k]
                    nearest_values = self.X_train[nearest_indices, j]
                    imputed_value = np.mean(nearest_values)
                    X_imputed[i, j] = imputed_value
        
        return X_imputed
    
    def calculate_distances(self, x, X):
        return np.sqrt(np.sum((X - x)**2, axis=1))