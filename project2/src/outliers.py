import numpy as np 
from sklearn.model_selection import cross_val_score


def IQR_y_outliers(X_data, y_data):
    ''' aims at removing all rows whose label (i.e. shielding) is considered as outlier.
        output:
     - X_filtered
     - y_filtered
    '''
    q1, q3 = np.percentile(y_data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    
    assert(q1 != q3), 'Q1 and Q3 have the same value'
    
    idx = np.where((y_data > lower_bound) & (y_data < upper_bound))
    X_filtered = X_data[idx]
    y_filtered = y_data[idx]
    
    assert(X_filtered.shape[0] == y_filtered.shape[0])
    
    return X_filtered, y_filtered

class IQR_outlier():
    def __init__(self, l_qtile= 5, h_qtile=95):
        self.l_qtile = l_qtile
        self.h_qtile = h_qtile
        
    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """
        if (hasattr(self,'feat_qutiles')):
            del self.feat_qutiles
        
    def IQR(self,ys):
        """Compute the quartiles for a feature passed in argument
        """
        if self.l_qtile is None or self.h_qtile is None:
            raise ValueError("Quantiles not initialized")
            
        q1, q3 = np.percentile(ys, [self.l_qtile,self.h_qtile])
        iqr = q3 - q1
        lower_bound = q1 - (iqr * 1.5)
        upper_bound = q3 + (iqr * 1.5)
        return np.array([lower_bound,upper_bound])

    
    def fit(self, X, y=None):
        """ compute the quartiles used to remove outliers later on
        Parameters
        ----------
        X : {array-like}, shape [n_samples, n_features]
            The data used to compute the different features to erase
        """
        # Reset internal state before fitting
        self._reset()
        self.feat_qutiles = np.array([self.IQR(feat) for feat in X.T])
        return self

            
    def transform(self, X,y):
        """Perform standardization by centering and scaling
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data we want to take out the outliers of
        """
        if self.feat_qutiles is None:
            raise ValueError("Data not fitted yet.")
            
        masks_indces = []
        for feat,bounds in zip(X.T,self.feat_qutiles):
            masks_indces.append(np.where((feat < bounds[0]) | (feat > bounds[1])))
            
        #hstack reducs everything in one dimension
        mask_final = np.hstack(masks_indces)
        X_trans = np.delete(X,mask_final,axis = 0)
        y_trans = np.delete(y,mask_final,axis = 0)
        
        return X_trans,y_trans
    
    def fit_transform(self,X,y):
        """fit X and transform X and y accordingly"""
        self.fit(X)
        return self.transform(X,y)
