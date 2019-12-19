import numpy as np 
from sklearn.feature_selection import SelectKBest,f_regression
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA


def load_data(n_samples, tot_data_x, tot_data_y):    
    data_len = tot_data_x.shape[0]
    mask_data = np.random.permutation(data_len)[:n_samples]

    data_X = tot_data_x[mask_data]
    data_y = tot_data_y[mask_data]
    return data_X, data_y

def load_data_train_test(n_samples, tot_data_x, tot_data_y, iqr=False):
    data_X, data_y = load_data(n_samples, tot_data_x, tot_data_y, iqr=iqr)
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size = 0.2)
    return X_train, X_test, y_train, y_test


def plot_PCA(n_comp, X_train):
    """
    displays the 'elbow' of the PCA, ie the screeplot"""
    pca = PCA(n_components = n_comp)
    pca.fit(X_train)
    plt.subplots(figsize=(10,8))
    plt.figure(1)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()

    
def univ_feat_sel(X,y,keep = 0.2,ax = None):
    """
    Plot feature selection using the f_regression from skleanr: 
    it computes an F score based on the correlation of the feature and 
    compute a pvalue based on that.
    X: data
    y: label
    keep: ratio of initial features you want to keep 
    """
    k = int(keep * X.shape[1])
    model = SelectKBest(score_func=f_regression, k=k)
    fit = model.fit(X, y)
    feature_ord_univ = np.argsort(fit.scores_)
    if ax is None:
        plt.bar(feature_ord_univ,fit.scores_)
    else:
        ax.bar(feature_ord_univ,fit.scores_)
    
    
def do_PCA(X_train, X_test, n):
    """
    Useful method to reduce train/test sets outside of the pipeline
    """
    pca = PCA(n_components=n)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, X_test

def train_model(model, train_input, train_target, mini_batch_size, monitor_loss=False):
    '''Train the model using Mini-batch SGD'''
    
    criterion = nn.MSELoss() #regression task
    optimizer = optim.Adam(model.parameters(), lr = 1e-4) #1e-4 normalement
    nb_epochs = 10
    
    # Monitor loss
    losses = []
    
    for e in range(nb_epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            
            sum_loss += loss.item() #compute loss for each mini batch for 1 epoch
            
            optimizer.step()
        
        # Monitor loss
        losses.append(sum_loss)
        
        print('[epoch {:d}] loss: {:0.2f}'.format(e+1, sum_loss))
    
    if monitor_loss:
        return losses

    

def compute_pred(model, data_input):
    '''Given a trained model, output the prediction corresponding to data_input'''
    y_hat = model(data_input)
    return y_hat

def compute_score(y_actual, y_pred):
    mse = mean_squared_error(y_actual, y_pred)
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)

    return mse, mae, r2

def load_process_data(n, iqr=False, pca=False, scale=False):
    # load all datas
    # filter (iqr)
    if iqr:
        X_train, X_test, y_train, y_test = load_data_train_test(n, X_tot, y_tot, iqr=apply_iqr)
    else:
        X_train, X_test, y_train, y_test = load_data_train_test(n, X_tot, y_tot)
        
    if scale:
        min_max = MinMaxScaler()
        X_train, X_test = apply_scaler(min_max, X_train, X_test)
        
    if pca:
        n = 80
        X_train, X_test = do_PCA(X_train, X_test, n)
        nb_input_neurons = n

    return X_train, X_test, y_train, y_test

