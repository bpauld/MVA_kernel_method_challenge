import numpy as np
import pandas as pd
from kernels import *
from models import Kernel_PCA

def get_training_data(label = 0, datadir = "./data/", suffix="_mat100"):
    if label not in [0, 1, 2]:
        print("Label unknown")
        return
    if suffix not in ["", "_mat100"]:
        print("Suffix unknown")
        return
    
    #form full name of file
    Xfile_path = datadir + "Xtr" + str(label) + suffix + ".csv"
    Yfile_path = datadir + "Ytr" + str(label) + ".csv"
    
    #get training data
    if suffix == "_mat100":
        data = pd.read_csv(Xfile_path, sep=" ", header=None)
        X = data.to_numpy()
    else:
        data = pd.read_csv(Xfile_path, sep=",", header=None)
        X = data.to_numpy()
        X = X[1:, 1]
        X = X.reshape(len(X), 1)
    
    #get training labels
    labels = pd.read_csv(Yfile_path, sep=",")
    y = labels.to_numpy()   
    y = y[:, 1]
    y[y==0] = -1
    
    return X, y

def get_test_data(label = 0, datadir = "./data/", suffix="_mat100"):
    if label not in [0, 1, 2]:
        print("Label unknown")
        return
    if suffix not in ["", "_mat100"]:
        print("Suffix unknown")
        return
    
    #form full name of file
    Xfile_path = datadir + "Xte" + str(label) + suffix + ".csv"
    
    #get test data
    if suffix == "_mat100":
        data = pd.read_csv(Xfile_path, sep=" ", header=None)
        X = data.to_numpy()
    else:
        data = pd.read_csv(Xfile_path, sep=",", header=None)
        X = data.to_numpy()
        X = X[1:, 1]
        X = X.reshape(len(X), 1)
    
    return X

def format_predictions(pred_list, result_path="./test.csv", required_dim=1000):
    if len(pred_list) != 3:
        print("Less than 3 prediction vectors...")
    for pred in pred_list:
        if pred.shape[0] != required_dim:
            print("Problem with dimensions of the predictions")
            return
    
    final_pred = np.zeros(len(pred_list) * required_dim)
    for i in range(len(pred_list)):
        final_pred[i*len(pred_list[i]):(i+1)*len(pred_list[i])] = pred_list[i].astype(int)
        
    
    #format into csv file
    df = pd.DataFrame(final_pred).astype(int)
    df.to_csv(result_path, sep=",", header=["Bound"], index_label="Id")
    
def train_test_split(X, y, train_ratio=0.8, shuffle=False):
    n = y.shape[0]
    Z = np.zeros((n, X.shape[1] + 1))
    Z[:,:X.shape[1]] = X
    Z[:, X.shape[1]] = y
    
    if shuffle:
        np.random.shuffle(Z)
    
    
    end_train = int(train_ratio * n)
    X_train = Z[:end_train, :Z.shape[1] - 1]
    y_train = Z[:end_train, Z.shape[1] - 1]
    X_val = Z[end_train:, :Z.shape[1] - 1]
    y_val = Z[end_train:, Z.shape[1] - 1]
    return X_train, y_train, X_val, y_val

def cross_val_score(estimator, X, y, kernel,
                    mu=0, cv=3, score="accuracy", shuffle=True, PCA=False, PCA_tol=1e-5,
                    **kwargs):
    n = y.shape[0]
    Z = np.c_[X, y]
    
    if shuffle:
        np.random.shuffle(Z)
    X_copy = Z[:,:Z.shape[1] - 1]
    y_copy = Z[:,Z.shape[1] - 1]
    
    size_val_set = int(n / cv)
    score_list = []
    for i in range(cv):
        val_indices = np.arange(i*size_val_set, (i+1)*size_val_set, 1)
        train_indices = np.append(np.arange(0, i*size_val_set, 1), np.arange( (i+1)*size_val_set, n, 1))
        
        X_train = X_copy[train_indices, :]
        y_train = y_copy[train_indices]
        X_val = X_copy[val_indices, :]
        y_val = y_copy[val_indices]
        
        if PCA:
            #center the data
            train_mean = np.mean(X_train, axis=0)
            X_train = X_train - train_mean
            X_val = X_val - train_mean
            K_train = linear_kernel(X_train, X_train)
            K_val = linear_kernel(X_val, X_train)
            PCA_model = Kernel_PCA(K_train)
            PCA_model.fit(tol=PCA_tol)
            X_train = PCA_model.project(K_train)
            X_val = PCA_model.project(K_val)
        
        
        #print(X_train)
        K_train = kernel(X_train, X_train, **kwargs)
        K_val = kernel(X_val, X_train, **kwargs)
        #print(K_train)
        model = estimator(K=K_train, y_train=y_train, K_val=K_val, y_val=y_val, mu=mu)
        
        model.fit(**kwargs)
        
        if score=="accuracy":
            score_list.append(model.val_accuracy())
            
    return score_list
            
        
        
        
        
        
    
    