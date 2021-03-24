import numpy as np


def loss_logistic(w, X, y, mu=0):
    (n , d) = X.shape
    return 1/n * np.sum(np.log(1 + np.exp(-y * np.dot(X, w)))) + mu/2 * np.dot(w, w)

def grad_logistic(w, X, y, mu=0):
    (n, d) = X.shape
    yXw = y * np.dot(X, w)
    return 1/n * np.dot(np.transpose(X), -y * np.exp(-yXw) / (1 + np.exp(-yXw))) + mu * w

def loss_logistic_kernel(alpha, K, y, mu=0, val=False):
    n = K.shape[0]
    Kalpha = np.dot(K, alpha)
    if val:
        return 1/n * np.sum(np.log(1 + np.exp(-y * Kalpha)))
    else:
        return 1/n * np.sum(np.log(1 + np.exp(-y * Kalpha))) + mu/2 * np.dot(alpha, Kalpha)

def grad_logistic_kernel(alpha, K, y, mu=0):
    n = K.shape[0]
    yKalpha = y * np.dot(K, alpha)
    return 1/n * np.dot(K, -y * np.exp(-yKalpha) / (1 + np.exp(-yKalpha))) + mu * np.dot(K, alpha)

def accuracy(w, X, y):
    pred = np.sign(np.dot(X, w)).astype(int)
    #pred[pred==-1] = 0
    accuracy = 0
    for i in range(len(y)):
        if pred[i] == y[i]:
            accuracy += 1
    return accuracy / len(y)

def accuracy_kernel(alpha, K, y):
    pred = np.sign(np.dot(K , alpha)).astype(int)
    #pred[pred==-1] = 0
    accuracy = 0
    for i in range(len(y)):
        if pred[i] == y[i]:
            accuracy += 1
    return accuracy / len(y)



def GD(loss, grad, X, y, mu=0, w_init=None, stepsize=1e-1, max_iter=100, verbose=True, X_val=None, y_val=None, batch_size=32):
    (n, d) = X.shape
    
    if w_init is None:
        w = np.zeros(d)
    else:
        w = w_init
    for i in range(max_iter):
        gradient = grad(w, X, y, mu=mu)
        w = w - stepsize * gradient
        if verbose:
            output  = ""
            output += "Epoch: {}. ".format(i)
            output += "Train loss: {}. ".format(loss(w, X, y, mu=mu))
            output += "Train grad norm.:{}. ".format(np.linalg.norm(gradient))
            output += "Train Acc.: {} ".format(accuracy(w, X, y))
            if X_val is not None:
                output += "Val. loss: {}. ".format(loss(w, X_val, y_val))
                output += "Val. Acc.: {}. ".format(accuracy(w, X_val, y_val))
            print(output)
                
    return w

def GD_kernel(loss, grad, K, y,
              mu=0, alpha_init=None, stepsize=1e-1, max_iter=100,
              verbose=True, K_val=None, y_val=None, batch_size=32):
    n = K.shape[0]
    
    if alpha_init is None:
        alpha = np.zeros(n)
    else:
        alpha = alpha_init
    for i in range(max_iter):
        #print(alpha)
        gradient = grad(alpha, K, y, mu=mu)
        alpha = alpha - stepsize * gradient
        if verbose:
            output  = ""
            output += "Epoch: {}. ".format(i)
            output += "Train loss: {}. ".format(loss(alpha, K, y, mu=mu))
            output += "Train grad norm.:{}. ".format(np.linalg.norm(gradient))
            output += "Train Acc.: {} ".format(accuracy_kernel(alpha, K, y))
            if K_val is not None:
                output += "Val. loss: {}. ".format(loss(alpha, K_val, y_val, val=True))
                output += "Val. Acc.: {}. ".format(accuracy_kernel(alpha, K_val, y_val))
            print(output)
                
    return alpha

