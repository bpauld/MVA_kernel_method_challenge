from optimizers import *
from cvxopt import matrix, solvers




class logistic_regression:
    def __init__(self, X_train, y_train, X_val=None, y_val=None, mu=0):
        (n_train, d) = X_train.shape
        self.X_train = X_train
        self.y_train = y_train
        self.n_train = n_train
        self.d = d
        self.w = np.zeros(d)
        self.X_val = X_val
        self.y_val = y_val
        self.mu = mu
    
    def fit(self, optimizer=GD, stepsize=1e-1, max_iter=100, batch_size=32, verbose=True):
        self.w = optimizer(loss=loss_logistic,
                           grad=grad_logistic,
                           X=self.X_train,
                           y=self.y_train,
                           mu=self.mu,
                           stepsize=stepsize,
                           max_iter=max_iter,
                           batch_size=batch_size,
                           verbose=verbose,
                           X_val=self.X_val,
                           y_val=self.y_val)
    
    
    def predict(self, X_test):
        pred = np.sign(np.dot(X_test, self.w))
        pred[pred==-1] = 0
        return pred
    
    
class kernel_logistic_regression:
    def __init__(self, K, y_train, K_val=None, y_val=None, mu=0):
        n_train = K.shape[0]
        self.K = K
        self.y_train = y_train
        self.n_train = n_train
        self.alpha = np.zeros(n_train)
        self.K_val = K_val
        self.y_val = y_val
        self.mu = mu
        
    def fit(self, optimizer=GD_kernel, stepsize=1e-1, max_iter=100, batch_size=32, verbose=True):
        print("Logistic regression...")
        self.alpha = optimizer(loss=loss_logistic_kernel,
                           grad=grad_logistic_kernel,
                           K=self.K,
                           y=self.y_train,
                           mu=self.mu,
                           stepsize=stepsize,
                           max_iter=max_iter,
                           batch_size=batch_size,
                           verbose=verbose,
                           K_val=self.K_val,
                           y_val=self.y_val)
        
    def val_accuracy(self):
        if self.K_val is not None:
            pred = np.sign(np.dot(self.K_val , self.alpha)).astype(int)
        
            accuracy = 0
            for i in range(len(self.y_val)):
                if pred[i] == self.y_val[i]:
                    accuracy += 1
            return accuracy / len(self.y_val)
        
    def train_accuracy(self):
        pred = np.sign(np.dot(self.K , self.alpha)).astype(int)
        
        accuracy = 0
        for i in range(len(self.y_train)):
            if pred[i] == self.y_train[i]:
                accuracy += 1
        return accuracy / len(self.y_train)
        
    def predict(self, K_test):
        pred = np.sign(np.dot(K_test , self.alpha)).astype(int)
        pred[pred==-1] = 0
        return pred
    
    
    
    
class SVM:
    def __init__(self, K, y_train, K_val=None, y_val=None, mu=0):
        n_train = K.shape[0]
        self.K = K
        self.y_train = y_train
        self.n_train = n_train
        self.alpha = np.zeros(n_train)
        self.K_val = K_val
        self.y_val = y_val
        self.mu = mu
        
    def fit(self, loss="hinge", show_progress=False, **kwargs):
        solvers.options['show_progress'] = show_progress
        if loss=="hinge":
            #print("SVM hinge loss...")
            Q = 2*self.K
            p = - 2 * self.y_train.astype('float')
            G1 = np.diag(self.y_train).astype('float')
            G = np.zeros((2* self.n_train, self.n_train))
            G[0:self.n_train,:] = G1
            G[self.n_train:, :] = -G1
            h = np.zeros(2 * self.n_train).astype('float')
            if self.mu != 0:
                h[0:self.n_train] = 1 / (2 * self.mu * self.n_train)
            else:
                h[0:self.n_train] = np.inf


            # cast for solver
            Q = matrix(Q)
            p = matrix(p)
            G = matrix(G)
            h = matrix(h)

            #get solution
            solver = solvers.qp
            sol = solver(Q, p, G, h)
            #print(np.array(sol['x']))
            self.alpha = np.array(sol['x'])
            return np.array(sol['x'])
    
        elif loss=="squared_hinge":
            #print("SVM squared hinge loss...")
            Q = 2* (self.K + self.mu * self.n_train * np.eye(self.n_train))
            p = - 2 * self.y_train.astype('float')
            G = -np.diag(self.y_train).astype('float')
            h = np.zeros(self.n_train).astype('float')
            
            # cast for solver
            Q = matrix(Q)
            p = matrix(p)
            G = matrix(G)
            h = matrix(h)
            
            #get solution
            sol = solvers.qp(Q, p, G, h)
            self.alpha = np.array(sol['x'])
            return np.array(sol['x'])
            
            
    def val_accuracy(self):
        if self.K_val is not None:
            pred = np.sign(np.dot(self.K_val , self.alpha)).astype(int)
        
            accuracy = 0
            for i in range(len(self.y_val)):
                if pred[i] == self.y_val[i]:
                    accuracy += 1
            return accuracy / len(self.y_val)
        
    def train_accuracy(self):
        pred = np.sign(np.dot(self.K , self.alpha)).astype(int)
        
        accuracy = 0
        for i in range(len(self.y_train)):
            if pred[i] == self.y_train[i]:
                accuracy += 1
        return accuracy / len(self.y_train)
    
    def predict(self, K_test):
        pred = np.sign(np.dot(K_test , self.alpha)).astype(int)
        pred[pred==-1] = 0
        return pred
    
    
class Kernel_PCA:
    def __init__(self, K_train):
        self.K_train = K_train
        self.n_train = self.K_train.shape[0]
        self.principal_components = None
    
    def fit(self, tol=1e-5):
        eig_vectors, eig_values, vh = np.linalg.svd(self.K_train, hermitian=True)

        #sort by eigenvalues, need to keep track of corresponding eigenvectors
        A = np.zeros((self.n_train , self.n_train+1))
        A[:, 0:self.n_train] = eig_vectors.T
        A[ :,self.n_train] = eig_values
        A = A[A[:,self.n_train].argsort()[::-1]]
        
        new_eig_vectors = A[:, 0:self.n_train].T
        new_eig_values = A[:, self.n_train]
        
        thresh = np.where(new_eig_values < tol)[0]
        if thresh.size == 0:
            thresh = self.n_train
        else:
            thresh = thresh[0]
        eig_values = new_eig_values[:thresh]
        eig_vectors = new_eig_vectors[:, 0:thresh]
        
        principal_components = np.zeros((self.n_train, thresh))
        for i in range(thresh):
            principal_components[:, i] = eig_vectors[:, i]/ np.sqrt(eig_values[i])
            
        self.principal_components = principal_components
    
    def project(self, K):
        return np.dot(K, self.principal_components)
        
        
        
