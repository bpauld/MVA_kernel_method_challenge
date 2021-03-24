import numpy as np
import itertools
from scipy import sparse


def RBF_kernel(X_left, X_right, sigma=1, sparse=False, **kwargs):
    #print(sigma)
    if sparse:
        X_left_norm = np.sum(X_left.power(2), axis = -1)
        X_right_norm = np.sum(X_right.power(2), axis = -1)
    else:
        X_left_norm = np.sum(X_left ** 2, axis = -1)
        X_right_norm = np.sum(X_right ** 2, axis = -1)
    
    
    
    K = np.exp(- 1 / (2 * sigma**2) * (X_left_norm[:,None] + X_right_norm[None,:] - 2 * np.dot(X_left, X_right.T)))
    
    return K

def linear_kernel(X_left, X_right, **kwargs):
    K = np.dot(X_left, X_right.T)
    return K

def spectrum_kernel(X_left, X_right, k=3, sub_kernel="RBF", sigma=1, **kwargs):
    spec_ker = spectrum_representation(k=k)
    X_left_new = spec_ker.fit(X_left)
    X_right_new = spec_ker.fit(X_right)
    
    
    
    #print(X_left_new)
    
    if sub_kernel=="RBF":
        return RBF_kernel(X_left_new, X_right_new, sigma=sigma)
    elif sub_kernel=="linear":
        return linear_kernel(X_left_new, X_right_new)
    if sub_kernel=="sparse_RBF":
        A = sparse.csr_matrix(X_left_new)
        B = sparse.csr_matrix(X_right_new)
        return RBF_kernel(A, B, sigma=sigma, sparse=True)
    else:
        print("Unknown sub-kernel")
    

def sum_kernel(X_left, X_right, kernel_list, **kwargs):
    K = np.zeros((X_left.shape[0], X_right.shape[0]))
    for kernel_dic in kernel_list:
        kernel_name = kernel_dic["name"]
        
        if "X_left" not in kernel_dic.keys():
            if kernel_name == "spectrum_kernel":
                sub_kernel = kernel_dic["sub_kernel"]
                k = kernel_dic["k"]
                if sub_kernel == "RBF":
                    K += spectrum_kernel(X_left, X_right, k=k, sub_kernel="RBF", sigma= kernel_dic["sigma"])
                else:
                    K += spectrum_kernel(X_left, X_right, k=k, sub_kernel=sub_kernel)
                    
            elif kernel_name == "RBF":
                K += RBF_kernel(X_left, X_right, sigma=kernel_dic["sigma"])
            elif kernel_name == "linear":
                K += linear_kernel(X_left, X_right, sigma=kernel_dic["sigma"])
            else:
                print("Kernel unknown")
                
        else:
            X_left_other = kernel_dic["X_left"]
            X_right_other = kernel_dic["X_right"]
            if kernel_name == "RBF":
                K += RBF_kernel(X_left_other, X_right_other, sigma=kernel_dic["sigma"])
            elif kernel_name == "linear":
                K += linear_kernel(X_left_other, X_right_other, sigma=kernel_dic["sigma"])
            else:
                print("Kernel unknown")   
                
    return K

class spectrum_representation():
    def __init__(self, k=3, alphabet=['T', 'A', 'C', 'G']):
        self.k = k
        self.alphabet = alphabet
        
        self.dic = {}
        idx = 0
        for item in itertools.product(alphabet, repeat=k):
            self.dic["".join(item)] = idx
            idx += 1
        #print(self.dic)
    
    def fit(self, X_left, X_right=None):
        #X is assumed to be a n x 1 matrix, where each row is a string
        n_left = X_left.shape[0]
        X_left_rep = np.zeros((n_left, len(self.dic)))
        for i in range(n_left):
            X_left_rep[i, :] = self.compute_representation(X_left[i][0])
        if X_right is None:
            X_right_rep = X_left_rep
        else:
            n_right = X_right.shape[0]
            X_right_rep = np.zeros((n_right, len(self.dic)))
            for i in range(n_right):
                X_right_rep[i, :] = self.compute_representation(X_right[i][0])
        #K = np.dot(X_left_rep, X_right_rep.T)
        return X_left_rep
        
            
    def compute_representation(self, s):
        all_substrings = [s[i: j] for i in range(len(s)) for j in range(i + 1, len(s) + 1) if len(s[i:j]) == self.k]
        representation = np.zeros(len(self.dic))
        for substring in all_substrings:
            representation[self.dic[substring]] += 1
        return representation
        
        
        
                 
    