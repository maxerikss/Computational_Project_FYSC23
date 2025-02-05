import numpy as np
import matplotlib.pyplot as plt

N = 501
V = -1
epsilon = 0
gamma = 0.05
LDOS_sites = [1, 2, 3 , 5, 10, 50, 100, 251] 

def H_matrix(n, e, V):
    "Create a hamilitonian"
    return np.diag(V * np.ones(n-1), -1) + np.diag(e * np.ones(n), 0) + np.diag(V * np.ones(n-1), 1)
    
H = H_matrix(N, epsilon, V)

print(H)