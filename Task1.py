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

eigenenergies, eigenvectors = np.linalg.eigh(H)

#%%
import numpy as np
import matplotlib.pyplot as plt

def construct_hamiltonian(N, epsilon, V):
    """
    Constructs the tridiagonal Hamiltonian matrix for a 1D tight-binding chain.
    """
    H = np.diag(epsilon * np.ones(N)) + np.diag(V * np.ones(N - 1), -1) + np.diag(V * np.ones(N - 1), 1)
    return H


def compute_ldos(H, energy_range, Gamma, sites):
    """
    Computes the LDOS for given sites in the chain using Eq. (8).
    """
    N = H.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(H)  # Solve for eigenvalues and eigenvectors
    
    LDOS = {site: np.zeros(len(energy_range)) for site in sites}
    
    for i, E in enumerate(energy_range):
        for site in sites:
            for lambda_idx in range(N):
                LDOS[site][i] += (Gamma / np.pi) * (eigenvectors[site-1, lambda_idx] ** 2) / ((E - eigenvalues[lambda_idx])**2 + Gamma**2)
    
    return LDOS, eigenvalues

# Define parameters
N = 501  # Chain length
epsilon = 0  # On-site energy
V = -1  # Hopping term
Gamma = 0.05  # Broadening factor
energy_range = np.linspace(-6, 6, 500)  # Energy range for LDOS calculation
sites = [1, 2, 3, 5, 10, 50, 100, 251]  # Sites to compute LDOS

# Construct the Hamiltonian
H = construct_hamiltonian(N, epsilon, V)

# Compute LDOS
LDOS, eigenvalues = compute_ldos(H, energy_range, Gamma, sites)

# Plot LDOS for the selected sites
plt.figure(figsize=(10, 6))
for site in sites:
    plt.plot(energy_range, LDOS[site], label=f"Site {site}")
plt.xlabel("Energy E")
plt.ylabel("LDOS gL_i(E, Γ)")
plt.title("Local Density of States for Selected Sites")
plt.legend()
plt.grid()
plt.show()
