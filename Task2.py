import numpy as np
import matplotlib.pyplot as plt
import time 
N = 501         # Chain length
V = -1          # Energy at neigbhour
epsilon = 0     # Energy at site
gamma = 0.05    # Broadening factor
LDOS_sites = [0, 1, 2, 4, 9, 49, 99, 250]
energy_range = np.linspace(-6, 6, 500)  # Energy range for LDOS calculation

def hamiltonian(N, epsilon, V, epsilon_0=-1, V_0=0):
    "Create a hamilitonian"
    size = N
    H = np.zeros((size, size))
    np.fill_diagonal(H, epsilon)
    np.fill_diagonal(H[1:], V)
    np.fill_diagonal(H[:, 1:], V)
    
    H[-1, -1] = epsilon_0  
    H[-1, 0] = V_0  
    H[0, -1] = V_0  
    
    return H

def sums(E, E_lambda, Gamma):
    return (Gamma / np.pi) / ((E - E_lambda)**2 + Gamma**2)

def compute_LDOS(H, sites, E_range=(-6, 6), Gamma=0.05, num_points=500):
    
    start_time = time.time()  # Benchmarking
    
    E_values = np.linspace(E_range[0], E_range[1], num_points)
    eigenvalues, eigenvectors = np.linalg.eigh(H)  
    
    ldos = {site: np.zeros(num_points) for site in sites} 
    
    for site in sites:
        for lambda_idx in range(len(eigenvalues)):
            ldos[site] += (np.abs(eigenvectors[site, lambda_idx])**2) * sums(E_values, eigenvalues[lambda_idx], Gamma)
    
    end_time = time.time()
    print(f"LDOS computation time: {end_time - start_time:.4f} seconds")
    
    return E_values, ldos


# Different parameter sets (e0, v0)
param_list = [(-1, 0), (-1, -0.3), (-1, -3)]

# Create figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharey=True)
fig.suptitle("Local Density of States for Selected Sites")

# Loop over different parameter sets and plot in subplots
for idx, (e0, v0) in enumerate(param_list):
    H = hamiltonian(N, epsilon, V, e0, v0)  # Compute Hamiltonian
    E_values, LDOS = compute_LDOS(H, LDOS_sites, E_range=(-6, 6), Gamma=0.05, num_points=500)
    
    for site in LDOS_sites:
        axes[idx].plot(energy_range, LDOS[site], label=f"i={site+1}")
    
    axes[idx].set_title(f"$\\epsilon_0={e0}, V_0={v0}$")
    axes[idx].set_ylabel(r"$g^L_{i}(E, \Gamma)$")
    axes[idx].legend()
    axes[idx].grid()

# Set common x-axis
axes[2].set_xlabel("Energy E")
plt.tight_layout()
plt.show()


plt.savefig("Comp_Proj1/Figures/task2.pdf")

