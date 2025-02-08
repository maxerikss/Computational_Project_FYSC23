import numpy as np
import matplotlib.pyplot as plt
import time 

N = 501         # Chain length
V = -1          # Energy at neigbhour
epsilon = 0     # Energy at site
gamma = 0.05    # Broadening factor
LDOS_sites = [0, 1, 2, 4, 9, 49, 99, 250]
energy_range = np.linspace(-6, 6, 500)  # Energy range for LDOS calculation


def hamiltonian(N, epsilon, V):
    "Create a hamilitonian"
    size = N
    H = np.zeros((size, size))
    np.fill_diagonal(H, epsilon)
    np.fill_diagonal(H[1:], V)
    np.fill_diagonal(H[:, 1:], V)

    return H

def sums(E_values, E_lambda, Gamma):
    return (Gamma / np.pi) / ((E_values - E_lambda) ** 2 + Gamma ** 2)

def compute_LDOS(H, sites, E_range=(-6, 6), Gamma=0.05, num_points=500):
    
    start_time = time.time() # Benchmarking
    
    E_values = np.linspace(E_range[0], E_range[1], num_points)
    eigenvalues, eigenvectors = np.linalg.eigh(H)  
    
    ldos = {site: np.zeros_like(E_values) for site in sites}
    
    for site in sites:
        for lambda_idx in range(len(eigenvalues)):
            ldos[site] += (np.abs(eigenvectors[site, lambda_idx])**2) * sums(E_values, eigenvalues[lambda_idx], Gamma)
    
    end_time = time.time()
    print(f"LDOS computation time: {end_time - start_time:.4f} seconds")
    
    return E_values, ldos

H = hamiltonian(N, epsilon, V)
E_values, LDOS = compute_LDOS(H, LDOS_sites, Gamma=gamma)

# Plot LDOS for the selected sites
plt.figure(figsize=(10, 6))
for site in LDOS_sites:
    plt.plot(energy_range, LDOS[site], label=f"i={site+1}")
plt.xlabel("Energy E")
plt.ylabel(r"$g^L_{i}(E, \Gamma)$")
plt.title("Local Density of States for Selected Sites")
plt.legend()
plt.grid()
plt.show()

#plt.savefig("Comp_Proj1/Figures/task1.pdf")
