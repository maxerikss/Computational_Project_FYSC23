import numpy as np
import matplotlib.pyplot as plt
import time

def construct_hamiltonian(N, epsilon=0, V=-1, adsorbate=False, epsilon_0=-1, V_0=0):

    size = N + 1 if adsorbate else N
    H = np.zeros((size, size))
    np.fill_diagonal(H, epsilon)  # Set diagonal elements
    np.fill_diagonal(H[1:], V)    # Set lower diagonal elements
    np.fill_diagonal(H[:, 1:], V) # Set upper diagonal elements
    
    if adsorbate:
        H[-1, -1] = epsilon_0  
        H[-1, 0] = V_0  
        H[0, -1] = V_0  
    
    return H

def lorentzian(E, E_lambda, Gamma):
    return (Gamma / np.pi) / ((E - E_lambda)**2 + Gamma**2)

def compute_ldos(H, sites, E_range=(-6,6), Gamma=0.05, num_points=500):
    
    start_time = time.time() # Benchmarking
    
    E_values = np.linspace(E_range[0], E_range[1], num_points)
    eigenvalues, eigenvectors = np.linalg.eigh(H)  # Diagonalize H
    
    ldos = {site: np.zeros_like(E_values) for site in sites}
    
    for site in sites:
        for lambda_idx in range(len(eigenvalues)):
            ldos[site] += (np.abs(eigenvectors[site, lambda_idx])**2) * lorentzian(E_values, eigenvalues[lambda_idx], Gamma)
    
    end_time = time.time()
    print(f"LDOS computation time: {end_time - start_time:.4f} seconds")
    
    return E_values, ldos

def plot_ldos_stacked(E_values, ldos_dict, sites, adsorbate=False):
    num_plots = len(ldos_dict)
    fig, axes = plt.subplots(num_plots, 1, figsize=(8, 9), sharex=True, sharey=True)
    fig.suptitle("Local Density of States for Selected Sites")
    if adsorbate:
        fig.suptitle("LDOS for Adsorbate Cases")
    
    if num_plots == 1:
        axes = [axes]
    
    for idx, ((epsilon_0, V_0), ldos) in enumerate(ldos_dict.items()):
        for site in sites:
            axes[idx].plot(E_values, ldos[site], label=f"Site {site+1}")
        if adsorbate:    
            axes[idx].set_title(f"ε0={epsilon_0}, V0={V_0}")
        axes[idx].set_ylabel("LDOS gL(E, Γ)")
        axes[idx].legend()
        axes[idx].grid()
    
    axes[-1].set_xlabel("Energy (E)")
    plt.tight_layout()
    plt.show()

# Parameters
N = 501
sites = [0, 1, 2, 4, 9, 49, 99, 250]
Gamma = 0.05

# Task 1: Clean Surface
H_clean = construct_hamiltonian(N)
E_values_clean, ldos_clean = compute_ldos(H_clean, sites, Gamma=Gamma)
plot_ldos_stacked(E_values_clean, {("Clean", "Surface"): ldos_clean}, sites)

# Task 2: Surface with Adsorbate
adsorbate_cases = [(-1, 0), (-1, -0.3), (-1, 3.0)]
ldos_results = {}

for epsilon_0, V_0 in adsorbate_cases:
    H_adsorbate = construct_hamiltonian(N, adsorbate=True, epsilon_0=epsilon_0, V_0=V_0)
    E_values_ads, ldos_ads = compute_ldos(H_adsorbate, [0] + sites, Gamma=Gamma)
    ldos_results[(epsilon_0, V_0)] = ldos_ads

plot_ldos_stacked(E_values_ads, ldos_results, [0] + sites, adsorbate=True)
