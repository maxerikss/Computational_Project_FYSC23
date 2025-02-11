#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:28:49 2025

@author: fredrik
"""

import numpy as np
import matplotlib.pyplot as plt

N = 501         # Chain length
V = -1          # Hopping term
epsilon = 0     # Energy at site
gamma = 0.05    # Broadening factor
LDOS_sites = [1, 2, 3, 5, 10, 50, 100, 251] # Sites we want to plot
energy_range = np.linspace(-6, 6, 1000)  # Energy range for LDOS calculation

def hamiltonian(n, epsilon, V, e0, v0):
    "Create a Hamiltonian"
    upper = np.diag(V * np.ones(n-1), 1)
    middle = np.diag(epsilon * np.ones(n), 0)
    lower = np.diag(V * np.ones(n-1), -1)
    H = upper + middle + lower
    H[0, -1] = v0
    H[-1, 0] = v0
    H[-1, -1] = e0
    return H

def sums(gamma, eigenvec, lamb, energy, eigenergy, site):
    "Each term in the sum"
    return (gamma / np.pi) * (eigenvec[site-1, lamb] ** 2) / ((energy - eigenergy[lamb])**2 + gamma**2)

def compute_LDOS(LDOS_sites, energy_range, H):
    "Function for fining the LDOS"
    # Find eigenenergies and eigenvectors
    eigenenergies, eigenvectors = np.linalg.eigh(H)

    # Initialize LDOS as a dictionary
    LDOS = {site: np.zeros(len(energy_range)) for site in LDOS_sites}
    
    # Compute LDOS for each site of interest
    for site in LDOS_sites:
        for i, E in enumerate(energy_range):
            for lamb in range(N):
                LDOS[site][i] += sums(gamma=gamma, eigenvec=eigenvectors, 
                                      lamb=lamb, energy=E, 
                                      eigenergy=eigenenergies, site=site)
    return LDOS # Return the LDOS

# Different parameter sets (e0, v0)
param_list = [(-1, 0), (-1, -0.3), (-1, -3)]

# Create figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(8, 9))
fig.suptitle("Local Density of States for Selected Sites")

# Loop over different parameter sets and plot in subplots
for idx, (e0, v0) in enumerate(param_list):
    H = hamiltonian(N, epsilon, V, e0, v0)  # Compute Hamiltonian
    LDOS = compute_LDOS(LDOS_sites, energy_range, H)  # Compute LDOS
    
    for site in LDOS_sites:
        axes[idx].plot(energy_range, LDOS[site], label=f"i={site}")
    
    axes[idx].set_title(f"$\epsilon_0={e0}, V_0={v0}$")
    axes[idx].set_ylabel(r"$g^L_{i}(E, \Gamma)$")
    axes[idx].legend()
    axes[idx].grid()

# Set common x-axis
axes[0].set_ylim(0, .8)
axes[1].set_ylim(0, .8)
axes[2].set_ylim(0, 3.2)
axes[2].set_xlabel("Energy E")
plt.tight_layout()

plt.savefig("Comp_Proj1/Figures/task2.pdf")
plt.show()


