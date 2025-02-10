import numpy as np
import matplotlib.pyplot as plt

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:28:49 2025

@author: fredrik
"""

N = 501         # Chain length
V = -1          # Hopping term
epsilon = 0     # Energy at site
gamma = 0.05    # Broadening factor
LDOS_sites = [1, 2, 3, 5, 10, 50, 100, 251] # Sites we want to plot
energy_range = np.linspace(-6, 6, 500)  # Energy range for LDOS calculation


def hamiltonian(n, epsilon, V):
    "Create a hamilitonian"
    upper =  np.diag(V * np.ones(n-1), 1)
    middle = np.diag(epsilon * np.ones(n), 0)
    lower = np.diag(V * np.ones(n-1), -1)
    return upper + middle + lower

def sums(gamma, eigenvec, lamb, energy, eigenergy, site):
    "Each term in the sum"
    return (gamma / np.pi) * (eigenvec[site-1, lamb] ** 2) / ((energy - eigenergy[lamb])**2 + gamma**2)
    
# Find the hamltonian
H = hamiltonian(N, epsilon, V)

# Find eigenergies and eigenvectors
eigenenergies, eigenvectors = np.linalg.eigh(H)

# initlaize the LDOS as a dictonary
LDOS = {site: np.zeros(len(energy_range)) for site in LDOS_sites}

# Do the calculation for each site which we are intrested in
for site in LDOS_sites:
    # Make a loop where we look through each position with the energy at that poition 
    for i, E in enumerate(energy_range):
        # Calculate the sum
        for lamb in range(N):
            LDOS[site][i] += sums(gamma=gamma, eigenvec=eigenvectors, 
                                  lamb=lamb, energy=E, 
                                  eigenergy=eigenenergies, site=site)

# Plot LDOS for the selected sites
plt.figure(figsize=(10, 6))
for site in LDOS_sites:
    plt.plot(energy_range, LDOS[site], label=f"i={site}")
plt.xlabel("Energy E")
plt.ylabel(r"$g^L_{i}(E, \Gamma)$")
plt.title("Local Density of States for Selected Sites")
plt.legend()
plt.grid()
plt.show()

plt.savefig("Comp_Proj1/Figures/task1.pdf")
