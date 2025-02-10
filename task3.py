#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:51:38 2025

@author: fredrik
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import time  

start_time = time.time()


Nl = 81        # Lattice size along one dimension
Ns = Nl**2     # Total number of atoms
V = -1         # Hopping parameter
epsilon = 0    # On-site energy
gamma = 0.05   # Broadening factor

Nh = (Nl + 1) // 2
Nhp = (Nl - 1) // 2

# Sites of interest for LDOS calculation
LDOS_sites = [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (2, 2)]
energy_range = np.linspace(-8, 8, 200)


def coord_to_index(i, j):
    "Convert (i', j') coordinates to matrix index m."
    "This was done by the method provided"
    ibar = i + Nhp 
    jbar = j + Nhp
    return Nl*ibar + jbar


def get_neighbors(i, j):
    "This function check if there are any neighbors nearby that exsists"
    # List all possible nearest neighbors (up, down, left, right)
    neighbors = [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]   
    
    valid_neighbors = []

    # Check each potential neighbor to ensure it is within the valid range
    for ni, nj in neighbors:
        if -Nhp <= ni <= Nhp and -Nhp <= nj <= Nhp:
            valid_neighbors.append((ni, nj))  

    return valid_neighbors # Return the list of valid nearest neighbors


def hamiltonian(Ns, epsilon, V):
    "Construct the Hamiltonian matrix for a 2D square lattice."
    H = csr_matrix((Ns, Ns), dtype=complex).tolil()
    H.setdiag([epsilon] * (Ns+1))    
    for i in range(-Nhp, Nhp + 1):
        for j in range(-Nhp, Nhp + 1):
            m = coord_to_index(i, j)
            for ni, nj in get_neighbors(i, j):
                n = coord_to_index(ni, nj)
                H[m, n] = V  # Nearest-neighbor hopping
    return H.tocsr()


def sums(gamma, eigenvecs, lamb, energy, eigvals, site):
    "Compute each term in the LDOS sum."
    return (gamma / np.pi) * (np.abs(eigenvecs[site, lamb]) ** 2) / ((energy - eigvals[lamb])**2 + gamma**2)


def compute_LDOS(LDOS_sites, energy_range, H):
    "Compute the Local Density of States (LDOS)."

    # Find eigenenergies and eigenvectors
    eigenenergy, eigenvectors = np.linalg.eigh(H.toarray()) # Convert sparse matrix to dense calculation
    
    # Initialize LDOS as a dictionary
    LDOS = {site: np.zeros(len(energy_range)) for site in LDOS_sites}
    
    # Compute LDOS for each site of interest
    for site in LDOS_sites:
        site_index = coord_to_index(site[0], site[1])
        for i, E in enumerate(energy_range):
            for lamb in range(Ns):
                LDOS[site][i] += sums(gamma=gamma, eigenvecs=eigenvectors, 
                                      lamb=lamb, energy=E, 
                                      eigvals=eigenenergy, site=site_index)
    return LDOS 

# Compute Hamiltonian and LDOS
H = hamiltonian(Ns, epsilon, V)
LDOS = compute_LDOS(LDOS_sites, energy_range, H)

# Plot LDOS
plt.figure(figsize=(8, 6))
for site in LDOS_sites:
    plt.plot(energy_range, LDOS[site], label=f"Site {site}")
plt.xlabel("Energy E")
plt.ylabel(r"$g^L_{i}(E, \Gamma)$")
plt.title(f"Local Density of States for Selected Sites with $\epsilon={epsilon}, V={V}$")
plt.legend()
plt.grid()

plt.savefig("Comp_Proj1/Figures/task3.pdf")
plt.show()


end_time = time.time()
print(f"Total time taken: {np.round((end_time - start_time)/60,1)} minutes")
