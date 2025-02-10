#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:29:10 2025

@author: fredrik
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import time  # Import the time module to measure execution time


Nl = 3        # Lattice size along one dimension
Ns = Nl**2     
Nh = (Nl + 1) // 2
Nhp = (Nl - 1) // 2
energy_range = np.linspace(-8, 8, 1000)
V = -1         # Hopping parameter
epsilon = 0    # On-site energy
gamma = 0.05   # Broadening factor


def coord_to_index(i, j, Nhp=Nhp, Nl=Nl):
    """ Convert (i', j') coordinates to matrix index m. """
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
    H = csr_matrix((Ns, Ns)).tolil()
    H.setdiag([epsilon] * (Ns+1))    
    for i in range(-Nhp, Nhp + 1):
        for j in range(-Nhp, Nhp + 1):
            m = coord_to_index(i, j)
            for ni, nj in get_neighbors(i, j):
                n = coord_to_index(ni, nj)
                H[m, n] = V  # Nearest-neighbor hopping
    return H.tocsr()


def hamiltonian_adsorbate(Ns, epsilon, V, adsorbate_type, epsilon_0, V_0, Nhp=Nhp, Nl=Nl):
    """ Construct the Hamiltonian matrix including adsorbate effects """
    H = csr_matrix((Ns+1, Ns+1)).tolil()
    H.setdiag([epsilon] * (Ns+1))    
    
    # Fill the original surface Hamiltonian
    for i in range(-Nhp, Nhp + 1):
        for j in range(-Nhp, Nhp + 1):
            m = coord_to_index(i, j, Nhp, Nl)
            for ni, nj in get_neighbors(i, j):
                n = coord_to_index(ni, nj, Nhp, Nl)
                H[n, m] = V  # Nearest-neighbor hopping
    
    # Define adsorbate interaction
    adsorbate = Ns  # The additional adsorbate index
    H[adsorbate, adsorbate] = epsilon_0  # On-site energy of adsorbate
    
    if adsorbate_type == "atop":
        m = coord_to_index(0, 0)
        H[m, adsorbate] = V_0
        H[adsorbate, m] = V_0
    
    elif adsorbate_type == "bridge":
        m1 = coord_to_index(0, 0)
        m2 = coord_to_index(1, 0)
        H[m1, adsorbate] = V_0 / np.sqrt(2)
        H[m2, adsorbate] = V_0 / np.sqrt(2)
        H[adsorbate, m1] = V_0 / np.sqrt(2)
        H[adsorbate, m2] = V_0 / np.sqrt(2)
    
    elif adsorbate_type == "center":
        sites = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for i, j in sites:
            m = coord_to_index(i, j, Nhp, Nl)
            H[m, adsorbate] = V_0 / 2
            H[adsorbate, m] = V_0 / 2
    
    elif adsorbate_type == "impurity":
        m = coord_to_index(0, 0, Nhp, Nl)
        H[m, m] = epsilon_0  # Replace surface atom energy
        for ni, nj in get_neighbors(0, 0):
            n = coord_to_index(ni, nj, Nhp, Nl)
            H[m, n] = V_0
            H[n, m] = V_0
    
    return H.tocsr()

def sums(gamma, eigenvecs, lamb, energy, eigvals, site):
    "Compute each term in the LDOS sum."
    return (gamma/np.pi) * (np.abs(eigenvecs[site,lamb])**2) / ((energy-eigvals[lamb])**2 + gamma**2)


def compute_LDOS(H, energy_range, Ns=Ns, LDOS_site = (0,0)):
    "Compute the Local Density of States (LDOS)."
    
    # convert to dense array
    H = H.toarray()   

    # Find eigenenergies and eigenvectors
    eigenenergy, eigenvectors = np.linalg.eigh(H)  

    # Initialize LDOS as a dictionary
    LDOS = np.zeros(len(energy_range))
    
    # Compute LDOS 
    site_index = coord_to_index(LDOS_site[0], LDOS_site[1])

    for i, E in enumerate(energy_range):
            for lamb in range(Ns):
                LDOS[i] += sums(gamma=gamma, eigenvecs=eigenvectors, 
                                      lamb=lamb, energy=E, 
                                      eigvals=eigenenergy, site=site_index)
    return LDOS


# Clean surface LDOS
clean = compute_LDOS(hamiltonian(Ns, epsilon, V), 
                     energy_range)  

# bridge
H = hamiltonian_adsorbate(Ns, epsilon, V, "bridge", -2, -1.3)
bridge = compute_LDOS(H, energy_range)


plt.figure(figsize=(10, 5))
plt.title("Local Density of States With Adsorbates for Site (0,0)")

plt.plot(energy_range, clean, label="clean", c='black')
plt.plot(energy_range, bridge, label="bridge", c='red')

plt.ylabel(r"$g^L_{i}(E, \Gamma)$")
plt.ylim(0,4)
plt.legend()
plt.grid()
plt.xlabel("Energy E")
plt.tight_layout()

plt.show()
 
print(hamiltonian_adsorbate(Ns, epsilon, V, "bridge", -2, -1.3*np.sqrt(2)).toarray())


