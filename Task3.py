import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve, splu
import matplotlib.pyplot as plt

# Parameters
Nl = 81        # Nr of atoms along one side 
Ns = Nl**2     # Total number of atoms 
Nh = (Nl + 1) // 2
Nhp = (Nl - 1) // 2

V0 = -1        # Hopping parameter
epsilon = 0    # On-site energy
Gamma = 0.05   # Small imaginary part for broadening


# Define conversion from (i', j') to matrix index m
def coords_to_index(i, j):
    ibar = i+Nhp
    jbar = j+Nhp
    return Nl*(ibar-1) + jbar

# Define nearest neighbors
def get_neighbors(i, j):
    neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
    valid_neighbors = [(ni, nj) for ni, nj in neighbors if -Nhp <= ni <= Nhp and -Nhp <= nj <= Nhp]
    return valid_neighbors

# Construct Hamiltonian matrix H
H = csr_matrix((Ns, Ns), dtype=complex).tolil()

for i in range(-Nhp, Nhp + 1):
    for j in range(-Nhp, Nhp + 1):
        m = coords_to_index(i, j)
        H[m, m] = epsilon  # On-site energy
        for ni, nj in get_neighbors(i, j):
            n = coords_to_index(ni, nj)
            H[m, n] = V0  # Nearest neighbor hopping

# Convert to CSR format for fast operations
H_sparse = H.tocsr()

# Define sites of interest
sites = [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (2, 2)]

# Energy range for LDOS plot
E_values = np.linspace(-8, 8, 200)
ldos_results = {site: [] for site in sites}

# Precompute LU decomposition for fast solving
identity_sparse = eye(Ns, format='csr')

for E in E_values:
    A = (E * identity_sparse - H_sparse + 1j * Gamma * identity_sparse).tocsc()
    lu_solver = splu(A)  # Compute LU decomposition
    
    for site in sites:
        rhs = np.zeros(Ns, dtype=complex)
        m = coords_to_index(site[0], site[1])
        rhs[m] = 1  # Delta function at site m
        solution = lu_solver.solve(rhs)
        ldos_results[site].append(-np.imag(solution[m]) / np.pi)  # Include -1/pi factor

# Plot LDOS
plt.figure(figsize=(8,6))
for site in sites:
    plt.plot(E_values, ldos_results[site], label=f"Site {site}")
plt.xlabel("Energy E")
plt.ylabel("LDOS gL(E, Γ)")
plt.title("Local Density of States (LDOS) vs Energy")
plt.legend()
plt.grid()
plt.show()
