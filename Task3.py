import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import gmres

# Parameters 
Nl = 81         # Number of atoms along one side of the surface (Nl = 81)
Ns = Nl**2      # Total number of atoms on the surface
V0 = -1         # Hopping parameter
epsilon = 0     # On-site energy
Gamma = 0.05    # Small imaginary part for broadening
E = 0           # Energy at which we calculate LDOS

# Intiliaze the Hamiltonian
H = np.zeros((Ns, Ns))

# Function to convert (i, j) to matrix index m
def coords_to_index(i, j, Nl, Np):
    return Nl * (i + Np) + (j + Np)

# Nearest-neighbor coordinates
neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Constructing the Hamiltonian matrix H for the clean surface
N_p = round((Nl - 1) / 2)  # The shift for i, j to the 1D matrix index
for i in range(-N_p, N_p + 1):
    for j in range(-N_p, N_p + 1):
        m = coords_to_index(i, j, Nl, N_p)
        # For each neighbor
        for di, dj in neighbors:
            ni, nj = i + di, j + dj
            if -N_p <= ni <= N_p and -N_p <= nj <= N_p:  # Check if neighbor is within bounds
                n = coords_to_index(ni, nj, Nl, N_p)
                # Set the hopping terms (V0)
                H[m, n] = V0

# Convert the Hamiltonian matrix to a sparse format
H = csr_matrix(H)

# Define the LDOS calculation function using GMRES (iterative solver)
def compute_ldos(H_sparse, sites, E, Gamma):
    """
    Compute the LDOS for given sites at energy E with broadening Gamma using GMRES.
    """
    Ns = H_sparse.shape[0]
    identity_sparse = csr_matrix(np.eye(Ns))
    
    
    # Define the right-hand side (identity vector for Green's function calculation)
    rhs = np.zeros(Ns, dtype=complex)
    
    ldos_values = {}
    for site in sites:
        m = coords_to_index(site[0], site[1], Nl, (Nl - 1)//2)
        
        # Calculate the Green's function using GMRES
        # G(E) = (E - H + i*Gamma)^(-1)
        A = E * identity_sparse - H_sparse + 1j * Gamma * identity_sparse
        rhs[m] = 1  # Delta function for site m
        
        # Solve for the Green's function at site m using GMRES
        solution, _ = gmres(A, rhs)
        
        # LDOS is the imaginary part of the Green's function diagonal element
        ldos_values[site] = np.imag(solution[m])
    
    return ldos_values

# Sites for which we want to compute LDOS
sites = [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (2, 2)]

# Compute the LDOS at the specified sites
ldos_values = compute_ldos(H, sites, E, Gamma)

# Output the computed LDOS values
for site, ldos in ldos_values.items():
    print(f"LDOS at site {site}: {ldos}")
