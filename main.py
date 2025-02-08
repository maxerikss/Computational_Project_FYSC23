import numpy as np
import matplotlib.pyplot as plt
import time
import os

def get_task_filename(task_num, folder="Comp_Proj1/Figures"):
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"task{task_num}.png")

def construct_hamiltonian_1D(N, epsilon=0, V=-1, adsorbate=False, epsilon_0=-1, V_0=0):
    size = N + 1 if adsorbate else N
    H = np.zeros((size, size))
    np.fill_diagonal(H, epsilon)
    np.fill_diagonal(H[1:], V)
    np.fill_diagonal(H[:, 1:], V)

    if adsorbate:
        H[-1, -1] = epsilon_0
        H[-1, 0] = V_0
        H[0, -1] = V_0

    return H

def construct_hamiltonian_2D(Nl, epsilon=0, V=-1, adsorption_type=None, epsilon_0=-2.0, V_0=-1.3):
    Ns = Nl * Nl
    size = Ns + 1 if adsorption_type else Ns
    H = np.zeros((size, size))
    np.fill_diagonal(H, epsilon)

    for i in range(Ns):
        if (i + 1) % Nl != 0:
            H[i, i + 1] = V
            H[i + 1, i] = V
        if i + Nl < Ns:
            H[i, i + Nl] = V
            H[i + Nl, i] = V

    if adsorption_type:
        ad_index = Ns
        H[ad_index, ad_index] = epsilon_0

        if adsorption_type == "atop":
            H[ad_index, 0] = V_0
            H[0, ad_index] = V_0
        elif adsorption_type == "bridge":
            H[ad_index, 0] = V_0 / np.sqrt(2)
            H[ad_index, 1] = V_0 / np.sqrt(2)
            H[0, ad_index] = V_0 / np.sqrt(2)
            H[1, ad_index] = V_0 / np.sqrt(2)
        elif adsorption_type == "center":
            H[ad_index, 0] = V_0 / 2
            H[ad_index, 1] = V_0 / 2
            H[ad_index, Nl] = V_0 / 2
            H[ad_index, Nl + 1] = V_0 / 2
            H[0, ad_index] = V_0 / 2
            H[1, ad_index] = V_0 / 2
            H[Nl, ad_index] = V_0 / 2
            H[Nl + 1, ad_index] = V_0 / 2
        elif adsorption_type == "impurity":
            H[0, 0] = epsilon_0
            H[0, 1] = V_0
            H[0, Nl] = V_0
            H[1, 0] = V_0
            H[Nl, 0] = V_0

    return H

def lorentzian(E, E_lambda, Gamma):
    return (Gamma / np.pi) / ((E - E_lambda) ** 2 + Gamma**2)

def compute_ldos(H, sites, E_range=(-6, 6), Gamma=0.05, num_points=500):
    start_time = time.time()  # Benchmarking

    E_values = np.linspace(E_range[0], E_range[1], num_points)
    eigenvalues, eigenvectors = np.linalg.eigh(H)  # Diagonalize H

    ldos = {site: np.zeros_like(E_values) for site in sites}

    for site in sites:
        for lambda_idx in range(len(eigenvalues)):
            ldos[site] += (np.abs(eigenvectors[site, lambda_idx]) ** 2) * lorentzian(
                E_values, eigenvalues[lambda_idx], Gamma
            )

    end_time = time.time()
    print(f"LDOS computation time: {end_time - start_time:.4f} seconds")

    return E_values, ldos

def plot_ldos(E_values, ldos_dict, sites, task_num=1, selected_sites_2D=None, adsorption_types=None, adsorbate=False, clean_twoD=False):
    num_plots = len(ldos_dict)
    fig, axes = plt.subplots(num_plots, 1, figsize=(8, 9), sharex=True, sharey=True)
    title = f"Local Density of States for Selected Sites in 2D" if clean_twoD else f"Local Density of States for Selected Sites"
    fig.suptitle(title)
    if adsorbate:
        title = f"Local Density of States for Selected Sites in 2D for Adsorption cases" if clean_twoD else f"Local Density of States for Adsorbate Cases"
        fig.suptitle(title)

    if num_plots == 1:
        axes = [axes]

    for idx, ((epsilon_0, V_0), ldos) in enumerate(ldos_dict.items()):
        for i, site in enumerate(sites):
            site_label = (selected_sites_2D[i] if clean_twoD and selected_sites_2D else site + 1)
            axes[idx].plot(E_values, ldos[site], label=f"Site {site_label}")
        if adsorbate and adsorption_types:
            for adsorption_type in adsorption_types:
                title_str = f"ε0={epsilon_0}, V0={V_0} at {adsorption_type}"
            axes[idx].set_title(title_str)
        elif adsorbate and not adsorption_types:
            axes[idx].set_title(f"ε0={epsilon_0}, V0={V_0}")
        axes[idx].set_ylabel(r"$g^L_{i}(E, \Gamma)$")
        axes[idx].legend()
        axes[idx].grid()

    axes[-1].set_xlabel("Energy (E)")
    plt.tight_layout()

    filename = get_task_filename(task_num)
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.show()

# Parameters
N = 501
sites_1D = [0, 1, 2, 4, 9, 49, 99, 250]
Gamma = 0.05
Nl = 81
Ns = Nl * Nl
selected_sites_2D = [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (2, 2)]
sites_2D = [(i * Nl + j) for i, j in selected_sites_2D]

# Task 1: Clean Surface 1D
H_clean_1D = construct_hamiltonian_1D(N)
E_values_clean_1D, ldos_clean_1D = compute_ldos(H_clean_1D, sites_1D, Gamma=Gamma)
plot_ldos(E_values_clean_1D, {("Clean", "Surface"): ldos_clean_1D}, sites_1D, task_num=1)

# Task 2: Surface with Adsorbate 1D
adsorbate_cases = [(-1, 0), (-1, -0.3), (-1, 3.0)]
ldos_results_1D = {}

for epsilon_0, V_0 in adsorbate_cases:
    H_adsorbate = construct_hamiltonian_1D(N, adsorbate=True, epsilon_0=epsilon_0, V_0=V_0)
    E_values_ads, ldos_ads = compute_ldos(H_adsorbate, [0] + sites_1D, Gamma=Gamma)
    ldos_results_1D[(epsilon_0, V_0)] = ldos_ads

plot_ldos(E_values_ads, ldos_results_1D, [0] + sites_1D, adsorbate=True, task_num=2)

# Task 3: Clean Surface 2D
# This might not be very optimal or correct for that matter (it takes ~30 seconds to compute)
H_clean_2D = construct_hamiltonian_2D(Nl)
E_values_clean_2D, ldos_clean_2D = compute_ldos(H_clean_2D, sites_2D, E_range=(-8, 8), Gamma=Gamma)
plot_ldos(E_values_clean_2D, {("Clean", "Surface"): ldos_clean_2D}, sites_2D, selected_sites_2D=selected_sites_2D, clean_twoD=True, task_num=3)

# Task 4: Surface with Adsorbate 2D
# This might not be very optimal or correct for that matter (it takes ~30 seconds to compute each case, 2 minutes total)
adsorption_types = ["atop", "bridge", "center", "impurity"]
ldos_results_2D = {}

for adsorption_type in adsorption_types:
    H_adsorbate_2D = construct_hamiltonian_2D(Nl, adsorption_type=adsorption_type, epsilon_0=-2.0, V_0=-1.3)
    E_values_ads_2D, ldos_ads_2D = compute_ldos(H_adsorbate_2D, [0, H_adsorbate_2D.shape[0] - 1], E_range=(-8, 8), Gamma=Gamma)
    ldos_results_2D[("-2.0", f"-1.3, {adsorption_type}")] = ldos_ads_2D

plot_ldos(E_values_ads_2D, ldos_results_2D, [0, H_adsorbate_2D.shape[0] - 1], selected_sites_2D=selected_sites_2D, clean_twoD=True, adsorbate=True, task_num=4)
