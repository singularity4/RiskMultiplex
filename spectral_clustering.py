"""Spectral clustering of voxel clusters into regions, used by risk_brain_network_atlas.py.

For each cluster of connected voxels, the module computes:
    kernel              Gaussian kernel of the voxel coordinates (MNI, mm), K_ij = exp(-d_ij^2 / sigma^2)
    number of clusters  position of the largest eigengap of the normalised graph Laplacian (von Luxburg 2007)
    clustering          kernel non-negative spectral clustering (KNSC) with the normalised cut (Ncut, default) or
                        the ratio cut (Rcut); Tolić, Antulov-Fantulin & Kopriva (2018), Pattern Recognition 82,
                        40-55, doi:10.1016/j.patcog.2018.04.029
Notation: H is the cluster indicator matrix, K the kernel matrix and D_inv_sqrt = D^(-1/2), where D is the degree
matrix of K.
"""
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform

EPS = 1e-12
ALPHA = 10.0      # weight of the kernel reconstruction term
MU = 100.0        # weight of the orthogonality penalty ||H H' - I||^2


# ================================================================ KNSC update rules and objectives
def update_rcut(state, context):
    """Perform one multiplicative KNSC-Rcut update.

    Each factor is multiplied by the ratio of the negative to the positive part of its gradient (KKT conditions).
    """
    H, F = state["H"], state["F"]
    K, alpha, mu = context["K"], context["alpha"], context["mu"]

    negative_part = alpha * (F.T @ K) + 2 * mu * H
    positive_part = alpha * (F.T @ K @ F @ H) + 2 * mu * (H @ H.T) @ H
    H = H * negative_part / np.maximum(positive_part, EPS)

    negative_part = K @ H.T
    positive_part = K @ F @ H @ H.T
    F = F * negative_part / np.maximum(positive_part, EPS)
    return {"H": H, "F": F}


def objective_rcut(state, context):
    """Return the KNSC-Rcut objective alpha ||Phi(X) - Phi(X) F H||_F^2 + mu ||H H' - I||_F^2, computed from K."""
    H, F = state["H"], state["F"]
    K, alpha, mu = context["K"], context["alpha"], context["mu"]
    fidelity = np.trace(K) - 2 * np.trace(H.T @ F.T @ K) + np.trace(H.T @ F.T @ K @ F @ H)
    orthogonality = np.linalg.norm(H @ H.T - np.eye(H.shape[0]), "fro") ** 2
    return float(alpha * fidelity + mu * orthogonality)


def update_ncut(state, context):
    """Perform one multiplicative KNSC-Ncut update: the Rcut update with the data scaled by the fixed D^(-1/2)."""
    Z, F = state["H"], state["F"]
    K, alpha, mu = context["K"], context["alpha"], context["mu"]
    D_inv_sqrt = context["D_inv_sqrt"]
    K_scaled = K @ D_inv_sqrt

    negative_part = alpha * (F.T @ K_scaled) + 2 * mu * Z
    positive_part = alpha * (F.T @ K @ F @ Z) + 2 * mu * (Z @ Z.T) @ Z
    Z = Z * negative_part / np.maximum(positive_part, EPS)

    negative_part = K_scaled @ Z.T
    positive_part = K @ F @ Z @ Z.T
    F = F * negative_part / np.maximum(positive_part, EPS)
    return {"H": Z, "F": F}


def objective_ncut(state, context):
    """Return the KNSC-Ncut objective: the Rcut objective with the data scaled by D^(-1/2)."""
    Z, F = state["H"], state["F"]
    K, alpha, mu, D_inv_sqrt = context["K"], context["alpha"], context["mu"], context["D_inv_sqrt"]
    K_normalized = D_inv_sqrt @ K @ D_inv_sqrt
    fidelity = (np.trace(K_normalized) - 2 * np.trace(Z.T @ F.T @ K @ D_inv_sqrt)
                + np.trace(Z.T @ F.T @ K @ F @ Z))
    orthogonality = np.linalg.norm(Z @ Z.T - np.eye(Z.shape[0]), "fro") ** 2
    return float(alpha * fidelity + mu * orthogonality)


METHODS = {"knsc_ncut": (update_ncut, objective_ncut), "knsc_rcut": (update_rcut, objective_rcut)}


def labels(clustering_matrix):
    """Return the cluster label of each voxel: the row of H with the largest value (no k-means step)."""
    return np.argmax(clustering_matrix, axis=0)


# ================================================================ kernel and number of clusters
def gaussian_kernel(coordinates_mm):
    """Compute the Gaussian kernel matrix of the voxel coordinates.

    Choice: sigma is the median pairwise distance between voxels (median heuristic), so the kernel width scales with
    the cluster size without parameter tuning.
    """
    distances = pdist(coordinates_mm, metric="euclidean")
    nonzero_distances = distances[distances > 0]
    if not len(nonzero_distances):
        raise RuntimeError("The cluster has no distinct voxel positions.")
    sigma = float(np.median(nonzero_distances))
    kernel = np.exp(-squareform(distances ** 2) / (sigma ** 2))
    return sigma, kernel


def number_of_parts_by_eigengap(kernel, max_parts):
    """Estimate the number of clusters with the eigengap heuristic.

    Choice: the number of clusters is the position of the largest gap between consecutive eigenvalues of the symmetric
    normalised Laplacian L = I - D^(-1/2) K D^(-1/2), between 1 and max_parts (von Luxburg 2007). An estimate of 1
    leaves the cluster undivided.
    """
    n_voxels = kernel.shape[0]
    largest = min(max_parts, n_voxels - 1)
    if largest < 1:
        return 1, np.array([0.0]), np.array([])
    degrees = kernel.sum(axis=1)
    inverse_sqrt_degrees = 1.0 / np.sqrt(np.maximum(degrees, EPS))
    normalized_affinity = inverse_sqrt_degrees[:, None] * kernel * inverse_sqrt_degrees[None, :]
    laplacian = np.eye(n_voxels) - normalized_affinity
    eigenvalues = eigh(laplacian, subset_by_index=[0, largest], eigvals_only=True, check_finite=False)
    eigenvalues = np.sort(np.real(eigenvalues))
    eigengaps = np.diff(eigenvalues)
    n_parts = int(np.argmax(eigengaps[:largest]) + 1)
    return n_parts, eigenvalues, eigengaps


# ================================================================ clustering
def has_converged(objective_values, tolerance):
    """Return True if the relative decrease of the objective is below the tolerance (-inf runs all iterations)."""
    if len(objective_values) < 2:
        return False
    previous, current = objective_values[-2], objective_values[-1]
    return (previous - current) <= tolerance * max(1.0, abs(previous))


def knsc_clustering(kernel, n_parts, seed, iterations, tolerance, method="knsc_ncut", alpha=ALPHA, mu=MU):
    """Run KNSC from a random non-negative initialisation.

    Returns the cluster label of each voxel, the final objective value and the number of iterations.
    """
    update, objective = METHODS[method]
    degrees = kernel.sum(axis=1)
    context = {"K": kernel, "D_inv_sqrt": np.diag(1.0 / np.sqrt(np.maximum(degrees, EPS))), "alpha": alpha, "mu": mu}
    random_generator = np.random.default_rng(seed)
    state = {"H": random_generator.random((n_parts, kernel.shape[0])),
             "F": random_generator.random((kernel.shape[0], n_parts))}
    objective_values = [objective(state, context)]
    for _ in range(iterations):
        state = update(state, context)
        objective_values.append(objective(state, context))
        if has_converged(objective_values, tolerance):
            break
    return labels(state["H"]), float(objective_values[-1]), len(objective_values) - 1
