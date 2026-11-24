"""Tests of the methods on small synthetic examples with known results.

usage: python -m pytest tests
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import network_centrality  # noqa: E402
import spectral_clustering  # noqa: E402
from predictive_analysis import (hub_tests, leave_one_participant_out, one_predictor_test,  # noqa: E402
                                 prediction_scores, prediction_test, sign_flip_p)
from task_weighted_networks import (band_pass, compcor, network_from_signals, nuisance_design,  # noqa: E402
                                    remove_nuisance, task_weights, weighted_correlation)


# ================================================================ risk network atlas
def test_spectral_clustering_separates_two_clusters():
    """The eigengap finds two parts, and KNSC-Ncut and KNSC-Rcut assign each cluster to its own part."""
    cluster = np.random.default_rng(1).normal(size=(40, 3))
    coordinates = np.vstack([cluster, cluster + [20.0, 0.0, 0.0]])
    _, kernel = spectral_clustering.gaussian_kernel(coordinates)
    n_parts, _, _ = spectral_clustering.number_of_parts_by_eigengap(kernel, max_parts=5)
    assert n_parts == 2
    for method in ("knsc_ncut", "knsc_rcut"):
        labels, _, _ = spectral_clustering.knsc_clustering(kernel, 2, seed=1, iterations=300, tolerance=float("-inf"),
                                                           method=method)
        assert len(set(labels[:40])) == 1 and len(set(labels[40:])) == 1 and labels[0] != labels[40]


# ================================================================ denoising
def test_compcor_components_are_the_mean_signal_and_orthogonal_principal_components():
    """The first CompCor component is the mean signal of the mask; the others are orthogonal to it."""
    bold = np.random.default_rng(2).normal(size=(4, 4, 4, 80))
    mask = np.zeros((4, 4, 4), bool)
    mask[1:3, 1:3, 1:3] = True
    components = compcor(bold, mask)
    voxel_signals = bold[mask].T - bold[mask].T.mean(0)
    assert np.allclose(components[:, 0], voxel_signals.mean(1))
    assert np.allclose(components[:, 0] @ components[:, 1:], 0, atol=1e-8)


def test_nuisance_regression_leaves_residuals_orthogonal_to_the_design():
    """Design: constant, 5 + 5 CompCor components, 5 task regressors with their first differences, linear trend."""
    random = np.random.default_rng(3)
    columns = [f"wm_compcor_{index:02d}" for index in range(5)] + [f"csf_compcor_{index:02d}" for index in range(5)]
    confounds = pd.DataFrame(random.normal(size=(150, 10)), columns=columns)
    design = nuisance_design(confounds, random.normal(size=(150, 5)))
    assert design.shape == (150, 1 + 10 + 10 + 1)
    residuals = remove_nuisance(random.normal(size=(150, 3)), design)
    assert np.allclose(design.T @ residuals, 0, atol=1e-8)


def test_band_pass_filter_keeps_slow_and_removes_fast_signals():
    """The 0.01-0.1 Hz band-pass filter keeps a 0.04 Hz signal and removes a 0.2 Hz signal."""
    time = np.arange(600) * 2.0
    slow, fast = np.sin(2 * np.pi * 0.04 * time), np.sin(2 * np.pi * 0.2 * time)
    filtered = band_pass(np.column_stack([slow, fast]), 2.0)
    assert np.std(filtered[100:-100, 0]) > 0.9 * np.std(slow)
    assert np.std(filtered[100:-100, 1]) < 0.05


# ================================================================ task-weighted connectivity
def test_task_weights_are_zero_before_the_first_event_and_never_negative():
    """Task weights are the event boxcar convolved with the haemodynamic response and rectified."""
    weights = task_weights(100, 2.0, onsets=[50.0, 120.0], durations=[0.6, 0.6])
    assert (weights >= 0).all()
    assert (weights[:25] < 1e-12).all()              # the 25 volumes before the first event (TR 2 s, onset 50 s)
    assert weights.max() > 0.01


def test_network_edges_are_fisher_transformed_weighted_correlations():
    """With equal weights an edge is the Fisher-transformed Pearson correlation; volumes with zero weight are ignored."""
    signals = np.random.default_rng(4).normal(size=(120, 5))
    network = network_from_signals(signals, np.ones(120))
    assert np.allclose(network, network.T, equal_nan=True) and np.isnan(np.diag(network)).all()
    assert np.isclose(network[0, 1], np.arctanh(np.corrcoef(signals[:, 0], signals[:, 1])[0, 1]))
    weights = np.r_[np.ones(60), np.zeros(60)]
    assert np.isclose(weighted_correlation(signals[:, 0], signals[:, 1], weights),
                      np.corrcoef(signals[:60, 0], signals[:60, 1])[0, 1])


# ================================================================ centrality
def test_eigenvector_centrality_is_highest_for_the_hub_of_a_star():
    """In a star network, the hub has the highest eigenvector centrality."""
    network = np.zeros((5, 5))
    network[0, 1:] = network[1:, 0] = 1.0
    assert network_centrality.eigenvector_centrality(network).argmax() == 0


def test_multiplex_eigenvector_centralities():
    """When all layers of a multiplex network are the same network, both multiplex eigenvector centralities equal its
    eigenvector centrality; for different layers, the uniform centrality is the eigenvector centrality of the layers'
    positive edges added together, and the mean single-layer centrality is the mean of the layers' centralities."""
    random_generator = np.random.default_rng(5)
    layers = []
    for _ in range(4):
        network = random_generator.normal(size=(6, 6))
        network = (network + network.T) / 2
        np.fill_diagonal(network, np.nan)
        layers.append(network)
    eigenvector = network_centrality.eigenvector_centrality
    assert np.allclose(network_centrality.mean_single_layer_eigenvector_centrality([layers[0]] * 4), eigenvector(layers[0]))
    assert np.allclose(network_centrality.uniform_eigenvector_centrality([layers[0]] * 4), eigenvector(layers[0]))
    summed = sum(np.where(np.isfinite(network), np.maximum(network, 0), 0.0) for network in layers)
    assert np.allclose(network_centrality.uniform_eigenvector_centrality(layers), eigenvector(summed))
    assert np.allclose(network_centrality.mean_single_layer_eigenvector_centrality(layers),
                       np.mean([eigenvector(network) for network in layers], axis=0))



def test_versatility_of_identical_layers_is_their_eigenvector_centrality():
    """When all layers are the same network, every copy of a region has the same value in the leading eigenvector of the
    supra-adjacency matrix, so the versatility equals the eigenvector centrality of the layer (for any interlayer
    weight); for different layers it is a non-negative unit vector."""
    random_generator = np.random.default_rng(8)
    network = random_generator.normal(size=(7, 7))
    network = (network + network.T) / 2
    np.fill_diagonal(network, np.nan)
    for weight in (None, 0.05, 2.0):
        assert np.allclose(network_centrality.versatility([network] * 5, weight),
                           network_centrality.eigenvector_centrality(network))
    layers = [network + random_generator.normal(scale=0.5, size=(7, 7)) for _ in range(4)]
    layers = [(layer + layer.T) / 2 for layer in layers]
    value = network_centrality.versatility(layers)
    assert np.all(value >= 0) and np.isclose(np.linalg.norm(value), 1)


def test_absolute_link_measures_equal_the_positive_ones_for_non_negative_networks():
    """Robustness measures: with no negative edges, the eigenvector centralities of the absolute edge weights equal
    those of the positive edges."""
    random_generator = np.random.default_rng(10)
    layers = []
    for _ in range(3):
        network = np.abs(random_generator.normal(size=(6, 6)))
        network = (network + network.T) / 2
        np.fill_diagonal(network, np.nan)
        layers.append(network)
    assert np.allclose(network_centrality.absolute_eigenvector_centrality(layers[0]),
                       network_centrality.eigenvector_centrality(layers[0]))
    assert np.allclose(network_centrality.uniform_absolute_eigenvector_centrality(layers),
                       network_centrality.uniform_eigenvector_centrality(layers))

# ================================================================ stability and prediction
def test_identical_halves_give_an_icc_of_one_and_perfect_identification():
    """Identical odd-run and even-run profiles give ICC(3,1) = 1 and an identification accuracy of 1."""
    profiles = np.random.default_rng(6).normal(size=(6, 10))
    assert np.isclose(network_centrality.icc_3_1(profiles[:, 0], profiles[:, 0]), 1.0)
    accuracy, idiff = network_centrality.fingerprint(profiles, profiles)
    assert accuracy == 1.0 and idiff > 0


def test_leave_one_participant_out_prediction_recovers_a_linear_relation():
    """An outcome that is a linear function of the predictor is predicted exactly in held-out participants, and the
    numpy version for one predictor gives the same scores and permutation p-value as prediction_test."""
    data = pd.DataFrame({"subject": [f"{index:02d}" for index in range(12)], "session": "3t2", "x": np.arange(12.0)})
    data["y"] = 2 * data["x"] + 1
    _, observed, predicted = leave_one_participant_out(data, ["x"], "y", "7t2")
    assert np.isclose(prediction_scores(observed, predicted)["cv_r2"], 1.0)
    noisy = data.assign(y=data["y"] + np.random.default_rng(7).normal(scale=5.0, size=12))
    full = prediction_test(noisy, ["x"], "y", 200, np.random.default_rng(8), "7t2")
    fast = one_predictor_test(noisy, "x", "y", 200, np.random.default_rng(8))
    assert np.isclose(full["cv_r2"], fast["cv_r2"]) and np.isclose(full["p_perm_cv_r2"], fast["p_perm_cv_r2"])


def test_exact_sign_flip_p_matches_all_sign_patterns():
    """The chunked exact sign-flip p equals the share of all 2^n sign patterns with a sum at least the observed sum."""
    import itertools
    differences = np.random.default_rng(9).normal(0.3, 1, 18)
    patterns = np.array(list(itertools.product([1, -1], repeat=18)))
    expected = np.mean(patterns @ differences >= differences.sum() - 1e-12)
    assert np.isclose(sign_flip_p(differences, np.random.default_rng(0)), expected)


def test_hub_groups_are_defined_from_the_networks_only(tmp_path):
    """The hub group contains the participants in whom the target region ranks first, the top-2 group those in whom it
    ranks first or second, independently of the outcome."""
    rows, multiplex_rows, behaviour = [], [], []
    for index in range(12):
        participant = f"{index + 1:02d}"
        target_value = 0.8 + 0.01 * index if index < 9 else (0.5 + 0.01 * index if index < 11 else 0.1)  # 9 hubs
        values = {1: target_value, 2: 0.7, 3: 0.3}
        for roi, value in values.items():
            rows.append({"subject": participant, "session": "3t2", "network": "C", "roi": roi, "eigenvector_pos": value})
            multiplex_rows.append({"subject": participant, "session": "3t2", "roi": roi,
                                   **{measure: value for measure in ("multiplex_eigenvector_mean_single_layer",
                                                                     "multiplex_eigenvector_uniform",
                                                                     "multiplex_versatility")}})
        behaviour.append({"subject": participant, "session": "3t2", "rnp_mean": 0.3 + 0.03 * index})
    rows = hub_tests(pd.DataFrame(rows), pd.DataFrame(multiplex_rows), pd.DataFrame(behaviour), "rnp_mean", 10,
                     np.random.default_rng(1), 1, tmp_path)
    sizes = {(row["measure"], row["group"]): row["participants"] for row in rows}
    assert sizes[("eigenvector_pos", "hub")] == 9
    assert sizes[("multiplex_versatility", "top 2")] == 11
    assert sizes[("multiplex_eigenvector_uniform", "hub under all three definitions")] == 9


def test_motion_exclusion_follows_the_stringent_criteria(tmp_path):
    """Parkes et al. (2018), stringent: mean FD > 0.25 mm, more than 20% of scans above 0.2 mm, or any FD > 5 mm."""
    import paper_results
    cases = {"low": [np.nan] + [0.1] * 99, "many_above": [np.nan] + [0.1] * 70 + [0.3] * 29,
             "one_large": [np.nan] + [0.05] * 98 + [6.0], "high_mean": [np.nan] + [0.26] * 99}
    excluded = {}
    for name, fd in cases.items():
        folder = tmp_path / name / "runs"
        folder.mkdir(parents=True)
        pd.DataFrame({"framewise_displacement": fd}).to_csv(folder / "sub-01_run-1_confounds.tsv", sep="\t", index=False)
        excluded[name] = paper_results.motion_summary(tmp_path / name)["excluded"]
    assert excluded == {"low": False, "many_above": True, "one_large": True, "high_mean": True}
