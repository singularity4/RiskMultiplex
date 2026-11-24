"""Regional network centrality measures and their split-half stability within participants.

Input: output folders of task_weighted_networks.py (any number, at any folder depth).
  1. Eigenvector centrality for each region in each network (C, B and A)       -> centrality_all_nodes.tsv
     Multiplex network A (12 payoff layers with the same regions, different edges in each layer and no interlayer
     edges): mean single-layer and uniform multiplex eigenvector centrality, and versatility (De Domenico et al.
     2015), for each region. Robustness checks: eigenvector centrality of the absolute edge weights (C, B, A), and
     uniform multiplex eigenvector centrality of the absolute edge weights and of Ledoit-Wolf shrunk layers (A; the
     shrunk layers are computed in step 2 from the regional time series)
                                                                                   -> multiplex_centrality.tsv
  2. Networks C and A recomputed from the denoised regional time series for all runs, odd runs (1, 3, 5, 7) and even
     runs (2, 4, 6, 8). For each half: eigenvector centrality in network C, the two multiplex eigenvector centralities,
     versatility and the uniform centrality of the shrunk layers                   -> measures_by_half.tsv
  3. Split-half stability, computed without behavioural data: identification of each participant's regional profile
     from odd runs among all profiles from even runs (identification accuracy, Finn et al. 2015; differential
     identifiability Idiff, Amico & Goñi 2018), and ICC(3,1) between halves for the target region and averaged across
     regions                                                                       -> stability_by_measure.tsv
  4. Network C recomputed separately for each presentation order (safe first, risky first; all runs), with
     eigenvector centrality                                                        -> centrality_by_order.tsv
Step 1 also accepts networks from other pipelines if their file names follow task_weighted_networks.py.

usage: python network_centrality.py --networks FOLDER [FOLDER ...] --out FOLDER [--variant unsmoothed | smoothed]
                                    [--session 3t2] [--regions atlas/risk_atlas_regions.csv] [--check-saved]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from task_weighted_networks import (DUMMY_VOLUMES, LAYER_FACTORS, TARGET_LABEL, layer_name, network_from_runs,
                                    network_weights, order_weights, parse_file_name)


# ================================================================ regional centrality measures
def eigenvector_centrality(network):
    """Compute eigenvector centrality on positive edges (unit norm).

    Eigenvector centrality is high for regions that are coupled with other central regions.
    Choice: negative edges are set to 0 because the leading eigenvector is guaranteed to be non-negative only for
    non-negative matrices (Perron-Frobenius theorem; as in Lohmann et al. 2010).
    """
    positive = np.where(np.isfinite(network), np.maximum(network, 0), 0.0)
    np.fill_diagonal(positive, 0.0)
    eigenvalues, eigenvectors = np.linalg.eigh(positive)
    centrality = np.abs(eigenvectors[:, -1])
    return centrality / np.linalg.norm(centrality)


def mean_single_layer_eigenvector_centrality(layer_networks):
    """Compute the mean single-layer multiplex eigenvector centrality.

    The multiplex network has the same regions in every layer, different edges in each layer and no interlayer
    edges. The eigenvector centrality (positive edges, unit norm) of each layer is computed on its own and averaged
    over the layers (the independent-layer centralities of Solá et al. 2013, averaged).
    """
    return np.mean([eigenvector_centrality(network) for network in layer_networks], axis=0)


def uniform_eigenvector_centrality(layer_networks):
    """Compute the uniform multiplex eigenvector centrality (Solá et al. 2013).

    The eigenvector centrality of the positive edges of all layers added together: a region is central if it is
    connected, in any layer, to regions that are themselves central. All layers have equal weight; there are no
    interlayer edges and no free parameters.
    """
    summed = sum(np.where(np.isfinite(network), np.maximum(network, 0), 0.0) for network in layer_networks)
    return eigenvector_centrality(summed)


def versatility(layer_networks, interlayer_weight=None):
    """Compute the versatility, the multilayer eigenvector centrality of De Domenico et al. (2015).

    The layers (positive edges) form the diagonal blocks of a supra-adjacency matrix, and each region is connected
    to its replica in every other layer with the interlayer weight. The leading eigenvector of this matrix gives one
    value per region and layer; a region's versatility is the sum over its replicas (unit norm).
    Choice: the interlayer weight is the participant's mean positive edge weight within layers (as in Breedt et al.
    2023, where interlayer edges had the weight of intralayer edges).
    """
    layers = []
    for network in layer_networks:
        positive = np.where(np.isfinite(network), np.maximum(network, 0), 0.0)
        np.fill_diagonal(positive, 0.0)
        layers.append(positive)
    if interlayer_weight is None:
        interlayer_weight = float(np.mean([layer[layer > 0].mean() for layer in layers]))
    n_regions, n_layers = len(layers[0]), len(layers)
    supra = np.kron(np.ones((n_layers, n_layers)) - np.eye(n_layers), interlayer_weight * np.eye(n_regions))
    for index, layer in enumerate(layers):
        supra[index * n_regions:(index + 1) * n_regions, index * n_regions:(index + 1) * n_regions] = layer
    leading = np.abs(np.linalg.eigh(supra)[1][:, -1]).reshape(n_layers, n_regions)
    value = leading.sum(0)
    return value / np.linalg.norm(value)


def absolute_eigenvector_centrality(network):
    """Robustness check: eigenvector centrality of the absolute edge weights.

    Positive and negative correlations both count.
    """
    return eigenvector_centrality(np.abs(np.where(np.isfinite(network), network, 0.0)))


def uniform_absolute_eigenvector_centrality(layer_networks):
    """Robustness check: uniform multiplex eigenvector centrality of the absolute edge weights.

    The absolute edge weights of all layers are added together.
    """
    return eigenvector_centrality(sum(np.abs(np.where(np.isfinite(network), network, 0.0)) for network in layer_networks))


def shrunk_network_from_runs(signals_per_run, weights_per_run):
    """Robustness check: the network of network_from_runs with the weighted correlation matrix shrunk towards the
    identity by the Ledoit-Wolf estimator (Ledoit & Wolf 2004), then Fisher z (diagonal empty).

    Choice: the weighted time series are standardised so that their sample covariance equals the weighted correlation
    matrix; the shrinkage intensity is estimated for each network.
    """
    from sklearn.covariance import ledoit_wolf
    signals, weights = np.concatenate(signals_per_run, axis=0), np.concatenate(weights_per_run, axis=0)
    valid = np.isfinite(signals).all(1) & np.isfinite(weights) & (weights > 0)
    signals, weights = signals[valid], weights[valid] / weights[valid].sum()
    centred = signals - weights @ signals
    standard = centred / np.sqrt(weights @ centred ** 2)
    shrunk, _ = ledoit_wolf(standard * np.sqrt(weights * len(weights))[:, None], assume_centered=True)
    correlation = shrunk / np.sqrt(np.outer(np.diag(shrunk), np.diag(shrunk)))
    network = np.arctanh(np.clip(correlation, -0.999999, 0.999999))
    np.fill_diagonal(network, np.nan)
    return network


# ================================================================ split-half stability (odd vs even runs)
def icc_3_1(first, second):
    """Compute ICC(3,1) between two measurements of the same participants (consistency; Shrout & Fleiss 1979)."""
    measurements = np.column_stack([first, second])
    n_participants, n_measurements = measurements.shape
    grand_mean = measurements.mean()
    mean_square_participants = n_measurements * ((measurements.mean(1) - grand_mean) ** 2).sum() / (n_participants - 1)
    residuals = measurements - measurements.mean(1, keepdims=True) - measurements.mean(0, keepdims=True) + grand_mean
    mean_square_error = (residuals ** 2).sum() / ((n_participants - 1) * (n_measurements - 1))
    return float((mean_square_participants - mean_square_error)
                 / (mean_square_participants + (n_measurements - 1) * mean_square_error))


def fingerprint(odd, even):
    """Compute identification accuracy (Finn et al. 2015) and differential identifiability Idiff (Amico & Goñi 2018).

    odd, even: participants x regions profiles from the two halves, in the same participant order. Each profile from
    odd runs is matched to all profiles from even runs, and vice versa; the accuracy is the mean of both directions
    (chance level: 1 / participants).
    Choice: the group mean is subtracted from each profile, so that identification relies on individual differences
    rather than on the network structure shared by all participants.
    """
    n_participants = len(odd)
    odd_centred, even_centred = odd - odd.mean(0), even - even.mean(0)
    similarity = np.corrcoef(odd_centred, even_centred)[:n_participants, n_participants:]     # odd x even
    same_participant = np.arange(n_participants)
    accuracy = 0.5 * (np.mean(similarity.argmax(1) == same_participant) + np.mean(similarity.argmax(0) == same_participant))
    idiff = 100 * (np.mean(np.diag(similarity)) - np.mean(similarity[~np.eye(n_participants, dtype=bool)]))
    return float(accuracy), float(idiff)


# ================================================================ main program
HERE = Path(__file__).resolve().parent
# network version: (network folder, network file suffix, suffix of the denoised regional time series files)
VARIANTS = {"unsmoothed": ("networks_unsmoothed", "_connectivity-unsmoothed_z.csv", "_roi_timeseries_unsmoothed_cleaned.tsv"),
            "smoothed": ("networks", "_connectivity_z.csv", "_roi_timeseries_cleaned.tsv")}
HALVES = {"all": range(1, 9), "odd": (1, 3, 5, 7), "even": (2, 4, 6, 8)}   # Choice: halves defined by run parity


def read_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--networks", nargs="+", required=True, type=Path, help="output folders of task_weighted_networks.py")
    parser.add_argument("--out", required=True, type=Path, help="output folder")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="unsmoothed",
                        help="unsmoothed regional time series (CONN default) or 6 mm smoothed time series "
                             "(robustness check)")
    parser.add_argument("--session", default="3t2", help="session for the split-half stability analysis")
    parser.add_argument("--regions", type=Path, default=HERE / "atlas" / "risk_atlas_regions.csv", help="region table of the atlas")
    parser.add_argument("--target-label", default=TARGET_LABEL, help="label of the target region")
    parser.add_argument("--check-saved", action="store_true", help="compare the recomputed networks (all runs) with the saved networks")
    return parser.parse_args()


def centrality_table(network_files, file_suffix, region_names):
    """Step 1: compute the regional measures for each network file."""
    category, level = LAYER_FACTORS
    rows = []
    for path in network_files:
        parts = parse_file_name(path.name[:-len(file_suffix)])
        if not {"sub", "ses", "network"} <= set(parts) or parts["network"] not in ("A", "B", "C"):
            continue
        network = pd.read_csv(path, index_col=0).to_numpy(float)
        eigenvector = eigenvector_centrality(network)
        eigenvector_absolute = absolute_eigenvector_centrality(network)
        if parts["network"] == "C":
            layer = "all"
        elif parts["network"] == "B":
            layer = f"run-{parts['run']}"
        else:
            layer = layer_name((parts[category], int(parts[level])))
        off_diagonal = ~np.eye(len(network), dtype=bool)
        for region in range(len(network)):
            rows.append({"subject": parts["sub"], "session": parts["ses"], "network": parts["network"], "layer": layer,
                         "run": int(parts["run"]) if parts.get("run") else np.nan, category: parts.get(category),
                         level: int(parts[level]) if parts.get(level) else np.nan, "roi": region + 1,
                         "label": region_names.get(region + 1, ""), "eigenvector_pos": eigenvector[region],
                         "eigenvector_abs": eigenvector_absolute[region],
                         "nan_edges": int(np.isnan(network[region][off_diagonal[region]]).sum())})
    return pd.DataFrame(rows)


MULTIPLEX_COLUMNS = ["subject", "session", "layers", "roi", "label", "multiplex_eigenvector_mean_single_layer",
                     "multiplex_eigenvector_uniform", "multiplex_versatility", "multiplex_eigenvector_uniform_absolute",
                     "multiplex_eigenvector_uniform_shrunk"]


def multiplex_table(network_files, file_suffix, region_names):
    """Step 1, multiplex network A: the two multiplex eigenvector centralities, the versatility and the uniform
    centrality of the absolute edge weights of each region, for each participant and session.

    Choice: only participants and sessions with all payoff layers; participants are sorted by ID.
    """
    category, level = LAYER_FACTORS
    layer_files = {}
    for path in network_files:
        parts = parse_file_name(path.name[:-len(file_suffix)])
        if {"sub", "ses", "network"} <= set(parts) and parts["network"] == "A":
            layer_files.setdefault((parts["sub"], parts["ses"]), {})[(parts[category], int(parts[level]))] = path
    all_layers = sorted({layer for layers in layer_files.values() for layer in layers})
    rows = []
    for (participant, session), layers in sorted(layer_files.items()):
        if sorted(layers) != all_layers:
            continue
        layer_networks = [pd.read_csv(layers[layer], index_col=0).to_numpy(float) for layer in all_layers]
        mean_single_layer = mean_single_layer_eigenvector_centrality(layer_networks)
        uniform = uniform_eigenvector_centrality(layer_networks)
        multilayer = versatility(layer_networks)
        uniform_absolute = uniform_absolute_eigenvector_centrality(layer_networks)
        for region in range(len(uniform)):
            rows.append({"subject": participant, "session": session, "layers": len(layer_networks), "roi": region + 1,
                         "label": region_names.get(region + 1, ""),
                         "multiplex_eigenvector_mean_single_layer": float(mean_single_layer[region]),
                         "multiplex_eigenvector_uniform": float(uniform[region]),
                         "multiplex_versatility": float(multilayer[region]),
                         "multiplex_eigenvector_uniform_absolute": float(uniform_absolute[region])})
    return pd.DataFrame(rows, columns=MULTIPLEX_COLUMNS)


def read_run_signals(folders, session, signals_suffix):
    """Read the denoised regional time series, the saved weights and the recomputed payoff-layer and order weights
    of each run."""
    participants = {}
    for folder in folders:
        for participant_folder in sorted(Path(folder).rglob(f"sub-*_ses-{session}")):
            if not participant_folder.is_dir() or not (participant_folder / "runs").is_dir():
                continue
            participant = parse_file_name(participant_folder.name)["sub"]
            run_summaries = json.load(open(participant_folder / "subject_summary.json")).get("runs", {})
            # repetition time of each run; it differs slightly between runs in ds007508 (e.g. 2.29790 vs 2.29792 s)
            repetition_times = {int(run): float(summary["tr"]) for run, summary in run_summaries.items() if summary.get("tr")}
            runs = {}
            for signals_path in sorted(participant_folder.glob(f"runs/*_run-*{signals_suffix}")):
                run = int(parse_file_name(signals_path.name)["run"])
                if run not in repetition_times:
                    raise ValueError(f"{participant_folder}: no repetition time for run {run} in subject_summary.json")
                repetition_time = repetition_times[run]
                prefix = str(signals_path).replace(signals_suffix, "")
                signals = pd.read_csv(signals_path, sep="\t").to_numpy(float)
                confounds = pd.read_csv(prefix + "_confounds.tsv", sep="\t")
                layer_weights, weights_by_order = {}, {}
                if Path(prefix + "_events.tsv").exists():
                    events = pd.read_csv(prefix + "_events.tsv", sep="\t")
                    events["onset"] = events["onset"] - DUMMY_VOLUMES * repetition_time
                    layer_weights = network_weights(events, len(signals), repetition_time)[1]
                    weights_by_order = order_weights(events, len(signals), repetition_time)
                runs[run] = {"signals": signals, "weights": confounds["weights_first_payoff"].to_numpy(float),
                             "layer_weights": layer_weights, "order_weights": weights_by_order}
            if runs:
                participants[participant] = {"folder": participant_folder, "runs": runs}
    return participants


def network_from_run_subset(runs, run_numbers, layer=None):
    """Compute network C (layer=None) or one payoff layer from a subset of runs, concatenated as in
    task_weighted_networks.py."""
    used_runs = [run for run in run_numbers if run in runs]
    if not used_runs:
        return None
    signals = [runs[run]["signals"] for run in used_runs]
    if layer is None:
        weights = [runs[run]["weights"] for run in used_runs]
    else:
        weights = [runs[run]["layer_weights"].get(layer, np.zeros(len(runs[run]["signals"]))) for run in used_runs]
    return network_from_runs(signals, weights)


def centrality_and_stability(network_folders, out, variant="unsmoothed", session="3t2",
                             regions_path=HERE / "atlas" / "risk_atlas_regions.csv", target_label=TARGET_LABEL, check_saved=False,
                             subjects=None):
    """Run steps 1-4 on the output folders of task_weighted_networks.py and write the tables to out.

    subjects: the participant IDs to analyse (default: all); paper_results.py passes the participants who are not
    excluded for head motion.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    regions = pd.read_csv(regions_path)
    region_names = dict(zip(regions["roi"], regions["label"]))
    n_regions = len(regions)
    target = int(regions.loc[regions["label"].str.contains(target_label, regex=False), "roi"].iloc[0]) - 1
    network_folder, network_suffix, signals_suffix = VARIANTS[variant]

    # ---------------------------------------------------------------- 1. all network files
    network_files = sorted({path for folder in network_folders for path in Path(folder).rglob("*" + network_suffix)})
    if subjects is not None:
        subjects = {str(subject) for subject in subjects}
        network_files = [path for path in network_files if parse_file_name(path.name).get("sub") in subjects]
    centrality = centrality_table(network_files, network_suffix, region_names)
    if centrality.empty:
        raise FileNotFoundError(f"no network files *{network_suffix} found")
    centrality.to_csv(out / "centrality_all_nodes.tsv", sep="\t", index=False)
    print(f"centrality: {len(network_files)} network files, {centrality['subject'].nunique()} participants")
    multiplex = multiplex_table(network_files, network_suffix, region_names)
    multiplex.to_csv(out / "multiplex_centrality.tsv", sep="\t", index=False)
    print(f"multiplex centrality: {multiplex.groupby(['subject', 'session']).ngroups} multiplex networks A")

    # ---------------------------------------------------------------- 2. odd vs even runs
    participants = read_run_signals(network_folders, session, signals_suffix)
    if subjects is not None:
        participants = {participant: data for participant, data in participants.items() if participant in subjects}
    if not participants:
        print("no regional time series found; stability analysis skipped")
        return
    networks = {}
    for participant, data in participants.items():
        layers = sorted({layer for run in data["runs"].values() for layer in run["layer_weights"]})
        for half, run_numbers in HALVES.items():
            networks[(participant, half, "C")] = network_from_run_subset(data["runs"], run_numbers)
            for layer in layers:
                networks[(participant, half, layer)] = network_from_run_subset(data["runs"], run_numbers, layer)
    if check_saved:
        largest_difference = 0.0
        for (participant, half, layer), network in networks.items():
            if half != "all" or network is None:
                continue
            name = "network-C" if layer == "C" else f"network-A_{layer_name(layer)}"
            saved = participants[participant]["folder"] / network_folder / f"sub-{participant}_ses-{session}_{name}{network_suffix}"
            if saved.exists():
                difference = np.nanmax(np.abs(pd.read_csv(saved, index_col=0).to_numpy(float) - network))
                largest_difference = max(largest_difference, float(difference))
        print(f"recomputed vs saved networks: largest absolute difference {largest_difference:.2e}")

    layers = sorted({key[2] for key in networks if key[2] != "C"})
    rows = []
    for participant in participants:
        for half in HALVES:
            network_c = networks.get((participant, half, "C"))
            if network_c is None:
                continue
            measures = {"eigenvector_pos": eigenvector_centrality(network_c)}
            layer_networks = [networks.get((participant, half, layer)) for layer in layers]
            if layers and all(network is not None for network in layer_networks):
                measures["multiplex_eigenvector_mean_single_layer"] = mean_single_layer_eigenvector_centrality(layer_networks)
                measures["multiplex_eigenvector_uniform"] = uniform_eigenvector_centrality(layer_networks)
                measures["multiplex_versatility"] = versatility(layer_networks)
                runs = participants[participant]["runs"]
                runs_of_half = [run for run in HALVES[half] if run in runs]
                shrunk_layers = [shrunk_network_from_runs([runs[run]["signals"] for run in runs_of_half],
                                                          [runs[run]["layer_weights"].get(layer, np.zeros(len(runs[run]["signals"])))
                                                           for run in runs_of_half]) for layer in layers]
                measures["multiplex_eigenvector_uniform_shrunk"] = uniform_eigenvector_centrality(shrunk_layers)
            for name, values in measures.items():
                for region in range(n_regions):
                    rows.append({"subject": participant, "half": half, "measure": name, "roi": region + 1,
                                 "value": float(values[region])})
    measures_by_half = pd.DataFrame(rows)
    measures_by_half.to_csv(out / "measures_by_half.tsv", sep="\t", index=False)
    # add the shrunk uniform centrality (all runs) to the multiplex table; it exists for the session of step 2 only
    shrunk_all = measures_by_half[(measures_by_half["half"] == "all")
                                  & (measures_by_half["measure"] == "multiplex_eigenvector_uniform_shrunk")]
    shrunk_all = shrunk_all.rename(columns={"value": "multiplex_eigenvector_uniform_shrunk"})[
        ["subject", "roi", "multiplex_eigenvector_uniform_shrunk"]].assign(session=session)
    multiplex = multiplex.drop(columns="multiplex_eigenvector_uniform_shrunk", errors="ignore").merge(
        shrunk_all, on=["subject", "session", "roi"], how="left")[MULTIPLEX_COLUMNS]
    multiplex.to_csv(out / "multiplex_centrality.tsv", sep="\t", index=False)

    # ---------------------------------------------------------------- 3. split-half stability
    stability, participant_order = [], sorted(participants)
    for measure in measures_by_half["measure"].unique():
        values = measures_by_half[measures_by_half["measure"] == measure]
        odd = values[values["half"] == "odd"].pivot(index="subject", columns="roi", values="value").reindex(participant_order)
        even = values[values["half"] == "even"].pivot(index="subject", columns="roi", values="value").reindex(participant_order)
        complete = odd.notna().all(axis=1) & even.notna().all(axis=1)
        odd, even = odd[complete], even[complete]
        n_participants = len(odd)
        if n_participants < 3:
            continue
        accuracy, idiff = fingerprint(odd.to_numpy(), even.to_numpy())
        region_icc = [icc_3_1(odd.iloc[:, region].to_numpy(), even.iloc[:, region].to_numpy()) for region in range(n_regions)]
        stability.append({"measure": measure, "participants": n_participants, "identification_accuracy": accuracy,
                          "chance": 1 / n_participants, "idiff": idiff, "target_icc": region_icc[target],
                          "mean_node_icc": float(np.nanmean(region_icc))})
    stability = pd.DataFrame(stability)
    stability.to_csv(out / "stability_by_measure.tsv", sep="\t", index=False)
    print(stability.round(3).to_string(index=False) if not stability.empty
          else "the stability analysis requires at least 3 participants")

    # ---------------------------------------------------------------- 4. network C for each presentation order
    # Choice: computed as network C from all runs, with the first-payoff phases of one presentation order only.
    rows = []
    for participant, data in participants.items():
        runs = sorted(data["runs"])
        for order in sorted({order for run in runs for order in data["runs"][run]["order_weights"]}):
            weights = [data["runs"][run]["order_weights"].get(order, np.zeros(len(data["runs"][run]["signals"])))
                       for run in runs]
            network = network_from_runs([data["runs"][run]["signals"] for run in runs], weights)
            centrality = eigenvector_centrality(network)
            for region in range(n_regions):
                rows.append({"subject": participant, "session": session, "order": order, "roi": region + 1,
                             "label": region_names.get(region + 1, ""), "eigenvector_pos": float(centrality[region])})
    pd.DataFrame(rows).to_csv(out / "centrality_by_order.tsv", sep="\t", index=False)
    print(f"centrality by presentation order: {len(rows) // max(n_regions, 1)} networks")


if __name__ == "__main__":
    args = read_arguments()
    centrality_and_stability(args.networks, args.out, args.variant, args.session, args.regions, args.target_label,
                             args.check_saved)
