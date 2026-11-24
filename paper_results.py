"""All numbers, tables and figures in the manuscript, computed from the participants' networks.

Analysis choices in the order of the Methods section (each is set in the script named on the right; all are listed at
the top of SUMMARY.md):
  risk network  Neurosynth map for "risky" (uniformity test, FDR 0.01), clusters > 50 voxels divided by spectral
                clustering (KNSC-Ncut, number of clusters by eigengap): 23 regions   risk_brain_network_atlas.py
  networks      volumes weighted by the first payoff; session and the four task phases regressed out; 5 + 5
                CompCor components; band-pass 0.01-0.1 Hz; payoff layers: presentation order x safe payoff;
                unsmoothed regional time series (CONN default) and 6 mm smoothed time series (robustness check)
                                                                                     task_weighted_networks.py
  centrality    eigenvector centrality in network C; multiplex network A (the same regions in 12 payoff layers,
                different edges in each layer, no interlayer edges): mean single-layer and uniform multiplex
                eigenvector centrality and versatility; robustness checks: Ledoit-Wolf shrunk layers and absolute
                edge weights;
                split-half stability (odd vs even runs)
                                                                                     network_centrality.py
  prediction    target region: right inferior parietal lobule; outcomes: RNP and nu1; leave-one-participant-out
                cross-validation; 5,000 permutations; at least 8 participants; each region (group-level test with
                Holm correction, and prediction); each presentation order; RNP from nu1 alone and from nu1 plus the
                target region's eigenvector centrality; the target region's multiplex centralities; comparison
                with the baseline; prediction in the participants whose hub is the target region
                                                                                     predictive_analysis.py
  behaviour     RNP and nu1 from the choice models of de Hollander et al.              choice_models/

Steps:
1. unpack each zip file into OUT/unpacked/<zip name>/ (existing folders are reused)
2. inventory (OUT/inventory.tsv): status, runs and network files per version for each participant; the script stops
   if a participant appears twice
3. for unsmoothed and 6 mm smoothed regional time series: report (quality, group network, target-region figures),
   centrality and stability (network_centrality.py), prediction (predictive_analysis.py; at least 8 participants)
4. figures 1-3 of the paper (OUT/figures/)
5. OUT/SUMMARY.md with the main results. Failed steps are reported there and do not stop the remaining steps.

usage: python paper_results.py NETWORKS [NETWORKS ...] --out OUT [--session 3t2] [--permutations 5000]
NETWORKS: output folders of task_weighted_networks.py, zip files of these folders (task_weighted_networks_notebook.py),
or folders containing such zip files.
"""
import argparse
import json
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon

from network_centrality import VARIANTS, centrality_and_stability, eigenvector_centrality
from predictive_analysis import predict_behaviour
from task_weighted_networks import (DUMMY_VOLUMES, LAYER_FACTORS, NUISANCE_EVENTS, TARGET_LABEL, WEIGHTING_EVENT,
                                    parse_file_name)

HERE = Path(__file__).resolve().parent
ANALYSIS_SETTINGS = {"weighting event": WEIGHTING_EVENT, "nuisance events": ", ".join(NUISANCE_EVENTS),
                     "payoff magnitude networks": " x ".join(LAYER_FACTORS), "dummy volumes": DUMMY_VOLUMES,
                     "target": TARGET_LABEL, "outcomes": "RNP, nu1"}
# names of the measures and prediction models in SUMMARY.md, in the order of its tables
MEASURE_NAMES = {"eigenvector_pos": "eigenvector centrality",
                 "multiplex_eigenvector_mean_single_layer": "mean single-layer multiplex eigenvector centrality",
                 "multiplex_eigenvector_uniform": "uniform multiplex eigenvector centrality",
                 "multiplex_versatility": "versatility (multilayer eigenvector centrality)",
                 "multiplex_eigenvector_uniform_shrunk": "uniform multiplex eigenvector centrality, shrunk layers"}
EIGENVECTOR_DEFINITIONS = {"eigenvector_pos": "1. eigenvector centrality, network C",
                           "multiplex_eigenvector_mean_single_layer": "2. mean single-layer multiplex eigenvector centrality, A",
                           "multiplex_eigenvector_uniform": "3. uniform multiplex eigenvector centrality, A",
                           "multiplex_versatility": "4. versatility (multilayer eigenvector centrality), A",
                           "multiplex_eigenvector_uniform_shrunk": "robustness: 3 with Ledoit-Wolf shrunk layers, A",
                           "multiplex_eigenvector_uniform_absolute": "robustness: 3 with absolute edge weights, A",
                           "eigenvector_abs": "robustness: 1 with absolute edge weights, C"}
PREDICTION_MODELS = {"eigenvector_pos": "eigenvector centrality",
                     "eigenvector_abs": "robustness: eigenvector centrality of the absolute edge weights"}
SAFE_PAYOFFS = [5, 7, 10, 14, 20, 28]                                         # safe payoffs (CHF) of the payoff layers
ORDERS = (("safefirst", "safe first"), ("riskyfirst", "risky first"))      # presentation orders and their labels
RISK_NEUTRAL_RNP = 0.55                                                     # RNP of a risk-neutral decision maker
# Exclusion for head motion: the stringent criteria of Parkes et al. (2018), on the framewise displacement (FD) of all
# scans of the session: mean FD > 0.25 mm, more than 20% of scans with FD > 0.2 mm, or any FD > 5 mm
MOTION_MEAN_FD, MOTION_FD_THRESHOLD, MOTION_SHARE, MOTION_MAX_FD = 0.25, 0.2, 0.20, 5.0

# colours (light background)
INK, INK_LIGHT, MUTED, GRID, LIGHT_GREY, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, RED, MIDDLE_GREY = "#2a78d6", "#eb6834", "#e34948", "#f0efec"
BLUE_GREY_RED = LinearSegmentedColormap.from_list("blue_gray_red", ["#184f95", BLUE, MIDDLE_GREY, RED, "#a32b2b"])


def read_network(path):
    """Read a network table (regions x regions)."""
    return pd.read_csv(path, index_col=0).to_numpy(float)


def markdown_table(table, formats=None):
    """Format a table as Markdown; missing values are empty cells."""
    formats = formats or {}
    columns = list(table.columns)
    text = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns) + "\n"
    for _, row in table.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if not isinstance(value, str) and pd.isna(value):
                cells.append("")
            elif column in formats:
                cells.append(formats[column].format(value))
            else:
                cells.append(str(value))
        text += "| " + " | ".join(cells) + " |\n"
    return text


def read_table(path):
    """Read a table written by a previous step, or return None if the table is missing or empty."""
    if not path.exists():
        return None
    try:
        table = pd.read_csv(path, sep="\t", dtype={"participant": str})
    except pd.errors.EmptyDataError:
        return None
    return None if table.empty else table


def motion_summary(participant_folder):
    """Framewise displacement over all scans of the session and the exclusion of Parkes et al. (2018, stringent).

    FD is read from the confound tables of the runs (runs/*_confounds.tsv, column framewise_displacement); scans
    without a value (the first scan of a run) are left out.
    """
    tables = sorted(Path(participant_folder).glob("runs/*_confounds.tsv"))
    values = [pd.read_csv(path, sep="\t")["framewise_displacement"].to_numpy(float) for path in tables]
    fd = np.concatenate(values) if values else np.array([])
    fd = fd[np.isfinite(fd)]
    if not len(fd):
        return {"mean_fd_mm": np.nan, "share_fd_above_0.2": np.nan, "max_fd_mm": np.nan, "excluded": False}
    summary = {"mean_fd_mm": float(fd.mean()), "share_fd_above_0.2": float(np.mean(fd > MOTION_FD_THRESHOLD)),
               "max_fd_mm": float(fd.max())}
    summary["excluded"] = bool(summary["mean_fd_mm"] > MOTION_MEAN_FD or summary["share_fd_above_0.2"] > MOTION_SHARE
                               or summary["max_fd_mm"] > MOTION_MAX_FD)
    return summary


# ================================================================ report: quality and figures
def network_report(network_folders, regions_path, fits_path, out, variant="unsmoothed", session="3t2", target_label=TARGET_LABEL,
                   subjects=None):
    """Write the report: quality, group network, regional eigenvector centrality and target-region figures.

    The quality table lists all participants; the other parts use the participants in subjects (default: all), which
    paper_results.py sets to those not excluded for head motion.

    Target-region figures: target-region eigenvector centrality against RNP, across runs (B networks) and across payoff
    layers (A networks).
    """
    plt.rcdefaults()          # each figure starts from the matplotlib defaults
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": LIGHT_GREY, "axes.labelcolor": INK_LIGHT, "xtick.color": INK_LIGHT,
                         "ytick.color": INK_LIGHT, "text.color": INK, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                         "lines.linewidth": 2})

    regions = pd.read_csv(regions_path)
    n_regions = len(regions)
    region_names = list(regions["label"])
    target = int(regions.loc[regions["label"].str.contains(target_label, regex=False), "roi"].iloc[0]) - 1
    network_folder, network_suffix, _ = VARIANTS[variant]
    other_variant = "unsmoothed" if variant == "smoothed" else "smoothed"
    other_folder, other_suffix, _ = VARIANTS[other_variant]
    category, level = LAYER_FACTORS

    # ---------------------------------------------------------------- networks of all participants
    participants = {}
    for folder in network_folders:
        for participant_folder in sorted(Path(folder).rglob(f"sub-*_ses-{session}")):
            network_c_path = participant_folder / network_folder / f"{participant_folder.name}_network-C{network_suffix}"
            if not (participant_folder.is_dir() and network_c_path.exists()):
                continue
            data = {"folder": participant_folder, "C": read_network(network_c_path), "B": {}, "A": {}}
            for path in (participant_folder / network_folder).glob(f"*network-B_run-*{network_suffix}"):
                data["B"][int(parse_file_name(path.name)["run"])] = read_network(path)
            for path in (participant_folder / network_folder).glob(f"*network-A_*{network_suffix}"):
                parts = parse_file_name(path.name)
                data["A"][(parts[category], int(parts[level]))] = read_network(path)
            other_path = participant_folder / other_folder / f"{participant_folder.name}_network-C{other_suffix}"
            data["C_other"] = read_network(other_path) if other_path.exists() else None
            summary_path = participant_folder / "subject_summary.json"
            data["summary"] = json.load(open(summary_path)) if summary_path.exists() else {}
            participants[parse_file_name(participant_folder.name)["sub"]] = data
    if not participants:
        raise FileNotFoundError(f"no participants with {network_folder}/ network C found")
    labels = sorted(participants)
    n_participants = len(labels)
    print(f"{n_participants} participants: {labels} ({variant} regional time series)")

    fits = pd.read_csv(fits_path, sep="\t", dtype={"subject": str, "session": str})
    fits["subject"] = fits["subject"].str.zfill(2)
    fits = fits[fits["session"] == session].set_index("subject")

    # ---------------------------------------------------------------- 1. quality table
    rows = []
    for label in labels:
        summary = participants[label]["summary"]
        runs = summary.get("runs", {})
        anatomy = summary.get("anatomy", {})
        tissue = anatomy.get("tissue", {})
        rows.append({
            "participant": label, "runs": len(runs),
            "csf_fraction": tissue.get("csf", {}).get("fraction"), "gm_fraction": tissue.get("gm", {}).get("fraction"),
            "wm_fraction": tissue.get("wm", {}).get("fraction"),
            "wm_mask_voxels": anatomy.get("wm_mask_voxels"), "csf_mask_voxels": anatomy.get("csf_mask_voxels"),
            **{key: value for key, value in motion_summary(participants[label]["folder"]).items() if key != "excluded"},
            "median_tsnr": np.median([run["median_tsnr"] for run in runs.values()]) if runs else np.nan,
            "min_roi_coverage": min([run["min_roi_coverage"] for run in runs.values()]) if runs else np.nan,
            "distortion_corrected_runs": sum(bool(run.get("distortion_correction")) for run in runs.values()),
            "hours": round(summary.get("minutes", np.nan) / 60, 2) if summary.get("minutes") else np.nan,
            "rnp_3t": fits["rnp_mean"].get(label, np.nan)})
    quality = pd.DataFrame(rows)
    flags = []
    for _, row in quality.iterrows():
        problems = []
        if row["runs"] < 8:
            problems.append(f"{row['runs']} of 8 runs")
        if row["mean_fd_mm"] > MOTION_MEAN_FD:
            problems.append(f"mean FD > {MOTION_MEAN_FD} mm")
        if row["share_fd_above_0.2"] > MOTION_SHARE:
            problems.append(f"more than {MOTION_SHARE:.0%} of scans with FD > {MOTION_FD_THRESHOLD} mm")
        if row["max_fd_mm"] > MOTION_MAX_FD:
            problems.append(f"FD > {MOTION_MAX_FD} mm")
        if row["median_tsnr"] < 30:
            problems.append("tSNR < 30")
        if row["min_roi_coverage"] < 0.8:
            problems.append("an ROI less than 80% inside the EPI")
        if row["runs"] and row["distortion_corrected_runs"] < row["runs"]:
            problems.append("runs without distortion correction")
        flags.append("; ".join(problems) or "ok")
    quality["flags"] = flags
    quality.to_csv(out / "quality.tsv", sep="\t", index=False)
    if subjects is not None:
        labels = [label for label in labels if label in {str(subject) for subject in subjects}]
        participants = {label: participants[label] for label in labels}
        n_participants = len(labels)
        print(f"{n_participants} participants analysed after the exclusion for head motion")

    # ---------------------------------------------------------------- 2. group network C
    mean_network_c = np.nanmean(np.array([participants[label]["C"] for label in labels]), axis=0)
    figure, axis = plt.subplots(figsize=(8.6, 7.4))
    colour_limit = np.nanmax(np.abs(mean_network_c))
    image = axis.imshow(mean_network_c, cmap=BLUE_GREY_RED, vmin=-colour_limit, vmax=colour_limit)
    axis.set_xticks(range(n_regions)); axis.set_yticks(range(n_regions))
    axis.set_xticklabels(region_names, rotation=90, fontsize=8); axis.set_yticklabels(region_names, fontsize=8)
    for tick in (axis.get_xticklabels()[target], axis.get_yticklabels()[target]):
        tick.set_fontweight("bold"); tick.set_color(BLUE)
    axis.grid(False)
    colour_bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02); colour_bar.set_label("mean Fisher z")
    axis.set_title(f"Network C: mean over {n_participants} participants ({variant} regional time series); target region in blue",
                   fontsize=11, loc="left")
    figure.tight_layout(); figure.savefig(out / "fig1_group_network_C.png", dpi=160); plt.close(figure)

    # regional eigenvector centrality
    centrality = np.array([eigenvector_centrality(participants[label]["C"]) for label in labels])   # participants x regions
    ranking = np.argsort(centrality.mean(0))
    figure, axis = plt.subplots(figsize=(7.2, 7.0))
    for rank, region in enumerate(ranking):
        colour = BLUE if region == target else MUTED
        axis.scatter(centrality[:, region], np.full(n_participants, rank), s=14, color=colour, alpha=0.55, zorder=2)
        axis.plot([centrality[:, region].mean()] * 2, [rank - 0.35, rank + 0.35], color=colour if region == target else INK_LIGHT,
                  lw=2.5, zorder=3)
    axis.set_yticks(range(n_regions)); axis.set_yticklabels([region_names[region] for region in ranking], fontsize=8)
    axis.get_yticklabels()[list(ranking).index(target)].set_color(BLUE)
    axis.get_yticklabels()[list(ranking).index(target)].set_fontweight("bold")
    axis.set_xlabel("eigenvector centrality in network C")
    axis.set_title(f"Eigenvector centrality: participants (dots) and mean (bar), n = {n_participants}", fontsize=11, loc="left")
    figure.tight_layout(); figure.savefig(out / "fig2_node_centrality.png", dpi=160); plt.close(figure)

    # ---------------------------------------------------------------- 3. target region and risk attitude
    upper_triangle = np.triu_indices(n_regions, 1)
    target_table = pd.DataFrame({"participant": labels, "target_eigenvector_C": centrality[:, target],
                               "target_rank_of_23": [int((centrality[index] > centrality[index, target]).sum() + 1)
                                                   for index in range(n_participants)],
                               "network_mean_z": [np.nanmean(participants[label]["C"][upper_triangle]) for label in labels],
                               "rnp_3t": [fits["rnp_mean"].get(label, np.nan) for label in labels]})
    target_table["nu1_3t"] = [fits["nu1"].get(label, np.nan) for label in labels]
    target_table.to_csv(out / "target_by_participant.tsv", sep="\t", index=False)
    with_rnp = target_table["rnp_3t"].notna()
    rho = stats.spearmanr(target_table.loc[with_rnp, "target_eigenvector_C"], target_table.loc[with_rnp, "rnp_3t"])[0] \
        if with_rnp.sum() >= 4 else np.nan
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    axis.axhline(RISK_NEUTRAL_RNP, color=LIGHT_GREY, lw=1, ls="--", zorder=1)
    axis.text(0.01, 0.555, "risk neutral (0.55)", color=MUTED, fontsize=8, va="bottom", transform=axis.get_yaxis_transform())
    axis.scatter(target_table["target_eigenvector_C"], target_table["rnp_3t"], s=48, color=BLUE, zorder=3)
    for _, row in target_table.iterrows():
        axis.annotate(row["participant"], (row["target_eigenvector_C"], row["rnp_3t"]), xytext=(5, 3), textcoords="offset points",
                      fontsize=8, color=INK_LIGHT)
    axis.set_xlabel(f"target-region eigenvector centrality in network C ({region_names[target]})")
    axis.set_ylabel("risk attitude, RNP (3T)")
    axis.set_title(f"Target-region centrality and risk attitude, n = {int(with_rnp.sum())}\nSpearman rho = {rho:.2f}",
                   fontsize=11, loc="left")
    figure.tight_layout(); figure.savefig(out / "fig3_target_vs_rnp.png", dpi=160); plt.close(figure)

    # ---------------------------------------------------------------- 4. runs (B networks)
    rows = []
    for label in labels:
        for run, network in sorted(participants[label]["B"].items()):
            rows.append({"participant": label, "run": run, "target_eigenvector": eigenvector_centrality(network)[target],
                         "network_mean_z": np.nanmean(network[upper_triangle])})
    target_by_run = pd.DataFrame(rows, columns=["participant", "run", "target_eigenvector", "network_mean_z"])
    target_by_run.to_csv(out / "target_by_run.tsv", sep="\t", index=False)
    if not target_by_run.empty:
        figure, axis = plt.subplots(figsize=(6.4, 4.4))
        for label, runs in target_by_run.groupby("participant"):
            axis.plot(runs["run"], runs["target_eigenvector"], color=MUTED, lw=1, alpha=0.6, marker="o", ms=3)
        run_means = target_by_run.groupby("run")["target_eigenvector"].agg(["mean", "sem"])
        axis.errorbar(run_means.index, run_means["mean"], yerr=run_means["sem"], color=BLUE, lw=2.5, marker="o", ms=6,
                      capsize=3, zorder=4)
        axis.set_xticks(range(1, 9)); axis.set_xlabel("run (about 6 min each)"); axis.set_ylabel("target-region eigenvector centrality (network B)")
        axis.set_title(f"Target-region centrality over runs: participants (grey), mean and SE (blue), "
                       f"n = {target_by_run['participant'].nunique()}", fontsize=10, loc="left")
        figure.tight_layout(); figure.savefig(out / "fig4_target_over_runs.png", dpi=160); plt.close(figure)

    # ---------------------------------------------------------------- 5. payoff layers (A networks)
    rows = []
    for label in labels:
        for (order, safe_payoff), network in sorted(participants[label]["A"].items()):
            rows.append({"participant": label, "order": order, "safe": safe_payoff,
                         "target_eigenvector": eigenvector_centrality(network)[target]})
    target_by_layer = pd.DataFrame(rows, columns=["participant", "order", "safe", "target_eigenvector"])
    target_by_layer.to_csv(out / "target_by_payoff_layer.tsv", sep="\t", index=False)
    if not target_by_layer.empty:
        figure, axis = plt.subplots(figsize=(6.4, 4.4))
        for order, colour, name in (("safefirst", BLUE, "safe option first"), ("riskyfirst", ORANGE, "risky option first")):
            layer_means = target_by_layer[target_by_layer["order"] == order].groupby("safe")["target_eigenvector"].agg(["mean", "sem"])
            if layer_means.empty:
                continue
            axis.errorbar(layer_means.index, layer_means["mean"], yerr=layer_means["sem"], color=colour, lw=2, marker="o",
                          ms=6, capsize=3, label=name)
            axis.annotate(name, (layer_means.index[-1], layer_means["mean"].iloc[-1]), xytext=(6, 0),
                          textcoords="offset points", color=colour, fontsize=9, va="center")
        axis.set_xscale("log"); axis.set_xticks(SAFE_PAYOFFS); axis.set_xticklabels([str(payoff) for payoff in SAFE_PAYOFFS])
        axis.minorticks_off()
        axis.set_xlabel("safe payoff (log scale)"); axis.set_ylabel("target-region eigenvector centrality (payoff layer)")
        axis.legend(frameon=False, fontsize=9, loc="lower left")
        axis.set_title(f"Target-region centrality by payoff and order: mean and SE, n = {target_by_layer['participant'].nunique()}",
                       fontsize=10, loc="left")
        figure.tight_layout(); figure.savefig(out / "fig5_target_by_payoff.png", dpi=160); plt.close(figure)

    # ---------------------------------------------------------------- 6. smoothed vs unsmoothed networks
    pairs = [(label, participants[label]["C"], participants[label]["C_other"]) for label in labels
             if participants[label]["C_other"] is not None]
    if pairs:
        comparison = pd.DataFrame([{"participant": label,
                                    f"target_eigenvector_{variant}": eigenvector_centrality(network)[target],
                                    "target_eigenvector_other": eigenvector_centrality(other_network)[target],
                                    "edge_pattern_r": np.corrcoef(network[upper_triangle], other_network[upper_triangle])[0, 1]}
                                   for label, network, other_network in pairs])
        comparison.to_csv(out / "smoothed_vs_unsmoothed.tsv", sep="\t", index=False)

    # ---------------------------------------------------------------- report file
    report = [f"# Network report: {n_participants} participant{'s' if n_participants != 1 else ''}, session {session}, "
              f"{variant} regional time series", "",
              "Quality, group network and target-region eigenvector centrality. Stability and prediction are computed by network_centrality.py "
              "and predictive_analysis.py.", "", "## Quality", "",
              markdown_table(quality[["participant", "runs", "wm_fraction", "wm_mask_voxels", "csf_mask_voxels", "mean_fd_mm",
                                      "median_tsnr", "min_roi_coverage", "hours", "flags"]],
                             {"wm_fraction": "{:.2f}", "mean_fd_mm": "{:.2f}", "median_tsnr": "{:.1f}",
                              "min_roi_coverage": "{:.2f}", "hours": "{:.2f}"}),
              "## Target region", "",
              f"Target region: {region_names[target]} (region {target + 1}). Mean rank among the 23 regions by eigenvector centrality in "
              f"network C: {target_table['target_rank_of_23'].mean():.1f}. Spearman correlation with RNP "
              f"(n = {int(with_rnp.sum())}): rho = {rho:.2f}.", "",
              markdown_table(target_table, {"target_eigenvector_C": "{:.3f}", "network_mean_z": "{:.3f}", "rnp_3t": "{:.2f}",
                                          "nu1_3t": "{:.3f}"}),
              "## Figures", "",
              "- fig1_group_network_C.png: mean network C", "- fig2_node_centrality.png: eigenvector centrality of each region, participants and mean",
              "- fig3_target_vs_rnp.png: target-region eigenvector centrality and risk attitude (RNP)",
              "- fig4_target_over_runs.png: target-region eigenvector centrality over the 8 runs (temporal networks B)",
              "- fig5_target_by_payoff.png: target-region eigenvector centrality by safe payoff and order (payoff layers A)"]
    if pairs:
        report += ["", "## Smoothed and unsmoothed regional time series", "",
                   markdown_table(comparison, {f"target_eigenvector_{variant}": "{:.3f}", "target_eigenvector_other": "{:.3f}",
                                               "edge_pattern_r": "{:.2f}"})]
    (out / "report.md").write_text("\n".join(report) + "\n")
    print((out / "report.md").read_text()[:3000])


# ================================================================ paper figures
def figure_regions(regions_path, out_path, target_label=TARGET_LABEL):
    """Fig. 1: risk network regions, numbered at their centres on a glass brain (target region in blue), with a legend."""
    from nilearn import plotting
    from nilearn.plotting.displays._axes import coords_3d_to_2d
    plt.rcdefaults()
    regions = pd.read_csv(regions_path)
    target = int(regions.loc[regions["label"] == target_label, "roi"].iloc[0])
    centres = regions[["center_x", "center_y", "center_z"]].to_numpy(float)
    is_target = (regions["roi"] == target).to_numpy()

    figure = plt.figure(figsize=(10.0, 6.6))
    brain_axis = figure.add_axes([0.0, 0.40, 1.0, 0.60])
    display = plotting.plot_glass_brain(None, display_mode="lyrz", axes=brain_axis, annotate=True)
    display.add_markers(centres[~is_target], marker_color=ORANGE, marker_size=115, alpha=0.95, edgecolors="white", linewidths=0.8)
    display.add_markers(centres[is_target], marker_color=BLUE, marker_size=200, alpha=1.0, edgecolors="white", linewidths=1.2)
    for direction, view in display.axes.items():
        positions = coords_3d_to_2d(centres, direction)
        for (horizontal, vertical), (x_mm, _, _), region, target_region in zip(positions, centres, regions["roi"], is_target):
            if (direction == "l" and x_mm > 0) or (direction == "r" and x_mm < 0):
                continue                            # lateral views show only the labels of their own hemisphere
            view.ax.text(horizontal, vertical, str(region), fontsize=5.6 if not target_region else 6.6, color="white",
                         ha="center", va="center", fontweight="bold", zorder=1001)

    legend_axis = figure.add_axes([0.04, 0.0, 0.94, 0.38]); legend_axis.axis("off")
    names = [f"{region}  {label}" for region, label in zip(regions["roi"], regions["label"])]
    n_columns = 3
    per_column = int(np.ceil(len(names) / n_columns))
    for index, name in enumerate(names):
        column, row = divmod(index, per_column)
        target_region = regions["roi"].iloc[index] == target
        legend_axis.text(column / n_columns, 1 - (row + 0.5) / per_column, name + ("  (target)" if target_region else ""),
                         fontsize=9, color=BLUE if target_region else INK_LIGHT, fontweight="bold" if target_region else "normal",
                         va="center", transform=legend_axis.transAxes)
    figure.savefig(out_path, dpi=300, facecolor="white")
    print(f"written {out_path}; target region {target}")


def figure_networks(network_folders, regions_path, out_path, session="3t2", target_label=TARGET_LABEL, subjects=None):
    """Fig. 2: mean networks across participants (those in subjects; default: all): a, network C; b, the 12 payoff
    layers (A); c, the 8 runs (B).

    The colour limit is the 98th percentile of |mean z| across all panels, so that a few extreme edges do not
    determine the colour scale.
    """
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
                         "text.color": INK, "xtick.color": INK_LIGHT, "ytick.color": INK_LIGHT, "figure.facecolor": "white"})
    regions = pd.read_csv(regions_path)
    region_names = list(regions["label"])
    target = int(regions.loc[regions["label"] == target_label, "roi"].iloc[0]) - 1
    n_regions = len(region_names)
    network_folder, network_suffix, _ = VARIANTS["unsmoothed"]

    participant_folders = sorted(participant_folder for folder in network_folders
                                 for participant_folder in Path(folder).rglob(f"sub-*_ses-{session}")
                                 if (participant_folder / network_folder).is_dir()
                                 and (subjects is None or parse_file_name(participant_folder.name)["sub"] in set(subjects)))

    def mean_network(pattern):
        """Return the mean network across participants for one file pattern and the number of networks."""
        networks = [read_network(path) for participant_folder in participant_folders
                    for path in (participant_folder / network_folder).glob(pattern)]
        if not networks:                       # empty panel, e.g. for a run that is missing in all participants
            return np.full((n_regions, n_regions), np.nan), 0
        mean = np.nanmean(np.array(networks), axis=0)
        np.fill_diagonal(mean, np.nan)
        return mean, len(networks)

    network_c, n_participants = mean_network(f"*_network-C{network_suffix}")
    payoff_layers = {(order, payoff): mean_network(f"*_network-A_order-{order}_safe-{payoff:02d}{network_suffix}")[0]
                     for order, _ in ORDERS for payoff in SAFE_PAYOFFS}
    run_networks = {run: mean_network(f"*_network-B_run-{run}{network_suffix}")[0] for run in range(1, 9)}
    all_edges = np.concatenate([network_c.ravel()] + [network.ravel() for network in payoff_layers.values()] +
                               [network.ravel() for network in run_networks.values()])
    colour_limit = np.nanpercentile(np.abs(all_edges), 98)

    figure = plt.figure(figsize=(7.2, 5.9))
    # a: network C
    axis_c = figure.add_axes([0.12, 0.40, 0.36, 0.54])
    image = axis_c.imshow(network_c, cmap=BLUE_GREY_RED, vmin=-colour_limit, vmax=colour_limit)
    axis_c.set_xticks(range(n_regions)); axis_c.set_yticks(range(n_regions))
    axis_c.set_xticklabels(region_names, rotation=90, fontsize=5.2); axis_c.set_yticklabels(region_names, fontsize=5.2)
    for tick in (axis_c.get_xticklabels()[target], axis_c.get_yticklabels()[target]):
        tick.set_color(BLUE); tick.set_fontweight("bold")
    axis_c.set_title("a  Network C: all first payoffs", pad=4)
    for spine in axis_c.spines.values():
        spine.set_visible(False)

    def small_network(axis, network, title=None):
        """Draw a small network panel with the target region row and column marked."""
        axis.imshow(network, cmap=BLUE_GREY_RED, vmin=-colour_limit, vmax=colour_limit)
        axis.set_xticks([]); axis.set_yticks([])
        axis.plot([-0.5, n_regions - 0.5], [target, target], color=BLUE, lw=0.4, alpha=0.8)
        axis.plot([target, target], [-0.5, n_regions - 0.5], color=BLUE, lw=0.4, alpha=0.8)
        axis.set_xlim(-0.5, n_regions - 0.5); axis.set_ylim(n_regions - 0.5, -0.5)
        for spine in axis.spines.values():
            spine.set_color(LIGHT_GREY); spine.set_linewidth(0.5)
        if title:
            axis.set_title(title, fontsize=7, fontweight="normal", loc="center", pad=2)

    # b: payoff layers, 2 rows (order) x 6 columns (safe payoff)
    left, bottom, width, gap = 0.585, 0.56, 0.055, 0.006
    height = width * 7.2 / 5.9                              # square panels on the 7.2 x 5.9 inch figure
    figure.text(0.52, 0.80, "b  Payoff magnitude networks A", fontsize=8.5, fontweight="bold")
    for row, (order, order_name) in enumerate(ORDERS):
        for column, payoff in enumerate(SAFE_PAYOFFS):
            axis = figure.add_axes([left + column * (width + gap), bottom + (1 - row) * (height + 0.035), width, height])
            small_network(axis, payoff_layers[(order, payoff)], title=str(payoff) if row == 0 else None)
        figure.text(left - 0.008, bottom + (1 - row) * (height + 0.035) + height / 2, order_name,
                    fontsize=7, va="center", ha="right", color=INK_LIGHT)

    # c: run networks, 1 row x 8 columns
    figure.text(0.13 - 0.1, 0.235, "c  Temporal networks B: one per run", fontsize=8.5, fontweight="bold")
    run_width = 0.088
    for index, run in enumerate(range(1, 9)):
        axis = figure.add_axes([0.04 + index * (run_width + 0.012), 0.07, run_width, run_width * 7.2 / 5.9])
        small_network(axis, run_networks[run], title=f"run {run}")
    colour_bar_axis = figure.add_axes([0.905, 0.07, 0.012, 0.22])
    colour_bar = figure.colorbar(image, cax=colour_bar_axis)
    colour_bar.set_label("mean Fisher z", fontsize=7); colour_bar.ax.tick_params(labelsize=6)
    figure.savefig(out_path, dpi=300)
    print(f"written {out_path}: mean of {n_participants} participants; colour limit {colour_limit:.2f}")


def figure_measures(out_path):
    """Fig. 3: eigenvector centrality and the multiplex network A on small example networks (illustrative values)."""
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
                         "text.color": INK, "axes.labelcolor": INK_LIGHT, "xtick.color": INK_LIGHT,
                         "ytick.color": INK_LIGHT, "axes.edgecolor": LIGHT_GREY, "figure.facecolor": "white"})

    def draw(axis, positions, edges, sizes, colours, labels=None):
        """Draw an example network; positive edges are solid and negative edges dashed."""
        for node_a, node_b, weight in edges:
            axis.plot([positions[node_a][0], positions[node_b][0]], [positions[node_a][1], positions[node_b][1]],
                      color=INK_LIGHT if weight > 0 else ORANGE, lw=0.6 + 2.2 * abs(weight), alpha=0.55,
                      ls="-" if weight > 0 else "--", zorder=1)
        for node, (x, y) in positions.items():
            axis.scatter(x, y, s=sizes[node], color=colours[node], edgecolors="white", linewidths=1.0, zorder=3)
            if labels and node in labels:
                axis.text(x, y - 0.17, labels[node], ha="center", va="top", fontsize=7.5, color=INK_LIGHT, zorder=4)
        axis.set_xticks([]); axis.set_yticks([]); axis.set_aspect("equal")
        for spine in axis.spines.values():
            spine.set_visible(False)

    figure, axes = plt.subplots(1, 2, figsize=(5.2, 3.0))
    figure.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.04, wspace=0.25)

    # a eigenvector centrality: the same links in number and weight, different neighbours
    axis = axes[0]
    positions = {0: (-0.55, 0.0), 1: (0.6, 0.0), 2: (-1.15, 0.75), 3: (-1.15, -0.75), 4: (1.1, 0.8), 5: (1.25, -0.65),
                 6: (0.55, 1.05), 7: (1.55, 0.2)}
    edges = [(0, 2, 0.6), (0, 3, 0.6), (1, 4, 0.6), (1, 5, 0.6), (4, 6, 0.6), (4, 7, 0.6), (5, 7, 0.6), (6, 7, 0.5), (0, 1, 0.3)]
    draw(axis, positions, edges, {0: 330, 1: 330, 2: 90, 3: 90, 4: 200, 5: 170, 6: 120, 7: 200},
         {0: LIGHT_GREY, 1: BLUE, 2: LIGHT_GREY, 3: LIGHT_GREY, 4: LIGHT_GREY, 5: LIGHT_GREY, 6: LIGHT_GREY, 7: LIGHT_GREY})
    axis.set_title("a  Eigenvector centrality")
    axis.text(0.2, -1.2, "same links in number and weight;\nblue node coupled with well-coupled regions", ha="center",
              fontsize=7.5, color=INK_LIGHT)
    axis.set_xlim(-1.6, 1.9); axis.set_ylim(-1.45, 1.3)

    # b multiplex network A: the same regions in each payoff layer, different links in each, no links between layers
    axis = axes[1]
    random_generator = np.random.default_rng(3)
    layer_positions = {0: (0.0, 0.0), 1: (-0.55, 0.25), 2: (0.55, 0.2), 3: (-0.3, -0.3), 4: (0.35, -0.3)}
    layer_links = {"5 CHF": [(0, 1), (0, 2), (1, 3), (2, 4), (0, 4)],
                   "14 CHF": [(0, 1), (0, 3), (0, 4), (1, 3), (2, 4), (1, 2)],
                   "28 CHF": [(0, 2), (0, 3), (0, 4), (3, 4), (1, 3)]}
    for layer_offset, layer_label in [(1.45, "5 CHF"), (0.0, "14 CHF"), (-1.45, "28 CHF")]:
        plane = np.array([[-0.95, -0.55], [0.75, -0.55], [1.0, 0.55], [-0.7, 0.55]]) + [0, layer_offset]
        axis.add_patch(Polygon(plane, closed=True, color=GRID, alpha=0.55, zorder=0))
        axis.text(1.08, layer_offset, layer_label, fontsize=7.5, color=INK_LIGHT, va="center")
        for node_a, node_b in layer_links[layer_label]:
            weight = random_generator.uniform(0.2, 0.9)
            axis.plot([layer_positions[node_a][0], layer_positions[node_b][0]],
                      [layer_positions[node_a][1] + layer_offset, layer_positions[node_b][1] + layer_offset],
                      color=INK_LIGHT, lw=0.5 + 1.6 * weight, alpha=0.5, zorder=1)
        for node, (x, y) in layer_positions.items():
            axis.scatter(x, y + layer_offset, s=150 if node == 0 else 45, color=BLUE if node == 0 else LIGHT_GREY,
                         edgecolors="white", lw=0.8, zorder=3)
    axis.set_title("b  Multiplex network A")
    axis.text(0.2, -2.75, "same regions, different links\nin each payoff magnitude network", ha="center", fontsize=7.5,
              color=INK_LIGHT)
    axis.set_xlim(-1.1, 1.7); axis.set_ylim(-3.2, 2.2); axis.set_xticks([]); axis.set_yticks([]); axis.set_aspect("equal")
    for spine in axis.spines.values():
        spine.set_visible(False)

    figure.savefig(out_path, dpi=300)
    print(f"written {out_path}")


# ================================================================ steps 1-2: unpack and inventory
def unpack(network_inputs, unpacked_folder):
    """Unpack the zip files, given directly or found in the given folders, into unpacked_folder.

    Returns the zip files and the folders without zip files, which are used directly.
    """
    unpacked_folder.mkdir(parents=True, exist_ok=True)
    zip_files, folders = [], []
    for item in map(Path, network_inputs):
        if item.is_file() and item.suffix == ".zip":
            zip_files.append(item)
        elif item.is_dir():
            zips_inside = sorted(item.glob("*networks*.zip"))
            zip_files += zips_inside
            if not zips_inside:
                folders.append(item)
        else:
            raise FileNotFoundError(f"not a zip file or folder: {item}")
    for zip_path in zip_files:
        target = unpacked_folder / zip_path.stem
        if target.exists():
            print(f"{zip_path.name}: already unpacked, reused")
            continue
        with zipfile.ZipFile(zip_path) as archive:
            unsafe = [name for name in archive.namelist() if name.startswith("/") or ".." in Path(name).parts]
            if unsafe:
                raise ValueError(f"{zip_path.name}: unexpected paths in the zip file: {unsafe[:3]}")
            archive.extractall(target)
        print(f"{zip_path.name}: unpacked")
    return zip_files, folders


def network_counts(participant_folder):
    """Count the runs with denoised time series and the network files of each version in a participant folder."""
    counts = {"runs": len(list((participant_folder / "runs").glob("*_run-*_roi_timeseries_unsmoothed_cleaned.tsv")))}
    for variant, (network_folder, network_suffix, _) in VARIANTS.items():
        names = [path.name for path in (participant_folder / network_folder).glob("*" + network_suffix)]
        counts[f"B_{variant}"] = sum("network-B_" in name for name in names)
        counts[f"A_{variant}"] = sum("network-A_" in name for name in names)
        counts[f"C_{variant}"] = int(any("network-C" in name for name in names))
    counts["qc_images"] = len(list((participant_folder / "qc").glob("*.png")))
    return counts


def inventory(roots, session):
    """List all participants with the source of their networks, their status, and the numbers of runs and networks.

    Participants listed in a batch record (batch_provenance.json, if present) take the status of
    that batch; participant folders without a batch record (task_weighted_networks.py) take the processing time from
    their summary.
    """
    rows, folder_of = [], {}

    def add_once(participant, participant_folder):
        """Record the folder of a participant; stop if the participant appears in two folders."""
        if participant in folder_of:
            raise ValueError(f"participant {participant} is in two places ({folder_of[participant]} and "
                             f"{participant_folder}); keep one of them")
        folder_of[participant] = participant_folder.resolve()

    for root in roots:
        for record_path in sorted(root.rglob("batch_provenance.json")):
            record = json.load(open(record_path))
            batch_folder = record_path.parent
            for participant in record.get("participants", []):
                participant_folder = batch_folder / f"sub-{participant}_ses-{session}"
                row = {"participant": participant, "download": str(batch_folder.relative_to(root)),
                       "batch": record.get("batch"), "status": record.get("status", {}).get(participant, "?"),
                       "folder": participant_folder.exists()}
                if participant_folder.exists():
                    add_once(participant, participant_folder)
                    row.update(network_counts(participant_folder))
                rows.append(row)
    for root in roots:
        for participant_folder in sorted(root.rglob(f"sub-*_ses-{session}")):
            if not participant_folder.is_dir() or participant_folder.resolve() in folder_of.values():
                continue
            participant = parse_file_name(participant_folder.name)["sub"]
            add_once(participant, participant_folder)
            summary_path = participant_folder / "subject_summary.json"
            status = f"done in {json.load(open(summary_path))['minutes'] / 60:.2f} h" if summary_path.exists() else "incomplete"
            rows.append({"participant": participant, "download": str(participant_folder.parent.relative_to(root)),
                         "batch": None, "status": status, "folder": True, **network_counts(participant_folder)})
    participants = pd.DataFrame(rows)
    if participants.empty:
        raise FileNotFoundError("no participant folders found")
    counts = [column for column in participants.columns
              if column in ("batch", "runs", "qc_images") or column[:2] in ("A_", "B_", "C_")]
    participants[counts] = participants[counts].astype("Int64")
    has_networks = participants.get("C_unsmoothed", pd.Series(0, index=participants.index)).fillna(0).astype(int).astype(bool)
    motion = [motion_summary(folder_of[participant]) if participant in folder_of else {"excluded": False}
              for participant in participants["participant"]]
    for key in ("mean_fd_mm", "share_fd_above_0.2", "max_fd_mm", "excluded"):
        participants[key if key != "excluded" else "excluded_head_motion"] = [row.get(key, np.nan) for row in motion]
    participants["excluded_head_motion"] = participants["excluded_head_motion"].fillna(False).astype(bool)
    participants["analysed"] = has_networks & ~participants["excluded_head_motion"]
    return participants.sort_values("participant").reset_index(drop=True)


# ================================================================ step 5: summary
def write_summary(out, participants, steps, session, fits_path):
    """Write SUMMARY.md with the main results."""
    analysed = participants[participants["analysed"]]
    summary = [f"# Risk network centrality: {len(analysed)} participant{'s' if len(analysed) != 1 else ''} "
               f"(session {session})", "",
               "Analysis settings: " + "; ".join(f"{name}: {value}" for name, value in ANALYSIS_SETTINGS.items()), "",
               f"Created {time.strftime('%Y-%m-%d %H:%M')} by paper_results.py. Networks are computed from unsmoothed "
               "regional time series (CONN default) and, as a robustness check, from 6 mm smoothed time series.", ""]
    fits = pd.read_csv(fits_path, sep="\t", dtype={"subject": str, "session": str})
    fits["subject"] = fits["subject"].str.zfill(2)
    rnp = fits[fits["session"] == session].set_index("subject")["rnp_mean"]
    columns = ["participant", "batch", "status", "runs"] if "runs" in participants else ["participant", "batch", "status"]
    table = participants[columns].copy()
    table["rnp"] = [rnp.get(participant, np.nan) for participant in table["participant"]]
    table["analysed"] = participants["analysed"].map({True: "yes", False: "no"})
    summary += ["## Participants", "", markdown_table(table, {"rnp": "{:.2f}", "runs": "{:.0f}"})]
    if "excluded_head_motion" in participants:
        excluded = participants[participants["excluded_head_motion"]]
        summary += ["## Exclusion for head motion", "",
                    f"Stringent criteria of Parkes et al. (2018), over all scans of the session: mean FD > {MOTION_MEAN_FD} "
                    f"mm, more than {MOTION_SHARE:.0%} of scans with FD > {MOTION_FD_THRESHOLD} mm, or any FD > "
                    f"{MOTION_MAX_FD} mm. Excluded: {len(excluded)} of {len(participants)}.", ""]
        if len(excluded):
            summary += [markdown_table(excluded[["participant", "mean_fd_mm", "share_fd_above_0.2", "max_fd_mm"]],
                                       {"mean_fd_mm": "{:.3f}", "share_fd_above_0.2": "{:.3f}", "max_fd_mm": "{:.2f}"})]
    not_analysed = participants[~participants["analysed"]]
    if len(not_analysed):
        summary += ["Not analysed: " + "; ".join(f"{row.participant} ({row.status})" for row in not_analysed.itertuples()), ""]

    quality = read_table(out / "unsmoothed" / "report" / "quality.tsv")
    if quality is not None:
        columns = [column for column in ["participant", "runs", "csf_fraction", "gm_fraction", "wm_fraction", "wm_mask_voxels",
                                         "csf_mask_voxels", "mean_fd_mm", "median_tsnr", "min_roi_coverage", "flags"]
                   if column in quality]
        summary += ["## Quality", "", markdown_table(quality[columns], {
            "csf_fraction": "{:.2f}", "gm_fraction": "{:.2f}", "wm_fraction": "{:.2f}", "wm_mask_voxels": "{:.0f}",
            "csf_mask_voxels": "{:.0f}", "mean_fd_mm": "{:.2f}", "median_tsnr": "{:.1f}", "min_roi_coverage": "{:.2f}"})]

    rows = []
    for variant in VARIANTS:
        target_table = read_table(out / variant / "report" / "target_by_participant.tsv")
        comparison = read_table(out / variant / "report" / "smoothed_vs_unsmoothed.tsv")
        if target_table is None:
            continue
        with_rnp = target_table["rnp_3t"].notna()
        rho = stats.spearmanr(target_table.loc[with_rnp, "target_eigenvector_C"], target_table.loc[with_rnp, "rnp_3t"])[0] \
            if with_rnp.sum() >= 4 else np.nan
        rows.append({"version": variant, "target-region eigenvector centrality, mean": target_table["target_eigenvector_C"].mean(),
                     "target-region rank among 23 regions, mean": target_table["target_rank_of_23"].mean(),
                     "Spearman rho with RNP": rho,
                     "edge correlation with the other version, mean": comparison["edge_pattern_r"].mean()
                     if comparison is not None else np.nan})
    if rows:
        summary += [f"## Target region in network C ({TARGET_LABEL})", "",
                    markdown_table(pd.DataFrame(rows), {"target-region eigenvector centrality, mean": "{:.3f}",
                                                        "target-region rank among 23 regions, mean": "{:.1f}",
                                                        "Spearman rho with RNP": "{:.2f}",
                                                        "edge correlation with the other version, mean": "{:.2f}"})]

    stability_tables = []
    position = {measure: index for index, measure in enumerate(MEASURE_NAMES)}
    for variant in VARIANTS:
        stability = read_table(out / variant / "centrality" / "stability_by_measure.tsv")
        if stability is not None:
            stability["position"] = [position.get(measure, len(position)) for measure in stability["measure"]]
            stability = stability.sort_values("position", kind="stable")
            stability["measure"] = [MEASURE_NAMES.get(measure, measure) for measure in stability["measure"]]
            stability.insert(0, "version", variant)
            stability_tables.append(stability)
    if stability_tables:
        stability = pd.concat(stability_tables).rename(columns={
            "identification_accuracy": "identification", "idiff": "Idiff", "target_icc": "target-region ICC",
            "mean_node_icc": "mean ICC over regions"})
        columns = [column for column in ["version", "measure", "participants", "identification", "chance", "Idiff",
                                         "target-region ICC", "mean ICC over regions"] if column in stability]
        summary += ["## Stability between odd and even runs", "",
                    "Identification: each participant's profile over the 23 regions from the odd runs is matched to all "
                    "profiles from the even runs, and the reverse (Finn et al. 2015). Idiff: differential identifiability "
                    "(Amico & Goñi 2018). ICC: intraclass correlation ICC(3,1) between the two halves.", "",
                    markdown_table(stability[columns], {"identification": "{:.2f}", "chance": "{:.2f}", "Idiff": "{:.1f}",
                                                        "target-region ICC": "{:.2f}", "mean ICC over regions": "{:.2f}"})]

    rows = []
    for variant in VARIANTS:
        path = out / variant / "prediction" / "results.json"
        if not path.exists():
            continue
        results = json.load(open(path)).get("results", {})
        session_set = session if session in results else ("both" if "both" in results else None)
        if session_set is None:
            rows.append({"version": variant,
                         "model": f"not computed: at least 8 participants are needed (n = {len(analysed)})"})
            continue
        result = results[session_set]
        for model_key, model_name in PREDICTION_MODELS.items():
            if model_key in result:
                scores = result[model_key]
                rows.append({"version": variant, "model": model_name, "participants": scores["n_participants"],
                             "CV R2": scores["cv_r2"], "r(predicted, observed)": scores["r_pred_obs"],
                             "permutation p": scores["p_perm_cv_r2"]})
    if rows:
        summary += ["## Prediction of RNP from the target region in network C (leave one participant out)", "",
                    markdown_table(pd.DataFrame(rows), {"participants": "{:.0f}", "CV R2": "{:.2f}",
                                                        "r(predicted, observed)": "{:.2f}", "permutation p": "{:.3f}"})]

    prediction_formats = {"participants": "{:.0f}", "CV R2": "{:.2f}", "r(predicted, observed)": "{:.2f}",
                          "permutation p": "{:.3f}"}
    results_by_variant = {variant: json.load(open(out / variant / "prediction" / "results.json"))
                          for variant in VARIANTS if (out / variant / "prediction" / "results.json").exists()}
    rows = []
    for variant, results in results_by_variant.items():
        regions = read_table(out / variant / "prediction" / f"regions_{session}.tsv")
        if regions is not None:
            target = regions[(regions["measure"] == "eigenvector_pos")
                           & regions["label"].str.contains(TARGET_LABEL, regex=False)]
            for row in target.itertuples():
                rows.append({"version": variant, "definition": EIGENVECTOR_DEFINITIONS["eigenvector_pos"],
                             "participants": row.participants, "r with RNP": row.r, "p": row.p, "CV R2": row.cv_r2,
                             "permutation p": row.p_perm_cv_r2})
        for row in results.get("multiplex", []):
            if row["session"] == session:
                rows.append({"version": variant, "definition": EIGENVECTOR_DEFINITIONS[row["measure"]],
                             "participants": row["participants"], "r with RNP": row["r"], "p": row["p"],
                             "CV R2": row["cv_r2"], "permutation p": row["p_perm_cv_r2"]})
    if rows:
        summary += ["## Eigenvector centrality of the target region: network C and multiplex network A", "",
                    "1: eigenvector centrality in network C (all 192 trials). 2: mean single-layer eigenvector "
                    "centrality in network A (12 payoff layers). 3: uniform multiplex eigenvector centrality, the "
                    "eigenvector centrality of the sum of the 12 layers (Solá et al. 2013). 4: versatility "
                    "(De Domenico et al. 2015). Group-level test: Pearson correlation with RNP across participants. "
                    "Prediction: leave one participant out, as for each region.", "",
                    markdown_table(pd.DataFrame(rows), {"participants": "{:.0f}", "r with RNP": "{:.2f}",
                                                        "p": "{:.3f}", "CV R2": "{:.2f}", "permutation p": "{:.3f}"})]

    rows = []
    for variant, results in results_by_variant.items():
        for row in results.get("multiplex_vs_baseline", []):
            if row["session"] == session:
                rows.append({"version": variant, "definition": EIGENVECTOR_DEFINITIONS[row["measure"]],
                             "participants": row["participants"], "CV R2 difference": row["cv_r2_difference"],
                             "95% interval": f"{row['interval_low']:.2f} to {row['interval_high']:.2f}",
                             "sign-flip p (one-sided)": row["p_sign_flip"], "participants better": row["participants_better"],
                             "Williams t": row["williams_t"], "p (one-sided)": row["p_williams"]})
    if rows:
        summary += ["## Multiplex measures against the baseline (network C)", "",
                    "Paired by participant: the difference in CV R2 is the sum over participants of (held-out error with "
                    "the baseline)^2 - (error with the multiplex measure)^2, divided by the total sum of squares. p: "
                    "one-sided sign-flip test of these differences. 95% interval: 5,000 bootstrap samples of "
                    "participants. Williams' t: the two correlations with RNP (Steiger 1980).", "",
                    markdown_table(pd.DataFrame(rows), {"participants": "{:.0f}", "CV R2 difference": "{:.2f}",
                                                        "sign-flip p (one-sided)": "{:.3f}",
                                                        "participants better": "{:.0f}", "Williams t": "{:.2f}",
                                                        "p (one-sided)": "{:.3f}"})]

    rows = []
    for variant, results in results_by_variant.items():
        for row in results.get("hubs", []):
            if row["session"] == session:
                rows.append({"version": variant, "definition": EIGENVECTOR_DEFINITIONS.get(row["measure"], row["measure"]),
                             "group": row["group"], "participants": row["participants"],
                             "r with RNP": row.get("r"), "CV R2": row.get("cv_r2"),
                             "permutation p": row.get("p_perm_cv_r2")})
    if rows:
        summary += ["## Prediction of RNP in the participants in whom the target region is the hub", "",
                    "Groups (defined from the networks only): hub, the target region ranks first among the 23 regions; "
                    "top 2, it ranks first or second; hub under all three definitions, it ranks first under measures "
                    "1-3. Prediction: leave one participant out, 5,000 permutations, groups of at "
                    "least 8 participants (empty cells: fewer).", "",
                    markdown_table(pd.DataFrame(rows), {"participants": "{:.0f}", "r with RNP": "{:.2f}",
                                                        "CV R2": "{:.2f}", "permutation p": "{:.3f}"})]

    rows = []
    for variant, results in results_by_variant.items():
        further = results.get("further_outcomes", {}).get("nu1", {})
        session_set = session if session in further else ("both" if "both" in further else None)
        for model_key, model_name in PREDICTION_MODELS.items():
            if session_set and model_key in further[session_set]:
                scores = further[session_set][model_key]
                rows.append({"version": variant, "model": model_name, "participants": scores["n_participants"],
                             "CV R2": scores["cv_r2"], "r(predicted, observed)": scores["r_pred_obs"],
                             "permutation p": scores["p_perm_cv_r2"]})
    if rows:
        summary += ["## Prediction of ν1 from the target region in network C (leave one participant out)", "",
                    "ν1: noise of the first-payoff magnitude representation in the choice model of de Hollander et "
                    "al.", "",
                    markdown_table(pd.DataFrame(rows), prediction_formats)]

    rows = []
    for variant, results in results_by_variant.items():
        for measure, result in results.get("regions", {}).get(session, {}).items():
            target, best = result["target"], result["best_region_by_cv_r2"]
            rows.append({"version": variant, "measure": MEASURE_NAMES.get(measure, measure), "target-region r": target["r"],
                         "target-region Holm p": target["p_holm"], "target-region CV R2": target["cv_r2"],
                         "target-region rank by CV R2": target["rank_cv_r2"],
                         "highest CV R2": f"{best['label']} ({best['cv_r2']:.2f})",
                         "regions with Holm p < 0.05": ", ".join(result["regions_with_holm_p_below_0.05"]) or "none"})
    if rows:
        summary += ["## Each region in network C: group-level test and prediction of RNP", "",
                    "Group-level test: Pearson correlation across participants, Holm correction for the 23 regions. "
                    "Prediction: leave one participant out, as for the target region. All regions: "
                    f"unsmoothed/prediction/regions_{session}.tsv and smoothed/prediction/regions_{session}.tsv.", "",
                    markdown_table(pd.DataFrame(rows), {"target-region r": "{:.2f}", "target-region Holm p": "{:.3f}",
                                                        "target-region CV R2": "{:.2f}",
                                                        "target-region rank by CV R2": "{:.0f}"})]

    rows = []
    for variant, results in results_by_variant.items():
        for row in results.get("presentation_order", []):
            if row["session"] == session:
                rows.append({"version": variant, "order": dict(ORDERS).get(row["order"], row["order"]),
                             "measure": MEASURE_NAMES.get(row["measure"], row["measure"]),
                             "participants": row["participants"], "mean": row["mean"], "r with RNP": row["r"],
                             "p": row["p"], "CV R2": row["cv_r2"], "permutation p": row["p_perm_cv_r2"]})
    if rows:
        summary += ["## Target region in network C of each presentation order", "",
                    "Network C computed separately for each presentation order (half of the trials each).", "",
                    markdown_table(pd.DataFrame(rows), {"participants": "{:.0f}", "mean": "{:.3f}", "r with RNP": "{:.2f}",
                                                        "p": "{:.3f}", "CV R2": "{:.2f}", "permutation p": "{:.3f}"})]

    rows = []
    for variant, results in results_by_variant.items():
        tests = results.get("nu1_and_centrality", {})
        session_set = session if session in tests else ("both" if "both" in tests else None)
        if session_set:
            test = tests[session_set]
            rows.append({"version": variant, "participants": test["n_participants"], "CV R2, ν1": test["cv_r2_base"],
                         "CV R2, ν1 and eigenvector centrality": test["cv_r2_with_added"],
                         "difference": test["difference"], "permutation p": test["p_perm_difference"]})
    if rows:
        summary += ["## Prediction of RNP from ν1 alone and from ν1 plus the target region's eigenvector centrality", "",
                    "Permutation p: the target region's eigenvector centrality is permuted between participants.", "",
                    markdown_table(pd.DataFrame(rows), {"participants": "{:.0f}", "CV R2, ν1": "{:.2f}",
                                                        "CV R2, ν1 and eigenvector centrality": "{:.2f}",
                                                        "difference": "{:.2f}", "permutation p": "{:.3f}"})]

    summary += ["## Figures", "", "- paper: " + ", ".join(sorted(str(path.relative_to(out))
                                                                for path in (out / "figures").glob("*.png")))]
    for variant in VARIANTS:
        report_figures = sorted((out / variant / "report").glob("fig*.png"))
        if report_figures:
            summary += [f"- {variant}: " + ", ".join(str(path.relative_to(out)) for path in report_figures)]
    summary += ["", "## Steps", "", markdown_table(pd.DataFrame(steps))]
    (out / "SUMMARY.md").write_text("\n".join(summary) + "\n")


# ================================================================ main program
def run_step(steps, step, network_version, function, *arguments, **options):
    """Run one analysis step; a failure is printed and recorded, and the remaining steps still run."""
    print(f"\n==== {network_version} {step}", flush=True)
    time_start = time.time()
    try:
        function(*arguments, **options)
        status = "ok"
    except Exception as error:
        traceback.print_exc()
        status = f"failed: {type(error).__name__}: {error}"
    steps.append({"step": step, "version": network_version, "status": status, "seconds": round(time.time() - time_start, 1)})
    print(f"==== {network_version} {step}: {status} in {steps[-1]['seconds']} s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("networks", nargs="+", help="output folders of task_weighted_networks.py or zip files of these folders")
    parser.add_argument("--out", required=True, type=Path, help="output folder")
    parser.add_argument("--session", default="3t2", help="3t2 (3T) or 7t2 (7T)")
    parser.add_argument("--permutations", type=int, default=5000, help="number of permutations")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    regions_path = HERE / "atlas" / "risk_atlas_regions.csv"
    fits_path = HERE / "choice_models" / "choice_model_fits.tsv"          # RNP and nu1 for each participant and session

    # ---------------------------------------------------------------- 1-2. unpack and inventory
    _, folders = unpack(args.networks, out / "unpacked")
    roots = [out / "unpacked"] + folders
    participants = inventory(roots, args.session)
    participants.to_csv(out / "inventory.tsv", sep="\t", index=False)
    print(participants.to_string(index=False))
    if not participants["analysed"].any():
        raise SystemExit("no participant with networks; nothing to analyse")
    analysed = list(participants.loc[participants["analysed"], "participant"])
    excluded = list(participants.loc[participants["excluded_head_motion"], "participant"])
    print(f"excluded for head motion (Parkes et al. 2018, stringent): {excluded or 'none'}; analysed: {len(analysed)}")

    # ---------------------------------------------------------------- 3. analyses for both network versions
    steps = []
    for variant in VARIANTS:
        variant_folder = out / variant
        run_step(steps, "report", variant, network_report, roots, regions_path, fits_path, variant_folder / "report",
                 variant, args.session, subjects=analysed)
        run_step(steps, "centrality", variant, centrality_and_stability, roots, variant_folder / "centrality", variant,
                 args.session, regions_path, subjects=analysed)
        run_step(steps, "prediction", variant, predict_behaviour, variant_folder / "centrality" / "centrality_all_nodes.tsv",
                 [fits_path], variant_folder / "prediction", outcomes=("rnp_mean", "nu1"), regions_path=regions_path,
                 permutations=args.permutations, variant=variant,
                 centrality_by_order_path=variant_folder / "centrality" / "centrality_by_order.tsv",
                 multiplex_path=variant_folder / "centrality" / "multiplex_centrality.tsv", subjects=analysed)

    # ---------------------------------------------------------------- 4. paper figures
    (out / "figures").mkdir(exist_ok=True)
    run_step(steps, "figure_1", "paper", figure_regions, regions_path, out / "figures" / "fig1_risk_network_regions.png")
    run_step(steps, "figure_2", "paper", figure_networks, roots, regions_path, out / "figures" / "fig2_networks.png",
             args.session, subjects=analysed)
    run_step(steps, "figure_3", "paper", figure_measures, out / "figures" / "fig3_centrality_measures.png")

    # ---------------------------------------------------------------- 5. summary
    write_summary(out, participants, steps, args.session, fits_path)
    print(f"\nwritten: {out / 'SUMMARY.md'}")
    failed = [step for step in steps if step["status"] != "ok"]
    if failed:
        print("FAILED steps: " + ", ".join(f"{step['version']} {step['step']}" for step in failed))


if __name__ == "__main__":
    main()
