"""Out-of-sample prediction of risk attitude from target-region centrality.

Input: centrality_all_nodes.tsv from network_centrality.py and one or more behavioural tables with the columns subject,
session and one column per score (e.g. RNP and nu1 in choice_models/choice_model_fits.tsv).
Tests for the first outcome (e.g. rnp_mean), repeated for each further outcome given with --outcomes:
  prediction   prediction of the outcome from the target region's eigenvector centrality in network C with
               leave-one-participant-out cross-validation (both sessions of the held-out participant are left out
               together); cross-validated R2, mean absolute error, correlation between predicted and observed values,
               permutation p-value
Further tests for the first outcome:
  regions      each region's eigenvector centrality in network C, for each session: group-level test (Pearson
               correlation across participants, Holm correction for the number of regions) and prediction
  order        the target region's eigenvector centrality in network C of each presentation order
               (centrality_by_order.tsv from network_centrality.py): group-level test and prediction
  nu1          prediction from nu1 alone and from nu1 plus the target region's eigenvector centrality
  multiplex    the target region's mean single-layer and uniform multiplex eigenvector centrality and versatility in
               network A (multiplex_centrality.tsv from network_centrality.py), for each session: group-level test and
               prediction
  baseline     each multiplex measure against the baseline (eigenvector centrality in network C), paired by
               participant: difference in CV R2, one-sided sign-flip p, 95% bootstrap interval, participants better,
               Williams' t for the two dependent correlations
  hubs         prediction in the participants whose network has the target region as its hub (ranked first among all
               regions by the measure), in those with the target region among the two most central regions, and in
               those whose hub is the target region under all three eigenvector definitions (network C, mean
               single-layer, uniform); the groups are defined from the networks only
Choice: the tests require at least 8 participants (MIN_PARTICIPANTS).

usage: python predictive_analysis.py --centrality OUT/centrality_all_nodes.tsv
                                     --behaviour choice_models/choice_model_fits.tsv
                                     --out OUT [--outcomes rnp_mean nu1] [--permutations 5000]
                                     [--centrality-by-order OUT/centrality_by_order.tsv]
                                     [--multiplex-centrality OUT/multiplex_centrality.tsv]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from task_weighted_networks import TARGET_LABEL, SESSION_7T


# ================================================================ out-of-sample prediction
def leave_one_participant_out(data, predictors, outcome, session_7t):
    """Predict the outcome of each participant from a model fitted without this participant's data.

    All sessions of the held-out participant are left out together. The model is fitted by ordinary least squares,
    with session as a covariate when two sessions are present.
    Choice: out-of-sample prediction, as recommended for neuromarkers (Woo et al. 2017), with ordinary least squares and
    no hyperparameter tuning, so that no information from the held-out participant enters the model.
    """
    data = data.dropna(subset=predictors + [outcome]).copy()
    two_sessions = data["session"].nunique() > 1
    columns = predictors + (["session_7t"] if two_sessions else [])
    data["session_7t"] = (data["session"] == session_7t).astype(float)
    predicted = np.full(len(data), np.nan)
    participant_of_row = data["subject"].to_numpy()
    design = np.column_stack([np.ones(len(data))] + [data[column].to_numpy(float) for column in columns])
    observed = data[outcome].to_numpy(float)
    for participant in np.unique(participant_of_row):
        held_out = participant_of_row == participant
        coefficients = np.linalg.lstsq(design[~held_out], observed[~held_out], rcond=None)[0]
        predicted[held_out] = design[held_out] @ coefficients
    return data, observed, predicted


def prediction_scores(observed, predicted):
    """Compute the prediction scores.

    Cross-validated R2 (1 - residual sum of squares / total sum of squares; negative when the prediction is worse than
    the mean), mean absolute error, and the correlation between predicted and observed values.
    """
    residual_sum_of_squares = np.sum((observed - predicted) ** 2)
    total_sum_of_squares = np.sum((observed - observed.mean()) ** 2)
    return {"cv_r2": 1 - residual_sum_of_squares / total_sum_of_squares,
            "mae": float(np.mean(np.abs(observed - predicted))),
            "r_pred_obs": float(np.corrcoef(observed, predicted)[0, 1]) if np.std(predicted) > 0 else np.nan}


def prediction_test(data, predictors, outcome, n_permutations, random_generator, session_7t):
    """Compute the out-of-sample prediction scores and the permutation p-value.

    Choice: the outcome is permuted between participants (the sessions of a participant stay together) and the full
    cross-validation is repeated; p = (number of null R2 >= observed R2 + 1) / (permutations + 1).
    """
    data, observed, predicted = leave_one_participant_out(data, predictors, outcome, session_7t)
    scores = prediction_scores(observed, predicted)
    participants = data["subject"].unique()
    outcome_by_participant = {participant: data.loc[data["subject"] == participant].set_index("session")[outcome]
                              for participant in participants}
    null_r2 = []
    for _ in range(n_permutations):
        shuffled = dict(zip(participants, random_generator.permutation(participants)))
        shuffled_data = data.copy()
        shuffled_data[outcome] = [outcome_by_participant[shuffled[participant]].get(session, np.nan)
                                  for participant, session in zip(shuffled_data["subject"], shuffled_data["session"])]
        _, observed_shuffled, predicted_shuffled = leave_one_participant_out(shuffled_data, predictors, outcome, session_7t)
        if len(observed_shuffled) == len(observed):
            null_r2.append(prediction_scores(observed_shuffled, predicted_shuffled)["cv_r2"])
    null_r2 = np.array(null_r2)
    scores["p_perm_cv_r2"] = float((np.sum(null_r2 >= scores["cv_r2"]) + 1) / (len(null_r2) + 1)) if len(null_r2) else np.nan
    scores["n_rows"] = int(len(observed))
    scores["n_participants"] = int(len(participants))
    scores["predictors"] = predictors
    return scores


def one_predictor_test(data, predictor, outcome, n_permutations, random_generator):
    """Compute prediction_test for one predictor and one row per participant, with all permutations at once.

    The scores equal those of prediction_test (the same leave-one-participant-out OLS and the same permutations
    of the outcome between participants, drawn from the generator in the same way). The computation is vectorised
    because the test is run for every region and presentation order.
    """
    data = data.dropna(subset=[predictor, outcome])
    if data["subject"].duplicated().any():
        raise ValueError("one_predictor_test requires one row per participant")
    x, y = data[predictor].to_numpy(float), data[outcome].to_numpy(float)
    participants = data["subject"].to_numpy()
    row_of_participant = {participant: row for row, participant in enumerate(participants)}
    rows = [np.arange(len(y))] + [np.array([row_of_participant[participant]
                                            for participant in random_generator.permutation(participants)])
                                  for _ in range(n_permutations)]
    outcomes = y[np.array(rows)]                                         # first row: observed outcome
    predicted = np.empty_like(outcomes)
    for held_out in range(len(y)):
        train = np.arange(len(y)) != held_out
        x_mean, y_mean = x[train].mean(), outcomes[:, train].mean(axis=1)
        slope = ((x[train] - x_mean) * (outcomes[:, train] - y_mean[:, None])).sum(axis=1) / ((x[train] - x_mean) ** 2).sum()
        predicted[:, held_out] = y_mean + slope * (x[held_out] - x_mean)
    residual_sum_of_squares = ((outcomes - predicted) ** 2).sum(axis=1)
    total_sum_of_squares = ((outcomes - outcomes.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    r2 = 1 - residual_sum_of_squares / total_sum_of_squares
    return {"cv_r2": float(r2[0]),
            "r_pred_obs": float(np.corrcoef(y, predicted[0])[0, 1]) if np.std(predicted[0]) > 0 else np.nan,
            "p_perm_cv_r2": float((np.sum(r2[1:] >= r2[0]) + 1) / (n_permutations + 1)) if n_permutations else np.nan,
            "n_participants": int(len(y))}


def added_predictor_test(data, base_predictors, added_predictor, outcome, n_permutations, random_generator, session_7t):
    """Test whether a predictor improves the out-of-sample prediction beyond the base predictors.

    Returns the cross-validated R2 of the base model and of the model with the added predictor, their difference and
    a permutation p-value for the difference. Choice: the added predictor is permuted between participants (the
    sessions of a participant stay together) and the larger model is cross-validated again, so that the relation
    between the base predictors and the outcome is kept in the null distribution.
    """
    data = data.dropna(subset=base_predictors + [added_predictor, outcome]).copy()
    _, observed, predicted_base = leave_one_participant_out(data, base_predictors, outcome, session_7t)
    _, _, predicted_full = leave_one_participant_out(data, base_predictors + [added_predictor], outcome, session_7t)
    base_r2 = prediction_scores(observed, predicted_base)["cv_r2"]
    difference = prediction_scores(observed, predicted_full)["cv_r2"] - base_r2
    participants = data["subject"].unique()
    values_by_participant = {participant: data.loc[data["subject"] == participant].set_index("session")[added_predictor]
                             for participant in participants}
    null_difference = []
    for _ in range(n_permutations):
        shuffled = dict(zip(participants, random_generator.permutation(participants)))
        shuffled_data = data.copy()
        shuffled_data[added_predictor] = [values_by_participant[shuffled[participant]].get(session, np.nan)
                                          for participant, session in zip(shuffled_data["subject"], shuffled_data["session"])]
        _, observed_shuffled, predicted_shuffled = leave_one_participant_out(
            shuffled_data, base_predictors + [added_predictor], outcome, session_7t)
        if len(observed_shuffled) == len(observed):
            null_difference.append(prediction_scores(observed_shuffled, predicted_shuffled)["cv_r2"] - base_r2)
    null_difference = np.array(null_difference)
    return {"cv_r2_base": float(base_r2), "cv_r2_with_added": float(base_r2 + difference), "difference": float(difference),
            "p_perm_difference": float((np.sum(null_difference >= difference) + 1) / (len(null_difference) + 1))
            if len(null_difference) else np.nan,
            "base_predictors": base_predictors, "added_predictor": added_predictor,
            "n_participants": int(len(participants))}


# ================================================================ main program
HERE = Path(__file__).resolve().parent
MIN_PARTICIPANTS = 8
SESSION_SETS = {"both": ["3t2", "7t2"], "3t2": ["3t2"], "7t2": ["7t2"]}
RANDOM_SEED = 20261123            # a single generator for all permutation tests, used in a fixed order
REGION_MEASURES = ("eigenvector_pos",)                 # measures tested for each region and each presentation order
# multiplex measures of the target region in network A: definitions 2-4, then the two robustness checks
MULTIPLEX_MEASURES = ("multiplex_eigenvector_mean_single_layer", "multiplex_eigenvector_uniform", "multiplex_versatility",
                      "multiplex_eigenvector_uniform_shrunk", "multiplex_eigenvector_uniform_absolute")
# eigenvector definitions 1-3, which define the group "hub under all three definitions"
DEFINITIONS_1_TO_3 = ("eigenvector_pos", "multiplex_eigenvector_mean_single_layer", "multiplex_eigenvector_uniform")
BOOTSTRAP_SAMPLES = 5000
EXACT_SIGN_FLIP_MAX = 26          # exact sign-flip test up to 2^26 sign patterns; random sign patterns above


def read_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--centrality", required=True, type=Path, help="centrality_all_nodes.tsv from network_centrality.py")
    parser.add_argument("--behaviour", nargs="+", required=True, type=Path,
                        help="behavioural tables with the columns subject, session and scores")
    parser.add_argument("--outcomes", nargs="+", default=["rnp_mean", "nu1"],
                        help="scores to predict; the further tests use the first")
    parser.add_argument("--centrality-by-order", type=Path, default=None,
                        help="centrality_by_order.tsv from network_centrality.py (default: next to --centrality)")
    parser.add_argument("--multiplex-centrality", type=Path, default=None,
                        help="multiplex_centrality.tsv from network_centrality.py (default: next to --centrality)")
    parser.add_argument("--correlate", nargs="*", default=["nu1"],
                        help="further scores read from the behavioural tables (nu1: prediction from nu1 and centrality)")
    parser.add_argument("--out", required=True, type=Path, help="output folder")
    parser.add_argument("--regions", type=Path, default=HERE / "atlas" / "risk_atlas_regions.csv", help="region table of the atlas")
    parser.add_argument("--target-label", default=TARGET_LABEL, help="label of the target region")
    parser.add_argument("--permutations", type=int, default=5000, help="number of permutations")
    parser.add_argument("--variant", default="unsmoothed", help="network version, recorded in results.json")
    return parser.parse_args()


def outcome_tests(target_table, outcome, n_permutations, random_generator):
    """Predict the outcome from the target region's eigenvector centrality in network C (and, as a robustness check,
    from the eigenvector centrality of the absolute edge weights), for each set of sessions."""
    results = {}
    for session_set, sessions in SESSION_SETS.items():
        data = target_table[target_table["session"].isin(sessions)]
        if data["subject"].nunique() < MIN_PARTICIPANTS or data[outcome].notna().sum() < MIN_PARTICIPANTS:
            continue
        results[session_set] = {measure: prediction_test(data, [measure], outcome, n_permutations, random_generator,
                                                         SESSION_7T)
                                for measure in ("eigenvector_pos", "eigenvector_abs") if measure in data}
    return results


def region_tests(network_c, behaviour, outcome, n_permutations, random_generator, region_names, target_region, out):
    """Test each region's eigenvector centrality in network C against the outcome, for each session.

    Group-level test: Pearson correlation across participants, one test per region, Holm correction for the number of
    regions. Prediction: leave one participant out, with permutations, as for the target region.
    Choice: Holm's method controls the family-wise error rate (Buergi et al. also used family-wise correction);
    sessions are tested separately so that each participant contributes one value per test.
    """
    from statsmodels.stats.multitest import multipletests
    results = {}
    for session in sorted(network_c["session"].unique()):
        data = network_c[network_c["session"] == session].merge(
            behaviour[["subject", "session", outcome]], on=["subject", "session"], how="left").dropna(subset=[outcome])
        if data["subject"].nunique() < MIN_PARTICIPANTS:
            continue
        rows = []
        for measure in REGION_MEASURES:
            for region, region_data in data.groupby("roi"):
                region_data = region_data.dropna(subset=[measure])
                r, p = stats.pearsonr(region_data[measure], region_data[outcome])
                scores = one_predictor_test(region_data, measure, outcome, n_permutations, random_generator)
                rows.append({"measure": measure, "roi": int(region), "label": region_names.get(region, ""),
                             "participants": scores["n_participants"], "r": float(r), "p": float(p),
                             "cv_r2": scores["cv_r2"], "r_pred_obs": scores["r_pred_obs"],
                             "p_perm_cv_r2": scores["p_perm_cv_r2"]})
        table = pd.DataFrame(rows)
        results[session] = {}
        for measure in REGION_MEASURES:
            of_measure = table["measure"] == measure
            table.loc[of_measure, "p_holm"] = multipletests(table.loc[of_measure, "p"], method="holm")[1]
            table.loc[of_measure, "rank_cv_r2"] = table.loc[of_measure, "cv_r2"].rank(ascending=False, method="min")
            measure_table = table[of_measure]
            target = measure_table[measure_table["roi"] == target_region].iloc[0]
            best = measure_table.loc[measure_table["cv_r2"].idxmax()]
            results[session][measure] = {
                "target": {"r": float(target["r"]), "p": float(target["p"]), "p_holm": float(target["p_holm"]),
                         "cv_r2": float(target["cv_r2"]), "p_perm_cv_r2": float(target["p_perm_cv_r2"]),
                         "rank_cv_r2": int(target["rank_cv_r2"])},
                "best_region_by_cv_r2": {"label": best["label"], "r": float(best["r"]), "p_holm": float(best["p_holm"]),
                                         "cv_r2": float(best["cv_r2"]), "p_perm_cv_r2": float(best["p_perm_cv_r2"])},
                "regions_with_holm_p_below_0.05": list(measure_table.loc[measure_table["p_holm"] < 0.05, "label"]),
                "n_regions": int(len(measure_table))}
        table["rank_cv_r2"] = table["rank_cv_r2"].astype(int)
        table["order_of_measure"] = table["measure"].map({measure: index for index, measure in enumerate(REGION_MEASURES)})
        table = table.sort_values(["order_of_measure", "cv_r2"], ascending=[True, False]).drop(columns="order_of_measure")
        table.to_csv(out / f"regions_{session}.tsv", sep="\t", index=False)
    return results


def order_tests(centrality_by_order, behaviour, outcome, n_permutations, random_generator, target_region, out):
    """Test the target region's eigenvector centrality in network C of each presentation order against the outcome.

    The same group-level test and prediction as in region_tests, for each session and order.
    """
    target_rows = centrality_by_order[centrality_by_order["roi"] == target_region].merge(
        behaviour[["subject", "session", outcome]], on=["subject", "session"], how="left").dropna(subset=[outcome])
    rows = []
    for (session, order), data in target_rows.groupby(["session", "order"]):
        if data["subject"].nunique() < MIN_PARTICIPANTS:
            continue
        for measure in REGION_MEASURES:
            data_of_measure = data.dropna(subset=[measure])
            r, p = stats.pearsonr(data_of_measure[measure], data_of_measure[outcome])
            scores = one_predictor_test(data_of_measure, measure, outcome, n_permutations, random_generator)
            rows.append({"session": session, "order": order, "measure": measure,
                         "participants": scores["n_participants"], "mean": float(data_of_measure[measure].mean()),
                         "r": float(r), "p": float(p), "cv_r2": scores["cv_r2"], "r_pred_obs": scores["r_pred_obs"],
                         "p_perm_cv_r2": scores["p_perm_cv_r2"]})
    table = pd.DataFrame(rows)
    table.to_csv(out / "target_by_order.tsv", sep="\t", index=False)
    return rows


def nu1_and_centrality_tests(target_table, outcome, n_permutations, random_generator):
    """Predict the outcome from nu1 alone and from nu1 plus the target region's eigenvector centrality, for each set
    of sessions."""
    results = {}
    for session_set, sessions in SESSION_SETS.items():
        data = target_table[target_table["session"].isin(sessions)].dropna(subset=["nu1", "eigenvector_pos", outcome])
        if data["subject"].nunique() < MIN_PARTICIPANTS:
            continue
        results[session_set] = added_predictor_test(data, ["nu1"], "eigenvector_pos", outcome, n_permutations,
                                                    random_generator, SESSION_7T)
    return results


def multiplex_tests(multiplex, behaviour, outcome, n_permutations, random_generator, target_region, out):
    """Test the target region's multiplex centralities in network A (definitions 2-4 and the robustness checks) against
    the outcome.

    The same group-level test and prediction as in region_tests, for each session.
    """
    target_rows = multiplex[multiplex["roi"] == target_region].merge(
        behaviour[["subject", "session", outcome]], on=["subject", "session"], how="left").dropna(subset=[outcome])
    rows = []
    for session, data in target_rows.groupby("session"):
        if data["subject"].nunique() < MIN_PARTICIPANTS:
            continue
        for measure in MULTIPLEX_MEASURES:
            if measure not in data or data[measure].notna().sum() < MIN_PARTICIPANTS:
                continue
            data_of_measure = data.dropna(subset=[measure])
            r, p = stats.pearsonr(data_of_measure[measure], data_of_measure[outcome])
            scores = one_predictor_test(data_of_measure, measure, outcome, n_permutations, random_generator)
            rows.append({"session": session, "measure": measure, "participants": scores["n_participants"],
                         "mean": float(data_of_measure[measure].mean()), "r": float(r), "p": float(p),
                         "cv_r2": scores["cv_r2"], "r_pred_obs": scores["r_pred_obs"],
                         "p_perm_cv_r2": scores["p_perm_cv_r2"]})
    pd.DataFrame(rows, columns=["session", "measure", "participants", "mean", "r", "p", "cv_r2", "r_pred_obs",
                                "p_perm_cv_r2"]).to_csv(out / "target_multiplex.tsv", sep="\t", index=False)
    return rows


def sign_flip_p(differences, random_generator, n_random=200000):
    """One-sided sign-flip p of the sum of paired differences (share of sign patterns with a sum at least as large).

    Choice: all 2^n sign patterns up to EXACT_SIGN_FLIP_MAX participants (in chunks of 2^16), random patterns above.
    """
    differences = np.asarray(differences, float)
    n, observed = len(differences), differences.sum()
    if n > EXACT_SIGN_FLIP_MAX:
        signs = random_generator.choice([-1.0, 1.0], size=(n_random, n))
        return float((np.sum(signs @ differences >= observed - 1e-12) + 1) / (n_random + 1))
    low = min(n, 16)
    patterns = 1 - 2 * ((np.arange(2 ** low)[:, None] >> np.arange(low)) & 1)
    partial = patterns @ differences[:low]
    count = 0
    for high in range(2 ** (n - low)):
        signs = 1 - 2 * ((high >> np.arange(n - low)) & 1)
        count += int(np.sum(partial + signs @ differences[low:] >= observed - 1e-12))
    return count / 2 ** n


def williams_t(r_base, r_new, r_base_new, n):
    """Williams' t for two dependent correlations with one variable in common (Steiger 1980); one-sided p that the
    correlation of the new measure is stronger."""
    determinant = 1 - r_base ** 2 - r_new ** 2 - r_base_new ** 2 + 2 * r_base * r_new * r_base_new
    mean_r = (r_base + r_new) / 2
    t = (abs(r_new) - abs(r_base)) * np.sqrt((n - 1) * (1 + r_base_new) /
                                             (2 * (n - 1) / (n - 3) * determinant + mean_r ** 2 * (1 - r_base_new) ** 3))
    return float(t), float(stats.t.sf(t, n - 3))


def baseline_comparison(target_c, multiplex, behaviour, outcome, random_generator, target_region, out):
    """Compare each multiplex measure of the target region with the baseline (eigenvector centrality in network C).

    Paired by participant, for each session: the gain g_i = (held-out error with the baseline)^2 - (held-out error
    with the multiplex measure)^2; the difference in CV R2 is sum(g_i) / total sum of squares. p: one-sided sign-flip
    test of the g_i.
    95% interval: BOOTSTRAP_SAMPLES bootstrap samples of participants (copies of a participant are held out together).
    Williams' t compares the two correlations with the outcome.
    """
    target_rows = multiplex[multiplex["roi"] == target_region]
    rows = []
    for session, data in target_rows.groupby("session"):
        data = data.merge(target_c[["subject", "session", "eigenvector_pos"]], on=["subject", "session"]).merge(
            behaviour[["subject", "session", outcome]], on=["subject", "session"], how="left").dropna(subset=[outcome])
        if data["subject"].nunique() < MIN_PARTICIPANTS:
            continue
        data = data.reset_index(drop=True)

        def errors(frame, predictor):
            _, observed, predicted = leave_one_participant_out(frame, [predictor], outcome, SESSION_7T)
            return observed - predicted

        for measure in MULTIPLEX_MEASURES:
            if measure not in data or data[measure].notna().sum() < MIN_PARTICIPANTS:
                continue
            frame = data.dropna(subset=[measure, "eigenvector_pos"]).reset_index(drop=True)
            total = np.sum((frame[outcome] - frame[outcome].mean()) ** 2)
            gain = errors(frame, "eigenvector_pos") ** 2 - errors(frame, measure) ** 2    # per participant
            bootstrap_differences = []
            for _ in range(BOOTSTRAP_SAMPLES):
                index = random_generator.integers(0, len(frame), len(frame))
                sample = frame.iloc[index].reset_index(drop=True)
                total_sample = np.sum((sample[outcome] - sample[outcome].mean()) ** 2)
                bootstrap_differences.append((np.sum(errors(sample, "eigenvector_pos") ** 2)
                                              - np.sum(errors(sample, measure) ** 2)) / total_sample)
            r_base = stats.pearsonr(frame["eigenvector_pos"], frame[outcome])[0]
            r_new = stats.pearsonr(frame[measure], frame[outcome])[0]
            r_base_new = stats.pearsonr(frame["eigenvector_pos"], frame[measure])[0]
            t, p_t = williams_t(r_base, r_new, r_base_new, len(frame))
            rows.append({"session": session, "measure": measure, "participants": int(len(frame)),
                         "cv_r2_difference": float(gain.sum() / total), "p_sign_flip": sign_flip_p(gain, random_generator),
                         "interval_low": float(np.percentile(bootstrap_differences, 2.5)),
                         "interval_high": float(np.percentile(bootstrap_differences, 97.5)),
                         "participants_better": int(np.sum(gain > 0)), "r_baseline": float(r_base), "r_measure": float(r_new),
                         "r_baseline_measure": float(r_base_new), "williams_t": t, "p_williams": p_t})
    pd.DataFrame(rows).to_csv(out / "target_multiplex_vs_baseline.tsv", sep="\t", index=False)
    return rows


def hub_tests(network_c, multiplex, behaviour, outcome, n_permutations, random_generator, target_region, out):
    """Predict the outcome from the target region's centrality in the participants in whom it is the hub.

    For each session and each eigenvector measure (network C, mean single-layer, uniform, versatility and the
    robustness checks), the rank of the target region among all regions (1: most central) defines the groups: "hub"
    (rank 1), "top 2" (rank 1 or 2) and "hub under all three definitions" (rank 1 for network C, mean single-layer and
    uniform). Prediction in each group as in region_tests (leave one participant out, permutations), for groups of at
    least MIN_PARTICIPANTS participants.
    Choice: the groups are defined from the networks only, without the outcome.
    """
    rows = []
    for session in sorted(network_c["session"].unique()):
        tables = {"eigenvector_pos": network_c[network_c["session"] == session].pivot(
            index="subject", columns="roi", values="eigenvector_pos")}
        of_session = multiplex[multiplex["session"] == session]
        for measure in MULTIPLEX_MEASURES:
            if measure in of_session:
                tables[measure] = of_session.pivot(index="subject", columns="roi", values=measure)
        outcome_of = behaviour[behaviour["session"] == session].set_index("subject")[outcome]
        with_outcome = set(outcome_of.dropna().index)
        tables = {measure: table.loc[sorted(set(table.dropna().index) & with_outcome)]
                  for measure, table in tables.items()}
        tables = {measure: table for measure, table in tables.items() if len(table)}
        ranks = {measure: table.rank(axis=1, ascending=False, method="min")[target_region]
                 for measure, table in tables.items()}
        definitions = [measure for measure in DEFINITIONS_1_TO_3 if measure in ranks]
        hub_under_all = sorted(set.intersection(*[set(ranks[measure].index[ranks[measure] == 1])
                                                  for measure in definitions])) if definitions else []
        for measure, table in tables.items():
            participants = list(table.index)
            groups = {"hub": [participant for participant in participants if ranks[measure][participant] == 1],
                      "top 2": [participant for participant in participants if ranks[measure][participant] <= 2],
                      "hub under all three definitions": [participant for participant in hub_under_all
                                                          if participant in participants]}
            for group, members in groups.items():
                row = {"session": session, "measure": measure, "group": group, "participants": len(members),
                       "participants_in_session": len(participants)}
                if len(members) >= MIN_PARTICIPANTS:
                    data = pd.DataFrame({"subject": members, "session": session,
                                         "value": table.loc[members, target_region].to_numpy(float),
                                         outcome: outcome_of.loc[members].to_numpy(float)})
                    r, p = stats.pearsonr(data["value"], data[outcome])
                    scores = one_predictor_test(data, "value", outcome, n_permutations, random_generator)
                    row.update({"r": float(r), "p": float(p), "cv_r2": scores["cv_r2"], "r_pred_obs": scores["r_pred_obs"],
                                "p_perm_cv_r2": scores["p_perm_cv_r2"]})
                rows.append(row)
    pd.DataFrame(rows).to_csv(out / "target_hub_tests.tsv", sep="\t", index=False)
    return rows


def predict_behaviour(centrality_path, behaviour_paths, out, outcomes=("rnp_mean", "nu1"), correlate=("nu1",),
                      regions_path=HERE / "atlas" / "risk_atlas_regions.csv", target_label=TARGET_LABEL, permutations=5000,
                      variant="unsmoothed", centrality_by_order_path=None, multiplex_path=None, subjects=None):
    """Run all tests on the centrality tables from network_centrality.py and write the tables and results.json.

    subjects: the participant IDs to analyse (default: all); paper_results.py passes the participants who are not
    excluded for head motion.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    random_generator = np.random.default_rng(RANDOM_SEED)
    regions = pd.read_csv(regions_path)
    region_names = dict(zip(regions["roi"], regions["label"]))
    target_region = int(regions.loc[regions["label"].str.contains(target_label, regex=False), "roi"].iloc[0])

    centrality = pd.read_csv(centrality_path, sep="\t", dtype={"subject": str, "session": str})
    analysed = (lambda table: table) if subjects is None else (
        lambda table: table[table["subject"].isin({str(subject) for subject in subjects})])
    centrality = analysed(centrality)
    behaviour = None
    for path in behaviour_paths:
        table = pd.read_csv(path, sep="\t", dtype={"subject": str, "session": str})
        table["subject"] = table["subject"].str.zfill(2)
        columns = ["subject", "session"] + [column for column in table.columns
                                            if column in outcomes or column in correlate]
        table = table[list(dict.fromkeys(columns))]
        behaviour = table if behaviour is None else behaviour.merge(table, on=["subject", "session"], how="outer")
    report = {"variant": variant, "target_roi": target_region, "target_label": region_names[target_region],
              "n_network_files": int(centrality.groupby(["subject", "session", "network", "layer"]).ngroups),
              "participants_by_session": centrality.groupby("session")["subject"].nunique().to_dict()}

    # ---------------------------------------------------------------- target-region measures per participant and session
    network_c = centrality[centrality["network"] == "C"]
    target_c = network_c[network_c["roi"] == target_region][
        ["subject", "session"] + [column for column in ("eigenvector_pos", "eigenvector_abs") if column in network_c]]
    target_table = target_c.merge(behaviour, on=["subject", "session"], how="left")
    target_table.to_csv(out / "target_summary.tsv", sep="\t", index=False)

    # ---------------------------------------------------------------- prediction
    first_outcome, *further_outcomes = outcomes
    report["results"] = outcome_tests(target_table, first_outcome, permutations, random_generator)
    for outcome in further_outcomes:
        report.setdefault("further_outcomes", {})[outcome] = outcome_tests(target_table, outcome, permutations,
                                                                           random_generator)

    # ---------------------------------------------------------------- each region, presentation order, nu1, multiplex
    #                                                                  measures, comparison with the baseline, hubs
    report["regions"] = region_tests(network_c, behaviour, first_outcome, permutations, random_generator, region_names,
                                     target_region, out)
    by_order_path = Path(centrality_by_order_path) if centrality_by_order_path else Path(centrality_path).with_name(
        "centrality_by_order.tsv")
    if by_order_path.exists():
        by_order = analysed(pd.read_csv(by_order_path, sep="\t", dtype={"subject": str, "session": str}))
        report["presentation_order"] = order_tests(by_order, behaviour, first_outcome, permutations, random_generator,
                                                   target_region, out)
    if "nu1" in target_table and first_outcome != "nu1":
        report["nu1_and_centrality"] = nu1_and_centrality_tests(target_table, first_outcome, permutations, random_generator)
    multiplex_path = Path(multiplex_path) if multiplex_path else Path(centrality_path).with_name("multiplex_centrality.tsv")
    if multiplex_path.exists():
        multiplex = analysed(pd.read_csv(multiplex_path, sep="\t", dtype={"subject": str, "session": str}))
        report["multiplex"] = multiplex_tests(multiplex, behaviour, first_outcome, permutations, random_generator,
                                              target_region, out)
        report["multiplex_vs_baseline"] = baseline_comparison(network_c[network_c["roi"] == target_region], multiplex,
                                                              behaviour, first_outcome, random_generator, target_region, out)
        report["hubs"] = hub_tests(network_c, multiplex, behaviour, first_outcome, permutations, random_generator,
                                   target_region, out)

    json.dump(report, open(out / "results.json", "w"), indent=2, default=float)
    print(json.dumps(report, indent=2, default=float)[:3000])


if __name__ == "__main__":
    args = read_arguments()
    predict_behaviour(args.centrality, args.behaviour, args.out, args.outcomes, args.correlate, args.regions,
                      args.target_label, args.permutations, args.variant, args.centrality_by_order,
                      args.multiplex_centrality)
