"""Choice models of de Hollander et al. (2026), refitted per session: risk attitude (RNP) and first-payoff magnitude
noise (nu1).

Behavioural data: risk_experiment/paper/behavioral_data in github.com/gilles86/risk_experiment (sub-XX_behavior.tsv).

RNP   hierarchical Bayesian probit model 'probit_simple' of de Hollander et al. (Methods, eq. 9-10):
          chose_risky ~ x + (x | subject),   x = log(n_risky / n_safe),   Bambi default priors, per session
      RNP = clip(exp(intercept / slope), 0, 1) for each participant and posterior draw.
      Choice: RNP is the measure of risk attitude, as in de Hollander et al.: the safe/risky payoff ratio at which a
      participant is indifferent, which equals the winning probability at which a risk-neutral decision maker would be
      indifferent. The true winning probability is 0.55; RNP = 0.55 indicates risk neutrality, lower values indicate
      risk aversion and higher values risk seeking.
nu1   perceptual magnitude-comparison model (PMCM), model 1 of de Hollander et al., fitted per session.
      Choice: nu1, the noise of the first-payoff magnitude representation, is the PMCM parameter used in this study.
The compare step compares both measures with the source data published by the authors and writes
choice_model_fits.tsv.

usage:
  python choice_models.py rnp --behaviour BEHAVIOUR_FOLDER --out FOLDER          requires bambi; about 1 min per session
  python choice_models.py nu1 --behaviour BEHAVIOUR_FOLDER --out FOLDER          requires bauer and PyMC; about 10 min per session
  python choice_models.py compare --out FOLDER --source-data SOURCE_DATA_FOLDER  writes FOLDER/choice_model_fits.tsv
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 20261123
SESSIONS = ("3t2", "7t2")
NEUTRAL_RNP = 0.55        # true winning probability of the risky option


def read_behaviour(behaviour_folder):
    """Read the trials of all participants (sub-XX_behavior.tsv) and add a 'subject' column."""
    tables = []
    for path in sorted(Path(behaviour_folder).glob("sub-*_behavior.tsv")):
        trials = pd.read_csv(path, sep="\t", dtype={"session": str})
        trials["subject"] = path.name[4:6]
        tables.append(trials)
    return pd.concat(tables, ignore_index=True)


# ================================================================ RNP: hierarchical probit model (Bambi)
def fit_rnp(behaviour_folder, out):
    """Estimate RNP for each participant and session.

    Sampler settings as in de Hollander et al.: adapt_diag initialisation, target_accept 0.9, 1000 tuning steps,
    1000 draws and 4 chains.
    """
    import bambi
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    draws, tune, target_accept = 1000, 1000, 0.9
    trials = read_behaviour(behaviour_folder)
    trials = trials[trials["chose_risky"].notna()].copy()
    trials["chose_risky"] = trials["chose_risky"].astype(str).str.lower().map({"true": 1, "false": 0}).astype(int)
    trials["x"] = trials["log_risky_safe"].astype(float)

    rows, diagnostics = [], {}
    for session in SESSIONS:
        time_start = time.time()
        session_trials = trials[trials["session"] == session].reset_index(drop=True)
        model = bambi.Model("chose_risky ~ x + (x|subject)", session_trials, family="bernoulli", link="probit")
        inference_data = model.fit(draws=draws, tune=tune, chains=4, cores=2, init="adapt_diag",
                                   target_accept=target_accept, random_seed=RANDOM_SEED, progressbar=False)
        try:
            inference_data.to_netcdf(str(out / f"rnp_probit_simple_ses-{session}_trace.nc"))
        except Exception as error:
            print("trace not saved:", error, flush=True)
        posterior = inference_data.posterior
        intercept_deviations, slope_deviations = posterior["1|subject"], posterior["x|subject"]
        participant_dimension = [name for name in intercept_deviations.dims if name not in ("chain", "draw")][0]
        intercept = posterior["Intercept"] + intercept_deviations           # participant-level intercepts
        slope = posterior["x"] + slope_deviations                           # participant-level slopes
        rnp = np.clip(np.exp(intercept / slope), 0, 1)
        for participant in rnp[participant_dimension].values:
            rnp_draws = rnp.sel({participant_dimension: participant}).values.ravel()
            low, high = np.quantile(rnp_draws, [0.025, 0.975])
            if high < NEUTRAL_RNP:
                risk_profile = "risk-averse"
            elif low > NEUTRAL_RNP:
                risk_profile = "risk-seeking"
            else:
                risk_profile = "risk-neutral"
            rows.append({"subject": str(participant), "session": session,
                         "rnp_mean": rnp_draws.mean(), "rnp_median": np.median(rnp_draws), "rnp_low95": low,
                         "rnp_high95": high, "risk_profile": risk_profile,
                         "intercept_mean": intercept.sel({participant_dimension: participant}).values.ravel().mean(),
                         "slope_mean": slope.sel({participant_dimension: participant}).values.ravel().mean()})
        r_hat = {name: split_r_hat(posterior[name].values) for name in ["Intercept", "x"]}
        for name, samples in (("1|subject", intercept_deviations), ("x|subject", slope_deviations), ("rnp", rnp)):
            values = samples.transpose("chain", "draw", ...).values
            r_hat[name + " (max over participants)"] = max(split_r_hat(values[..., index]) for index in range(values.shape[-1]))
        diagnostics[session] = {"n_trials": int(len(session_trials)), "n_subjects": int(session_trials["subject"].nunique()),
                                "minutes": round((time.time() - time_start) / 60, 1), "split_rhat": r_hat}
        print(session, "done in", diagnostics[session]["minutes"], "min; largest r-hat", max(r_hat.values()), flush=True)
    fits = pd.DataFrame(rows).sort_values(["session", "subject"])
    fits.to_csv(out / "rnp_per_participant_session.tsv", sep="\t", index=False)
    json.dump({"model": "chose_risky ~ x + (x|subject), probit, Bambi defaults, per session",
               "rnp": "clip(exp(intercept_i / slope_i), 0, 1)", "bambi": bambi.__version__, "seed": RANDOM_SEED,
               "draws": draws, "tune": tune, "target_accept": target_accept, "diagnostics": diagnostics},
              open(out / "rnp_fit_info.json", "w"), indent=2, default=str)


def split_r_hat(samples):
    """Compute the split R-hat (without rank normalisation) of samples with shape (chain, draw)."""
    n_chains, n_draws = samples.shape
    half = n_draws // 2
    split_chains = np.concatenate([samples[:, :half], samples[:, half:2 * half]], axis=0)
    within_variance = split_chains.var(axis=1, ddof=1).mean()
    between_variance = half * split_chains.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((half - 1) / half * within_variance + between_variance / half) / within_variance))


# ================================================================ nu1: PMCM (bauer)
PMCM_PARAMETERS = ["n1_evidence_sd", "n2_evidence_sd", "risky_prior_mu", "safe_prior_mu", "risky_prior_sd", "safe_prior_sd"]


def fit_nu1(behaviour_folder, out, draws=3000, tune=1500, chains=4):
    """Fit PMCM model 1 per session.

    Software as in de Hollander et al.: bauer 0.1.0 (commit e246d78) and PyMC 5.25.1, with the authors' compatibility
    patch for pm.Data(mutable).
    """
    import inspect
    import pymc as pm
    try:
        data_lacks_mutable = "mutable" not in inspect.signature(pm.Data).parameters
    except (ValueError, TypeError):
        data_lacks_mutable = False
    if data_lacks_mutable:          # recent PyMC versions do not accept pm.Data(mutable=...), which bauer passes
        original_data = pm.Data

        def data_without_mutable(*args, **kwargs):
            kwargs.pop("mutable", None)
            return original_data(*args, **kwargs)
        pm.Data = data_without_mutable
    from bauer.models import RiskModel
    import arviz as az

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    trials = read_behaviour(behaviour_folder)
    trials = trials[trials["choice"].notna()].copy()
    trials["p1"], trials["p2"] = trials["prob1"].astype(float), trials["prob2"].astype(float)
    trials["choice"] = trials["choice"].astype(float) == 2.0          # as in de Hollander et al.
    for session in SESSIONS:
        session_trials = trials[trials["session"] == session].set_index(["subject", "run", "trial_nr"])
        model = RiskModel(session_trials[["n1", "n2", "p1", "p2", "choice"]], "full")
        model.build_estimation_model()
        time_start = time.time()
        inference_data = model.sample(draws=draws, tune=tune, target_accept=0.9, chains=chains, cores=min(chains, 2),
                                      random_seed=RANDOM_SEED)
        minutes = (time.time() - time_start) / 60
        az.to_netcdf(inference_data, str(out / f"ses-{session}_model-1_draws{draws}_trace.netcdf"))
        posterior = inference_data.posterior
        rows = []
        for participant in posterior["subject"].values:
            row = {"subject": str(participant), "session": session}
            for parameter in PMCM_PARAMETERS:
                parameter_draws = posterior[parameter].sel(subject=participant).values.ravel()
                row[parameter] = float(parameter_draws.mean())
                if parameter == "n1_evidence_sd":
                    row["nu1_median"] = float(np.median(parameter_draws))
                    row["nu1_low95"] = float(np.quantile(parameter_draws, 0.025))
                    row["nu1_high95"] = float(np.quantile(parameter_draws, 0.975))
            rows.append(row)
        fits = pd.DataFrame(rows).rename(columns={"n1_evidence_sd": "nu1"})
        fits.to_csv(out / f"ses-{session}_model-1_per_participant_draws{draws}.tsv", sep="\t", index=False)
        r_hat = az.rhat(inference_data, var_names=PMCM_PARAMETERS)
        info = {"model": "PMCM model 1 (RiskModel 'full'), per session, bauer e246d78, PyMC " + pm.__version__,
                "session": session, "draws": draws, "tune": tune, "chains": chains, "target_accept": 0.9,
                "minutes": round(minutes, 1), "divergences": int(inference_data.sample_stats["diverging"].sum()),
                "max_rhat": {parameter: float(r_hat[parameter].max()) for parameter in PMCM_PARAMETERS}}
        json.dump(info, open(out / f"ses-{session}_model-1_fit_info_draws{draws}.json", "w"), indent=2)
        print(session, f"done in {minutes:.1f} min; largest r-hat of nu1 {info['max_rhat']['n1_evidence_sd']:.3f}", flush=True)


# ================================================================ comparison with the published results and combined table
def compare(out, source_data, draws=3000):
    """Compare the refitted measures with the published results and write the combined table.

    RNP: the per-session estimates compared with the authors' group RNP and risk profiles. nu1: the 3T-7T
    correlations of the PMCM parameters compared with the authors' Fig. S6. Writes choice_model_fits.tsv (RNP and nu1
    per participant and session) and comparison_with_authors.txt.
    """
    out, report = Path(out), []
    rnp = pd.read_csv(out / "rnp_per_participant_session.tsv", sep="\t", dtype={"subject": str})
    rnp["subject"] = rnp["subject"].str.zfill(2)
    authors_rnp = pd.read_csv(f"{source_data}/figure_01_rnp_stake_effect_rnp.tsv", sep="\t", dtype={"subject": str})
    authors_group_rnp = authors_rnp["rnp"].mean()   # group-level draws (single placeholder subject); only the mean is comparable
    authors_profiles = pd.read_csv(f"{source_data}/figure_02C_participant_diffs.tsv", sep="\t", dtype={"subject": str})
    authors_profiles = authors_profiles.drop_duplicates("subject").set_index("subject")["risk_profile"]
    comparison = rnp.pivot(index="subject", columns="session", values="rnp_mean").join(
        authors_profiles.rename("authors_risk_profile"))
    report += [f"RNP: participants {len(comparison)}; r(ours 3T, ours 7T) = {comparison[['3t2', '7t2']].corr().iloc[0, 1]:.3f}",
               "mean RNP refitted 3T %.3f, 7T %.3f; authors' group-level RNP %.3f"
               % (comparison["3t2"].mean(), comparison["7t2"].mean(), authors_group_rnp)]
    comparison.to_csv(out / "rnp_vs_authors.tsv", sep="\t")
    our_profiles = rnp.groupby("subject")["risk_profile"].agg(lambda profiles: "/".join(sorted(set(profiles))))
    report.append(pd.crosstab(comparison["authors_risk_profile"], our_profiles.reindex(comparison.index),
                              rownames=["authors (both sessions)"], colnames=["ours (per session)"]).to_string())

    parameters = ["nu1"] + PMCM_PARAMETERS[1:]
    fits_3t = pd.read_csv(out / f"ses-3t2_model-1_per_participant_draws{draws}.tsv", sep="\t", dtype={"subject": str})
    fits_7t = pd.read_csv(out / f"ses-7t2_model-1_per_participant_draws{draws}.tsv", sep="\t", dtype={"subject": str})
    nu1 = pd.concat([fits_3t, fits_7t]).sort_values(["session", "subject"])
    parameters_3t, parameters_7t = fits_3t.set_index("subject")[parameters], fits_7t.set_index("subject")[parameters]
    both_sessions = parameters_3t.index.intersection(parameters_7t.index)
    parameters_3t, parameters_7t = parameters_3t.loc[both_sessions], parameters_7t.loc[both_sessions]
    our_correlations = pd.DataFrame({parameter_3t: [np.corrcoef(parameters_3t[parameter_3t], parameters_7t[parameter_7t])[0, 1]
                                                    for parameter_7t in parameters] for parameter_3t in parameters},
                                    index=parameters).T
    authors_correlations = pd.read_csv(f"{source_data}/figure_s06_session_param_corr.tsv", sep="\t", comment="#", index_col=0)
    authors_correlations.index = authors_correlations.columns = parameters
    report += ["", f"nu1 (PMCM model 1, per session): n = {len(both_sessions)} participants in both sessions; "
               f"group mean nu1 3T {parameters_3t['nu1'].mean():.3f}, 7T {parameters_7t['nu1'].mean():.3f}",
               "3T-7T correlation of the same parameter (diagonal of the authors' Fig. S6), refitted vs authors:"]
    report += [f"  {parameter:<18} {our_correlations.loc[parameter, parameter]:.3f}   "
               f"{authors_correlations.loc[parameter, parameter]:.3f}" for parameter in parameters]
    difference = (our_correlations - authors_correlations).abs()
    report.append(f"all 36 cells of the 3T x 7T matrix: max abs difference {difference.values.max():.3f}, "
                  f"mean {difference.values.mean():.3f}")
    (out / "comparison_with_authors.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))

    fits = rnp.merge(nu1, on=["subject", "session"], how="outer").sort_values(["session", "subject"])
    fits.to_csv(out / "choice_model_fits.tsv", sep="\t", index=False)


def read_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["rnp", "nu1", "compare"])
    parser.add_argument("--behaviour", type=Path, help="behavioural data folder, risk_experiment/paper/behavioral_data (rnp, nu1)")
    parser.add_argument("--out", required=True, type=Path, help="output folder")
    parser.add_argument("--source-data", type=Path, help="folder with the authors' source data tables (compare)")
    parser.add_argument("--draws", type=int, default=3000, help="nu1: posterior draws per chain")
    parser.add_argument("--tune", type=int, default=1500, help="nu1: tuning steps per chain")
    parser.add_argument("--chains", type=int, default=4, help="nu1: number of chains")
    return parser.parse_args()


if __name__ == "__main__":
    args = read_arguments()
    if args.step == "rnp":
        fit_rnp(args.behaviour, args.out)
    elif args.step == "nu1":
        fit_nu1(args.behaviour, args.out, args.draws, args.tune, args.chains)
    else:
        compare(args.out, args.source_data, args.draws)
