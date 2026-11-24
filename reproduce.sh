#!/usr/bin/env bash
# Reproduces all results of the paper in results/.
#   bash reproduce.sh           all 30 participants (about 2.7 h per participant on 4 CPUs), 5,000 permutations
#   bash reproduce.sh --quick   test run with one participant and 100 permutations (about 3 h); the stability
#                               analysis (at least 3 participants) and the prediction (at least 8) are skipped
set -euo pipefail
cd "$(dirname "$0")"

participants=""                      # empty: all 30 participants
permutations=5000
if [[ "${1:-}" == "--quick" ]]; then
    participants="--participants 07"
    permutations=100
fi
mkdir -p data results

# FSL topup for distortion correction: an existing FSL installation, otherwise topup is installed into tools/fsl
fsl_dir="${FSLDIR:-$PWD/tools/fsl}"
if [[ ! -x "$fsl_dir/bin/topup" ]]; then
    python -c "from task_weighted_networks import install_topup; install_topup('$fsl_dir')"
fi

# 1. atlas: rebuilt and compared with atlas/ if the Neurosynth map and AAL3 are in data/
if [[ -f data/risky_uniformity-test_z_FDR_0.01.nii.gz && -f data/AAL3v1_for_SPM12.zip ]]; then
    python risk_brain_network_atlas.py --map data/risky_uniformity-test_z_FDR_0.01.nii.gz \
        --aal3-zip data/AAL3v1_for_SPM12.zip --out results/atlas
    cmp results/atlas/risk_atlas_regions.csv atlas/risk_atlas_regions.csv && echo "atlas reproduced"
fi

# 2. networks: download of OpenNeuro ds007508 into data/, preprocessing, denoising and task-weighted networks
python task_weighted_networks.py --bids data/ds007508 --download --fsl-dir "$fsl_dir" --out results/networks $participants

# 3. centrality, stability, prediction of risk attitude, figures and results/SUMMARY.md
python paper_results.py results/networks --out results --permutations "$permutations"
