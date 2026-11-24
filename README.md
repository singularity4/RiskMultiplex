# RiskMultiplex

Code for constructing task-weighted functional connectivity networks from fMRI data and computing node centrality. Each fMRI volume is weighted by a task phase, and edges are weighted correlations between regional time series. Functional connectivity networks are computed from all trials, for each run, and for each payoff magnitude and presentation order (a multiplex network).

## Network centrality

Four centrality measures are computed: eigenvector centrality in network C (baseline), mean single-layer and uniform multiplex eigenvector centrality, and versatility.

Nodes are the 23 regions of the risk network atlas. Edges are task-weighted Pearson correlations between regional time series, Fisher z-transformed, with negative edges set to zero. Eigenvectors are normalised to unit length.

1. **Eigenvector centrality in network C.** Network C is computed from the first-payoff phase of all 192 trials. The centrality is the leading eigenvector of its weighted adjacency matrix.
2. **Mean single-layer multiplex eigenvector centrality in network A.** Network A is a multiplex network with 12 layers, one for each combination of safe payoff and presentation order (16 trials per layer), without interlayer edges. Eigenvector centrality is computed in each layer and averaged over the 12 layers.
3. **Uniform multiplex eigenvector centrality in network A** (Solá et al. 2013). The leading eigenvector of the sum of the 12 layers.
4. **Versatility in network A** (De Domenico et al. 2015). The leading eigenvector of the supra-adjacency matrix, in which each node is connected to its replicas in all other layers; the interlayer weight is the participant's mean positive edge weight. The versatility of a node is the sum over its 12 replicas.

`network_centrality.py` writes measure 1 to `centrality_all_nodes.tsv` and measures 2–4 to `multiplex_centrality.tsv`. As robustness checks, measure 3 is also computed from Ledoit–Wolf shrinkage estimates of the layers, and measures 1 and 3 from absolute edge weights.

## Quick Start

    git clone https://github.com/singularity4/RiskMultiplex.git
    cd RiskMultiplex
    pip install -r requirements.txt

Tested with Python 3.13.

## Data

- fMRI and behavioural data: OpenNeuro [ds007508](https://openneuro.org/datasets/ds007508) (de Hollander et al. 2026), downloaded by `task_weighted_networks.py --download`.
- Atlas: `atlas/` contains the 23 regions of the risk network, obtained by spectral clustering of the Neurosynth uniformity-test map for "risky" ([neurosynth.org](https://neurosynth.org)) and labelled with AAL3 ([oxcns.org/aal3.html](https://www.oxcns.org/aal3.html)).
- Risk attitude: `choice_models/choice_model_fits.tsv` contains RNP and ν1 for each participant and session, estimated from the behavioural data in [gilles86/risk_experiment](https://github.com/gilles86/risk_experiment).

## Usage

    bash reproduce.sh            # all results
    bash reproduce.sh --quick    # test run with one participant

or step by step:

    python risk_brain_network_atlas.py --map data/risky_uniformity-test_z_FDR_0.01.nii.gz --aal3-zip data/AAL3v1_for_SPM12.zip --out results/atlas
    python task_weighted_networks.py --bids data/ds007508 --download --out results/networks
    python network_centrality.py --networks results/networks --out results/centrality
    python predictive_analysis.py --centrality results/centrality/centrality_all_nodes.tsv --behaviour choice_models/choice_model_fits.tsv --out results/prediction
    python paper_results.py results/networks --out results

| Script | Description |
|---|---|
| `risk_brain_network_atlas.py` | atlas construction from a Neurosynth map (spectral clustering, AAL3 labels) |
| `spectral_clustering.py` | kernel non-negative spectral clustering (KNSC-Ncut, KNSC-Rcut) |
| `task_weighted_networks.py` | preprocessing, denoising and task-weighted networks, from raw BIDS data or fMRIPrep outputs |
| `task_weighted_networks_notebook.py` | task-weighted networks in a Kaggle or Colab notebook |
| `network_centrality.py` | network centrality measures, including multiplex eigenvector centrality, and their split-half stability |
| `predictive_analysis.py` | cross-validated prediction of RNP and ν1, comparison with the baseline, subgroup analysis and group-level tests |
| `paper_results.py` | all numbers, tables and figures in the paper |
| `choice_models/choice_models.py` | risk attitude (RNP) and magnitude noise (ν1) from choice models |

Results are written to `results/`. All random seeds are fixed.

## Tests

    pip install pytest
    python -m pytest tests

The tests check the methods on small synthetic examples with known results.

## License

MIT
