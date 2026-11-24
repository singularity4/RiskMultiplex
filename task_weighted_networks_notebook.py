"""Network computation in a Kaggle or Colab notebook with task_weighted_networks.py.

Copy this file into one notebook cell, enable internet access, set the participants and the session below, and run the
cell. The script installs the required packages, downloads the participants' data from OpenNeuro (ds007508), computes
their networks and writes one zip file with the networks and run tables, which is the input to paper_results.py.
Processing takes about 2.7 h per participant on 4 CPUs (Kaggle), so one 12 h session fits up to 4 participants.
"""
import subprocess
import sys
import zipfile
from pathlib import Path

PARTICIPANTS = ["07", "11", "13", "16"]     # participant labels without "sub-"
SESSION = "3t2"                             # 3t2 (3T) or 7t2 (7T)
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()


def run(command):
    """Print and run a shell command."""
    print("+", command, flush=True)
    subprocess.run(command, shell=True, check=True)


# 1. code, Python packages and FSL topup (distortion correction)
repository = WORK / "RiskMultiplex"
if not repository.exists():
    run(f"git clone --depth 1 https://github.com/singularity4/RiskMultiplex.git {repository}")
run(f"{sys.executable} -m pip install --quiet -r {repository / 'requirements.txt'}")
sys.path.insert(0, str(repository))
from task_weighted_networks import install_topup   # noqa: E402
fsl_dir = install_topup("/tmp/fsl")

# 2. data download, preprocessing, denoising and networks for each participant
networks = WORK / f"networks_ses-{SESSION}"
run(f"{sys.executable} {repository / 'task_weighted_networks.py'} --bids {WORK / 'ds007508'} --download "
    f"--session {SESSION} --fsl-dir {fsl_dir} --out {networks} --participants {' '.join(PARTICIPANTS)}")

# 3. zip file for paper_results.py: networks, run tables and summaries (no images)
zip_path = WORK / f"ses-{SESSION}_sub-{'-'.join(PARTICIPANTS)}_networks.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(networks.rglob("*")):
        if path.is_file() and path.suffix in (".csv", ".tsv", ".json"):
            archive.write(path, path.relative_to(networks))
print(f"written {zip_path}")
