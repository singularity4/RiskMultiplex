"""Task-weighted functional brain networks for each participant, from raw BIDS data or fMRIPrep outputs.

Following Buergi et al. (2026), each fMRI volume is weighted by a task phase, here the first payoff of each trial.
Processing steps: preprocessing (option 1: raw BIDS data; option 2: fMRIPrep outputs), CONN-style
denoising, task weighting, weighted correlations between all pairs of regions (Fisher z), and three types of
networks:
    C   all 192 trials (8 runs)                                                network-C
    B   one network per run (24 trials each)                                   network-B_run-<run>
    A   one network per payoff condition, presentation order x safe payoff     network-A_order-<order>_safe-<CHF>
        (16 trials each)
Each network type is computed from unsmoothed regional time series (networks_unsmoothed/, CONN default) and from
time series smoothed with a 6 mm kernel (networks/, robustness check). task_weighted_networks_notebook.py runs this
script in a Kaggle or Colab notebook.

Option 1 (default), preprocessing of the raw BIDS data:
    python task_weighted_networks.py --bids ds007508 --out networks --participants 07 11
Option 2, fMRIPrep outputs (fMRIPrep run with --output-spaces MNI152NLin2009cAsym:res-2 --dummy-scans 3):
    python task_weighted_networks.py --bids ds007508 --fmriprep ds007508/derivatives/fmriprep --out networks
For other datasets, adapt the study design below. Comments starting with "Choice:" document methodological choices.
"""
import argparse
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage, signal, stats
from scipy.interpolate import CubicSpline

# ================================================================ study design: OpenNeuro ds007508 (de Hollander et al. 2026)
OPENNEURO_URL = "https://s3.amazonaws.com/openneuro.org/ds007508"
TASK = "task"                          # BIDS task label of the risky-choice runs
SESSIONS = {"3t": "3t2", "7t": "7t2"}    # risky-choice sessions
ANAT_SESSION = "3t1"                   # Choice: the 1 mm T1w image of session 3t1 is used for both sessions
RUNS = list(range(1, 9))               # 8 runs with 24 trials each
PARTICIPANTS = [f"{number:02d}" for number in range(2, 33) if number != 24]   # 30 participants
DUMMY_VOLUMES = 3                      # Choice: discard the first 3 volumes, as in de Hollander et al. (--dummy-scans 3)
FIELDMAP_DIRECTIONS = ("RL", "LR")     # reverse phase-encoding EPI directions (BIDS dir- label), in order of preference
TEMPLATE = "MNI152NLin2009cAsym"       # Choice: default fMRIPrep template, as in de Hollander et al. and Buergi et al.
# The repetition time is read from the BIDS sidecar of each run (2.2979 s at 3T, with small differences between runs).

# Choice: volumes are weighted by the first payoff (0.6 s), the phase in which de Hollander et al. located the
# magnitude representation in the right intraparietal cortex.
WEIGHTING_EVENT = "stimulus 1"
# Choice: all four task phases are regressed out (HRF-convolved, with first differences), corresponding to the removal
# of task effects in CONN (Buergi et al.: "session and task effects and their first-order derivatives").
NUISANCE_EVENTS = ["stimulus 1", "stimulus 2", "choice", "certainty"]

# Payoff layers (A networks): one layer per condition, presentation order x safe payoff (2 x 6 = 12 layers with 16
# trials each).
LAYER_FACTORS = ("order", "safe")


def add_payoff_layer_columns(events):
    """Add the presentation order and the safe payoff of each first-payoff event (prob1 == 1.0: safe option first)."""
    events = events.copy()
    events["order"] = np.where(events["prob1"] == 1.0, "safefirst", "riskyfirst")
    events["safe"] = np.where(events["prob1"] == 1.0, events["n1"], events["n2"]).astype(int)
    return events


def layer_name(layer):
    """Return the file-name label of a payoff layer, e.g. ('safefirst', 5) -> 'order-safefirst_safe-05'."""
    order, safe_payoff = layer
    return f"order-{order}_safe-{safe_payoff:02d}"


TARGET_LABEL = "Parietal_Inf_R"         # Choice: right inferior parietal target region (de Hollander et al.)
SESSION_7T = SESSIONS["7t"]            # session covariate when a prediction includes both sessions

# ================================================================ method settings (Buergi et al. 2026; CONN)
N_COMPCOR = 5                  # Choice: 5 CompCor components per tissue (Buergi et al.; CONN default)
LOW_HZ, HIGH_HZ = 0.01, 0.10   # Choice: band-pass 0.01-0.1 Hz (Buergi et al.; CONN default 0.008-0.09 Hz)
SMOOTHING_FWHM_MM = 6.0        # Choice: 6 mm smoothing kernel for the smoothed version, as in Buergi et al.
# Choice: T1w-to-template registration with ANTs SyN on skull-stripped images, as in fMRIPrep
REGISTRATION = "antsRegistrationSyN[s]"
RANDOM_SEED = 20260922                    # fixed seed for ANTs registration and segmentation
SEGMENTATION_CLIP_PERCENTILES = (1.0, 99.0)   # Choice: T1w intensities clipped before segmentation (ANTsPyNet default)
# Quality limits: plausible tissue fractions (young adults) and minimum mask sizes on the 2 mm grid; processing of a
# participant stops outside these limits.
TISSUE_FRACTION_RANGE = {"csf": (0.05, 0.35), "gm": (0.30, 0.65), "wm": (0.20, 0.55)}
MIN_MASK_VOXELS = {"wm": 10000, "csf": 200}


def ants_library():
    """Import ANTsPy (antspyx) on first use, so that other scripts can import this module without it."""
    import ants
    return ants


# ================================================================ atlas and template on the atlas grid
def atlas_and_template(atlas_path, regions_path, target_label, template, template_folder=None):
    """Load the atlas, the region table, the target region and the TemplateFlow template on the atlas grid (2 mm MNI)."""
    ants = ants_library()
    atlas_image = nib.load(str(atlas_path))
    atlas = np.asarray(atlas_image.dataobj).astype(int)
    regions = pd.read_csv(regions_path)
    n_regions = int(atlas.max())
    assert sorted(regions["roi"]) == list(range(1, n_regions + 1)), "atlas regions must be numbered 1..N"
    target_rows = regions[regions["label"].str.contains(target_label, regex=False)]
    assert len(target_rows) == 1, f"expected one region labelled {target_label}, found {len(target_rows)}"
    atlas_ants = ants.image_read(str(atlas_path), pixeltype="float")
    template_t1 = ants.image_read(str(template_file(template, "T1w", None, template_folder)))
    template_mask = ants.image_read(str(template_file(template, "mask", "brain", template_folder)))
    brain_on_grid = ants.resample_image_to_target(template_mask, atlas_ants, interp_type="nearestNeighbor").numpy() > 0
    return {"atlas_image": atlas_image, "atlas": atlas, "regions": regions, "n_regions": n_regions,
            "target_region": int(target_rows["roi"].iloc[0]), "atlas_ants": atlas_ants, "template_t1": template_t1,
            "template_mask": template_mask, "template_brain": template_t1 * template_mask, "brain_on_grid": brain_on_grid}


def template_file(template, suffix, description, template_folder=None):
    """Return the path of a 1 mm TemplateFlow file (from template_folder if given, otherwise downloaded)."""
    name = f"tpl-{template}_res-01{'_desc-' + description if description else ''}_{suffix}.nii.gz"
    if template_folder:
        found = Path(template_folder) / name
    else:
        import templateflow.api as templateflow
        found = templateflow.get(template, resolution=1, suffix=suffix, desc=description, extension=".nii.gz")
    if isinstance(found, list) or not Path(str(found)).exists() or Path(str(found)).name != name:
        raise FileNotFoundError(f"TemplateFlow: expected {name}, got {found}")
    return Path(str(found))


# ================================================================ option 1: preprocessing of raw data, anatomical image
def preprocess_anatomy(t1_path, space):
    """Preprocess the T1w image: bias-field correction, brain extraction, registration and tissue masks.

    The tissue masks are returned on the atlas grid.
    """
    ants = ants_library()
    template_t1, template_mask, atlas_ants = space["template_t1"], space["template_mask"], space["atlas_ants"]
    time_start = time.time()
    t1 = ants.n4_bias_field_correction(ants.image_read(str(t1_path)))     # N4 (Tustison et al. 2010)
    # brain mask: template brain mask transformed to the T1w image by a fast registration, dilated by one voxel
    head_registration = ants.registration(fixed=template_t1, moving=t1, type_of_transform="antsRegistrationSyNQuick[s]",
                                          random_seed=RANDOM_SEED)
    brain_mask = ants.apply_transforms(fixed=t1, moving=template_mask, transformlist=head_registration["invtransforms"],
                                       whichtoinvert=[True, False], interpolator="genericLabel")
    brain_mask = ants.iMath(brain_mask, "MD", 1)
    t1_brain = t1 * brain_mask
    time_mask = time.time()
    registration = ants.registration(fixed=space["template_brain"], moving=t1_brain, type_of_transform=REGISTRATION,
                                     random_seed=RANDOM_SEED)
    t1_to_template = registration["fwdtransforms"]
    time_registration = time.time()

    # Choice: three tissue classes by k-means segmentation (ANTs Atropos, MRF weight 0.2), ordered by intensity:
    # CSF < grey matter < white matter
    inside_brain = brain_mask.numpy() > 0
    intensities = t1.numpy()
    clip_low, clip_high = np.percentile(intensities[inside_brain], SEGMENTATION_CLIP_PERCENTILES)
    t1_for_segmentation = t1.new_image_like(np.clip(intensities, clip_low, clip_high).astype(np.float32))
    segmentation = ants.atropos(a=t1_for_segmentation, x=brain_mask, i="Kmeans[3]", m="[0.2,1x1x1]", c="[5,0]")
    tissue_classes = segmentation["segmentation"].numpy()
    tissue = {}
    for class_number, tissue_name in enumerate(("csf", "gm", "wm"), start=1):
        tissue_voxels = tissue_classes == class_number
        tissue[tissue_name] = {"fraction": float(tissue_voxels.sum() / inside_brain.sum()),
                               "mean_t1": float(intensities[tissue_voxels].mean()) if tissue_voxels.any() else float("nan")}
    problems = [f"{name} fraction {tissue[name]['fraction']:.3f} outside {low}-{high}"
                for name, (low, high) in TISSUE_FRACTION_RANGE.items()
                if not low <= tissue[name]["fraction"] <= high]
    if not tissue["csf"]["mean_t1"] < tissue["gm"]["mean_t1"] < tissue["wm"]["mean_t1"]:
        problems.append("tissue classes are not ordered CSF < GM < WM in T1w intensity")
    csf_probability, grey_matter_probability, white_matter_probability = segmentation["probabilityimages"]

    def eroded_mask_on_grid(probability):
        """Threshold a tissue probability map at 0.5, erode it by one voxel and transform it to the atlas grid.

        Choice: threshold and erosion in T1w space, as in the CONN default settings.
        """
        eroded_mask = ants.iMath(ants.threshold_image(probability, 0.5, 1.0e9), "ME", 1)
        on_grid = ants.apply_transforms(fixed=atlas_ants, moving=eroded_mask, transformlist=t1_to_template,
                                        interpolator="linear")
        return on_grid.numpy() > 0.5

    white_matter_mask, csf_mask = eroded_mask_on_grid(white_matter_probability), eroded_mask_on_grid(csf_probability)
    for name, mask in (("wm", white_matter_mask), ("csf", csf_mask)):
        if mask.sum() < MIN_MASK_VOXELS[name]:
            problems.append(f"{name} mask on the 2 mm grid has {int(mask.sum())} voxels (minimum {MIN_MASK_VOXELS[name]})")
    summary = {"truncation": [float(clip_low), float(clip_high)], "tissue": tissue,
               "wm_mask_voxels": int(white_matter_mask.sum()), "csf_mask_voxels": int(csf_mask.sum()),
               "minutes": {"bias_mask": round((time_mask - time_start) / 60, 1),
                           "registration": round((time_registration - time_mask) / 60, 1),
                           "segmentation": round((time.time() - time_registration) / 60, 1)}}
    if problems:
        raise RuntimeError("tissue segmentation failed the quality checks: " + "; ".join(problems) + f" ({summary})")
    return {"t1_brain": t1_brain, "t1_to_template": t1_to_template, "white_matter_mask": white_matter_mask,
            "csf_mask": csf_mask, "summary": summary}


# ================================================================ option 1: preprocessing of raw data, functional runs
def slice_timing_correction(bold, slice_times, repetition_time):
    """Shift each slice to the middle of the acquisition window (as in fMRIPrep) by cubic spline interpolation."""
    reference_time = (np.min(slice_times) + np.max(slice_times)) / 2.0
    volume_times = np.arange(bold.shape[3]) * repetition_time
    corrected = np.empty_like(bold)
    for slice_index, slice_time in enumerate(slice_times):
        acquisition_times = volume_times + slice_time
        target_times = np.clip(volume_times + reference_time, acquisition_times[0], acquisition_times[-1])
        corrected[:, :, slice_index, :] = CubicSpline(acquisition_times, bold[:, :, slice_index, :], axis=-1)(target_times)
    return corrected, reference_time


# BIDS phase-encoding direction -> topup acquisition vector
PHASE_ENCODING_VECTOR = {"i": "1 0 0", "i-": "-1 0 0", "j": "0 1 0", "j-": "0 -1 0", "k": "0 0 1", "k-": "0 0 -1"}


def pad_to_even_slices(image):
    """Append a copy of the top slice to an image with an odd number of slices (topup subsamples by a factor of 2)."""
    volumes = np.asarray(image.dataobj, dtype=np.float32)
    if volumes.shape[2] % 2 == 0:
        return nib.Nifti1Image(volumes, image.affine, image.header), False
    volumes = np.concatenate([volumes, volumes[:, :, -1:, ...]], axis=2)
    return nib.Nifti1Image(volumes, image.affine, image.header), True


def distortion_correction(corrected_path, reference_path, fieldmap_path, bold_metadata, fieldmap_metadata, folder, fsl_folder):
    """Correct susceptibility distortions with FSL topup.

    The field is estimated from the mean BOLD image and the mean reverse phase-encoding EPI and applied to the
    motion-corrected run.
    Choice: FSL topup with the standard b02b0 configuration (Andersson et al. 2003), the PEPOLAR approach of fMRIPrep.
    """
    folder.mkdir(parents=True, exist_ok=True)
    reference = nib.load(str(reference_path)); fieldmap_image = nib.load(str(fieldmap_path))
    fieldmap = np.asarray(fieldmap_image.dataobj, dtype=np.float32)
    if fieldmap.ndim == 4:
        fieldmap = fieldmap.mean(axis=3)
    if fieldmap.shape != reference.shape[:3]:
        raise ValueError(f"field map grid {fieldmap.shape} differs from the BOLD grid {reference.shape[:3]}")
    both = np.stack([np.asarray(reference.dataobj, dtype=np.float32), fieldmap], axis=3)
    both_image, padded = pad_to_even_slices(nib.Nifti1Image(both, reference.affine))
    both_image.to_filename(folder / "both.nii.gz")
    readout_bold = bold_metadata.get("TotalReadoutTime")
    readout_fieldmap = fieldmap_metadata.get("TotalReadoutTime", readout_bold)
    with open(folder / "acqparams.txt", "w") as acquisition_parameters:
        acquisition_parameters.write(f"{PHASE_ENCODING_VECTOR[bold_metadata['PhaseEncodingDirection']]} {readout_bold}\n")
        acquisition_parameters.write(f"{PHASE_ENCODING_VECTOR[fieldmap_metadata['PhaseEncodingDirection']]} {readout_fieldmap}\n")
    topup_settings = Path(fsl_folder) / "etc" / "flirtsch" / "b02b0.cnf"
    if not topup_settings.exists():   # standard FSL b02b0 configuration, written if the installation does not include it
        topup_settings = folder / "b02b0.cnf"
        topup_settings.write_text("--warpres=20,16,14,12,10,6,4,4,4\n--subsamp=2,2,2,2,2,1,1,1,1\n"
                                  "--fwhm=8,6,4,3,3,2,1,0,0\n--miter=5,5,5,5,5,10,10,20,20\n"
                                  "--lambda=0.005,0.001,0.0001,0.000015,0.000005,0.0000005,0.00000005,0.0000000005,0.00000000001\n"
                                  "--ssqlambda=1\n--regmod=bending_energy\n--estmov=1,1,1,1,1,0,0,0,0\n"
                                  "--minmet=0,0,0,0,0,1,1,1,1\n--splineorder=3\n--numprec=double\n--interp=spline\n--scale=1\n")
    run_command(f"{fsl_folder}/bin/topup --imain={folder}/both.nii.gz --datain={folder}/acqparams.txt "
                f"--config={topup_settings} --out={folder}/topup --fout={folder}/field_hz --iout={folder}/both_unwarped")
    corrected_image, _ = pad_to_even_slices(nib.load(str(corrected_path)))
    corrected_image.to_filename(folder / "mc_padded.nii.gz")
    run_command(f"{fsl_folder}/bin/applytopup --imain={folder}/mc_padded.nii.gz --datain={folder}/acqparams.txt "
                f"--inindex=1 --topup={folder}/topup --method=jac --out={folder}/sdc")
    undistorted = np.asarray(nib.load(str(folder / "sdc.nii.gz")).dataobj, dtype=np.float32)
    if padded:
        undistorted = undistorted[:, :, :-1, :]
    return undistorted


def run_command(command):
    """Print and run a shell command."""
    print("+", command, flush=True)
    return subprocess.run(command, check=True, shell=True)


def preprocess_run(run_files, anatomy, space, folder, dummy_volumes, fsl_folder=None, log=print):
    """Preprocess one run of raw BIDS data and resample it to the atlas grid.

    Returns the BOLD data (x, y, z, time; float32), the framewise displacement, the repetition time and a summary.
    Without FSL or a reverse phase-encoding EPI, distortion correction is skipped and the summary records this.
    """
    ants = ants_library()
    atlas_ants = space["atlas_ants"]
    time_start = time.time()
    bold_metadata = json.load(open(run_files["bold_json"]))
    repetition_time = float(bold_metadata["RepetitionTime"])
    bold_image = nib.load(str(run_files["bold"]))
    bold = np.asarray(bold_image.dataobj, dtype=np.float32)[..., dummy_volumes:]
    n_volumes = bold.shape[3]
    folder.mkdir(parents=True, exist_ok=True)

    # Choice: slice-timing correction if the sidecar contains slice times (3T data; the 7T sidecars contain none)
    slice_times = np.asarray(bold_metadata.get("SliceTiming", []), dtype=float)
    if slice_times.size == bold.shape[2]:
        bold, reference_time = slice_timing_correction(bold, slice_times, repetition_time)
    else:
        reference_time = None
    nib.Nifti1Image(bold, bold_image.affine).to_filename(folder / "stc.nii.gz")
    del bold
    step_times = {"slice_timing": time.time()}

    # Choice: rigid motion correction to the mean volume of the run (ANTs BOLDRigid); framewise displacement from ANTs,
    # used for quality control only
    bold_ants = ants.image_read(str(folder / "stc.nii.gz"))
    volumes = ants.ndimage_to_list(bold_ants)
    mean_volume = volumes[0] * 0
    for volume in volumes:
        mean_volume = mean_volume + volume / len(volumes)
    motion = ants.motion_correction(bold_ants, fixed=mean_volume, type_of_transform="BOLDRigid")
    ants.image_write(motion["motion_corrected"], str(folder / "mc.nii.gz"))
    framewise_displacement = np.asarray(motion["FD"], dtype=float)
    del bold_ants, volumes, motion
    mean_corrected = np.asarray(nib.load(str(folder / "mc.nii.gz")).dataobj, dtype=np.float32).mean(axis=3)
    nib.Nifti1Image(mean_corrected, bold_image.affine).to_filename(folder / "mc_mean.nii.gz")
    step_times["motion_correction"] = time.time()

    distortion_corrected = False
    corrected_path = folder / "mc.nii.gz"
    if fsl_folder and run_files.get("fieldmap") is not None:
        try:
            undistorted = distortion_correction(folder / "mc.nii.gz", folder / "mc_mean.nii.gz", run_files["fieldmap"],
                                                bold_metadata, json.load(open(run_files["fieldmap_json"])),
                                                folder / "topup", fsl_folder)
            nib.Nifti1Image(undistorted, bold_image.affine).to_filename(folder / "sdc.nii.gz")
            del undistorted
            corrected_path = folder / "sdc.nii.gz"; distortion_corrected = True
        except Exception as error:
            log(f"distortion correction failed ({error}); continuing without distortion correction")
    step_times["distortion_correction"] = time.time()

    # Choice: rigid registration of the mean EPI to the T1w image (mutual information); fMRIPrep uses boundary-based
    # registration, which requires FreeSurfer surfaces.
    corrected = ants.image_read(str(corrected_path))
    corrected_volumes = ants.ndimage_to_list(corrected)
    mean_epi = corrected_volumes[0] * 0
    for volume in corrected_volumes:
        mean_epi = mean_epi + volume / len(corrected_volumes)
    mean_epi_brain = mean_epi * ants.get_mask(mean_epi)
    bold_to_t1 = ants.registration(fixed=anatomy["t1_brain"], moving=mean_epi_brain, type_of_transform="Rigid",
                                   random_seed=RANDOM_SEED)
    transforms = anatomy["t1_to_template"] + bold_to_t1["fwdtransforms"]
    step_times["coregistration"] = time.time()
    # The BOLD -> T1w -> template transforms are composed into one displacement field on the atlas grid, so that each
    # volume is resampled only once. Choice: Lanczos windowed-sinc interpolation (as in fMRIPrep).
    to_atlas_grid_field = ants.apply_transforms(fixed=atlas_ants, moving=corrected_volumes[0], transformlist=transforms,
                                                compose=str(folder / "to_grid_"))
    if to_atlas_grid_field is None:
        raise RuntimeError("composing the transforms to the atlas grid failed")
    bold_on_grid = np.empty(space["atlas"].shape + (n_volumes,), dtype=np.float32)
    for index, volume in enumerate(corrected_volumes):
        bold_on_grid[..., index] = ants.apply_transforms(fixed=atlas_ants, moving=volume, transformlist=[to_atlas_grid_field],
                                                         interpolator="lanczosWindowedSinc").numpy()
    del corrected, corrected_volumes
    step_times["resampling"] = time.time()
    step_minutes, previous = {}, time_start
    for step, finished in step_times.items():
        step_minutes[step] = round((finished - previous) / 60, 1)
        previous = finished
    summary = {"slice_timing": reference_time is not None, "distortion_correction": distortion_corrected,
               "step_minutes": step_minutes}
    return bold_on_grid, framewise_displacement, repetition_time, summary


# ================================================================ option 2: fMRIPrep outputs
def find_one_file(folder, pattern, prefer=None):
    """Return the single file below folder that matches pattern; with several matches, keep those containing prefer."""
    found = sorted(Path(folder).rglob(pattern))
    if prefer:
        found = [path for path in found if prefer in path.name] or found
    if len(found) != 1:
        raise FileNotFoundError(f"expected one file {pattern} under {folder}, found {[path.name for path in found]}")
    return found[0]


def read_fmriprep_run(fmriprep_folder, participant, session, task, run, template, space, dummy_volumes):
    """Read one preprocessed fMRIPrep run (template space) on the atlas grid, without the dummy volumes.

    Returns the BOLD data, the framewise displacement, the repetition time and a summary.
    Choice: the fMRIPrep output in MNI152NLin2009cAsym at res-2 shares the voxel lattice of the atlas grid and is
    re-indexed without interpolation; other grids are resampled once (Lanczos windowed sinc).
    """
    participant_folder = Path(fmriprep_folder) / f"sub-{participant}"
    stem = f"sub-{participant}_ses-{session}_task-{task}_run-{run}"
    bold_path = find_one_file(participant_folder, f"{stem}_space-{template}*_desc-preproc_bold.nii.gz", prefer="res-2")
    sidecar = Path(str(bold_path).replace(".nii.gz", ".json"))
    bold_image = nib.load(str(bold_path))
    if sidecar.exists():
        repetition_time = float(json.load(open(sidecar))["RepetitionTime"])
    else:
        repetition_time = float(bold_image.header.get_zooms()[3])
    confounds_path = find_one_file(participant_folder, f"{stem}_desc-confounds_timeseries.tsv")
    framewise_displacement = pd.read_csv(confounds_path, sep="\t")["framewise_displacement"].to_numpy(float)[dummy_volumes:]
    framewise_displacement = np.nan_to_num(framewise_displacement, nan=0.0)
    bold_on_grid = to_atlas_grid(bold_image, space, volumes=slice(dummy_volumes, None))
    return bold_on_grid, framewise_displacement, repetition_time, {"slice_timing": "fMRIPrep",
                                                                   "distortion_correction": "fMRIPrep",
                                                                   "fmriprep_bold": bold_path.name}


def to_atlas_grid(image, space, volumes=slice(None)):
    """Map a 4D image in template space to the atlas grid: re-indexing if the voxel lattices coincide, else resampling."""
    grid_shape = space["atlas"].shape
    grid_voxels = np.indices(grid_shape).reshape(3, -1)
    world_mm = space["atlas_image"].affine @ np.vstack([grid_voxels, np.ones(grid_voxels.shape[1])])
    source_voxels = np.linalg.solve(image.affine, world_mm)[:3]
    nearest = np.round(source_voxels)
    kept_volumes = list(range(image.shape[3]))[volumes]
    on_lattice = np.allclose(source_voxels, nearest, atol=1e-3) and (nearest >= 0).all() and \
        all((nearest[axis] < image.shape[axis]).all() for axis in range(3))
    on_grid = np.empty(grid_shape + (len(kept_volumes),), dtype=np.float32)
    if on_lattice:
        i, j, k = nearest.astype(int)                    # voxel indices in the source image
        for position, volume in enumerate(kept_volumes):
            on_grid[..., position] = np.asarray(image.dataobj[..., volume], dtype=np.float32)[i, j, k].reshape(grid_shape)
        return on_grid
    ants = ants_library()
    for position, volume in enumerate(kept_volumes):
        volume_ants = ants.from_nibabel(nib.Nifti1Image(np.asarray(image.dataobj[..., volume], dtype=np.float32), image.affine))
        on_grid[..., position] = ants.resample_image_to_target(volume_ants, space["atlas_ants"],
                                                               interp_type="lanczosWindowedSinc").numpy()
    return on_grid


def fmriprep_tissue_masks(fmriprep_folder, participant, template, space, erosion_voxels=1):
    """Compute white matter and CSF masks on the atlas grid from the fMRIPrep tissue probability maps.

    Choice: the rule of option 1 (CONN default): probability > 0.5, erosion by one voxel on the grid of the probability
    map, linear interpolation to the atlas grid and threshold 0.5.
    """
    ants = ants_library()
    masks = {}
    for tissue in ("WM", "CSF"):
        path = find_one_file(Path(fmriprep_folder) / f"sub-{participant}",
                             f"sub-{participant}*_space-{template}*_label-{tissue}_probseg.nii.gz", prefer="res-2")
        mask = ants.threshold_image(ants.image_read(str(path)), 0.5, 1.0e9)
        if erosion_voxels:
            mask = ants.iMath(mask, "ME", erosion_voxels)
        on_grid = ants.resample_image_to_target(mask, space["atlas_ants"], interp_type="linear")
        masks[tissue.lower()] = on_grid.numpy() > 0.5
    for name in ("wm", "csf"):
        if masks[name].sum() < MIN_MASK_VOXELS[name]:
            raise RuntimeError(f"{name} mask on the atlas grid has {int(masks[name].sum())} voxels")
    return masks["wm"], masks["csf"]


def install_topup(fsl_folder="/tmp/fsl"):
    """Install FSL topup and applytopup from the FSL conda channel (with micromamba) into fsl_folder."""
    fsl_folder = Path(fsl_folder)
    if (fsl_folder / "bin" / "topup").exists():
        return fsl_folder
    micromamba_folder = Path("/tmp/micromamba")
    micromamba_folder.mkdir(parents=True, exist_ok=True)
    if not (micromamba_folder / "bin" / "micromamba").exists():
        run_command(f"curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C {micromamba_folder} bin/micromamba")
    environment = dict(os.environ, MAMBA_ROOT_PREFIX="/tmp/mamba")
    subprocess.run(f"{micromamba_folder}/bin/micromamba create -y -q -p {fsl_folder} "
                   "-c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ -c conda-forge fsl-topup",
                   shell=True, check=True, env=environment)
    return fsl_folder


# ================================================================ task weights (Buergi et al.)
def spm_hrf(repetition_time, oversampling=16, length_s=32.0):
    """Return the SPM canonical haemodynamic response function (HRF), normalised to unit sum.

    The HRF is sampled every repetition_time / oversampling seconds.
    Choice: SPM default parameters (peak at 6 s, undershoot at 16 s, peak-to-undershoot ratio 6) and 16 microtime bins
    per volume, as in SPM and CONN (Friston et al. 1998).
    """
    step = repetition_time / oversampling
    peak_delay, undershoot_delay, peak_dispersion, undershoot_dispersion, peak_undershoot_ratio, onset = 6.0, 16.0, 1.0, 1.0, 6.0, 0.0
    times = np.arange(0, int(length_s / step) + 1, dtype=float) - onset / step
    peak = stats.gamma.pdf(times, a=peak_delay / peak_dispersion, scale=peak_dispersion / step)
    undershoot = stats.gamma.pdf(times, a=undershoot_delay / undershoot_dispersion, scale=undershoot_dispersion / step)
    response = peak - undershoot / peak_undershoot_ratio
    total = response.sum()
    if not np.isfinite(total) or total == 0:
        raise ValueError("SPM canonical HRF normalization failed.")
    return response / total


def convolved_boxcar(n_volumes, repetition_time, onsets, durations, oversampling=16):
    """Convolve the event boxcar on the microtime grid with the HRF (onsets in seconds from the first retained volume)."""
    step = repetition_time / oversampling
    times = np.arange(0, n_volumes * repetition_time, step)
    boxcar = np.zeros(times.size, dtype=float)
    for onset, duration in zip(onsets, durations):
        boxcar[(times >= onset) & (times < onset + duration)] = 1.0
    return signal.fftconvolve(boxcar, spm_hrf(repetition_time, oversampling), mode="full")[:boxcar.size]


def task_weights(n_volumes, repetition_time, onsets, durations, oversampling=16):
    """Compute the connectivity weight of each volume: the rectified, HRF-convolved event boxcar.

    Choice: rectification as in Buergi et al. ("convolved with an SPM canonical HRF and rectified").
    The weights are sampled at the start of each volume (microtime bin 0).
    """
    rectified = np.maximum(convolved_boxcar(n_volumes, repetition_time, onsets, durations, oversampling), 0.0)
    return rectified[np.arange(n_volumes) * oversampling]


def task_regressor(n_volumes, repetition_time, onsets, durations, oversampling=16):
    """Compute a nuisance regressor: the HRF-convolved event boxcar, one value per volume, without rectification."""
    return convolved_boxcar(n_volumes, repetition_time, onsets, durations, oversampling)[np.arange(n_volumes) * oversampling]


def nuisance_task_regressors(events, n_volumes, repetition_time):
    """Compute the session regressor and one regressor per task phase (columns: session, then NUISANCE_EVENTS).

    Choice: the session regressor is a boxcar over the whole run convolved with the HRF (the CONN effect of session),
    which models the signal increase at the start of each run.
    """
    regressors = [task_regressor(n_volumes, repetition_time, [0.0], [n_volumes * repetition_time])]
    for phase in NUISANCE_EVENTS:
        phase_events = events[events["trial_type"] == phase]
        regressors.append(task_regressor(n_volumes, repetition_time, phase_events["onset"], phase_events["duration"]))
    return np.column_stack(regressors)


def network_weights(events, n_volumes, repetition_time):
    """Compute the volume weights for network C (all weighting events) and for each payoff layer of the A networks.

    events: BIDS events with onsets relative to the first retained volume. Returns (weights, {layer: weights}).
    """
    first_payoffs = add_payoff_layer_columns(events[events["trial_type"] == WEIGHTING_EVENT])
    all_weights = task_weights(n_volumes, repetition_time, first_payoffs["onset"], first_payoffs["duration"])
    layer_weights = {}
    for layer, layer_events in first_payoffs.groupby(list(LAYER_FACTORS)):
        layer_weights[layer] = task_weights(n_volumes, repetition_time, layer_events["onset"], layer_events["duration"])
    return all_weights, layer_weights


def order_weights(events, n_volumes, repetition_time):
    """Compute the first-payoff volume weights for each presentation order, as for network C.

    events: BIDS events with onsets relative to the first retained volume. Returns {order: weights}; an order without
    trials in the run is missing.
    """
    first_payoffs = add_payoff_layer_columns(events[events["trial_type"] == WEIGHTING_EVENT])
    return {order: task_weights(n_volumes, repetition_time, order_events["onset"], order_events["duration"])
            for order, order_events in first_payoffs.groupby("order")}


# ================================================================ denoising (Buergi et al.; CONN)
def compcor(bold, mask, n_components=N_COMPCOR):
    """Compute CONN-style CompCor regressors for one tissue mask.

    The regressors are the mean signal within the mask and the n - 1 largest principal components orthogonal to it.
    Choice: as in Buergi et al. ("the average BOLD signal as well as the largest principal components orthogonal to the
    BOLD average within each participant's eroded segmentation masks"; Behzadi et al. 2007).
    """
    voxel_signals = bold[mask].T.astype(float)                  # time x voxels
    voxel_signals = voxel_signals - voxel_signals.mean(axis=0)
    mean_signal = voxel_signals.mean(axis=1, keepdims=True)
    loadings = np.linalg.lstsq(mean_signal, voxel_signals, rcond=None)[0]
    residual = voxel_signals - mean_signal @ loadings
    components, singular_values, _ = np.linalg.svd(residual, full_matrices=False)
    return np.column_stack([mean_signal[:, 0], components[:, :n_components - 1]])


def compcor_columns(confounds, tissue, n_components=N_COMPCOR):
    """Return the CompCor column names of one tissue ('wm' or 'csf') in a confounds table."""
    columns = [name for name in confounds.columns if tissue.lower() in name.lower() and "comp" in name.lower()]
    if len(columns) >= n_components:
        return columns[:n_components]
    raise ValueError(f"Could not identify {n_components} {tissue.upper()} CompCor columns in the confounds table.")


def nuisance_design(confounds, task_effects, n_components=N_COMPCOR):
    """Build the nuisance design matrix.

    Columns: constant, white matter and CSF CompCor components, each task regressor with its first difference, and a
    linear trend.
    Choice: no motion regressors and no scrubbing, as in the connectivity denoising of Buergi et al.; the CONN default
    adds 12 realignment regressors. Head motion is handled by excluding participants with the stringent criteria of
    Parkes et al. (2018) in paper_results.py.
    """
    columns = [np.ones(len(confounds))]
    for name in compcor_columns(confounds, "wm", n_components) + compcor_columns(confounds, "csf", n_components):
        values = pd.to_numeric(confounds[name], errors="coerce").to_numpy(float)
        finite_values = values[np.isfinite(values)]
        fill_value = np.median(finite_values) if finite_values.size else 0.0
        columns.append(np.nan_to_num(values, nan=fill_value))
    task_effects = np.asarray(task_effects, dtype=float)
    if task_effects.ndim == 1:
        task_effects = task_effects[:, None]
    if task_effects.shape[0] != len(confounds):
        raise ValueError("task_effects must have one row per volume.")
    for index in range(task_effects.shape[1]):
        regressor = task_effects[:, index]
        columns.append(regressor)
        columns.append(np.r_[0.0, np.diff(regressor)])           # first-order derivative (Buergi et al.)
    columns.append(np.linspace(-1.0, 1.0, len(confounds)))      # linear trend within the run
    return np.column_stack(columns)


def remove_nuisance(signals, design):
    """Return the residuals of an ordinary least-squares fit of the nuisance design matrix."""
    coefficients, *_ = np.linalg.lstsq(design, signals, rcond=None)
    return signals - design @ coefficients


def band_pass(signals, repetition_time, low_hz=LOW_HZ, high_hz=HIGH_HZ):
    """Band-pass filter the time series.

    Choice: second-order Butterworth filter applied forward and backward (zero phase, which avoids a temporal shift
    relative to the task weights), after the nuisance regression, as in Buergi et al.
    """
    nyquist_hz = 0.5 / repetition_time
    if high_hz >= nyquist_hz:
        raise ValueError(f"High cutoff {high_hz} Hz must be below the Nyquist frequency {nyquist_hz:.4f} Hz.")
    filter_sections = signal.butter(2, [low_hz / nyquist_hz, high_hz / nyquist_hz], btype="bandpass", output="sos")
    return signal.sosfiltfilt(filter_sections, signals, axis=0)


def region_signals(bold, atlas, voxel_size_mm, n_regions, smoothing_fwhm_mm=SMOOTHING_FWHM_MM):
    """Compute the regional time series (time x regions): the mean signal of each region, with optional Gaussian
    smoothing.

    Choice: unsmoothed data (smoothing_fwhm_mm=0), the CONN default for regional time series, because smoothing
    mixes signal between neighbouring regions; the 6 mm version is a robustness check.
    """
    if smoothing_fwhm_mm is not None and smoothing_fwhm_mm > 0:
        sigma_mm = smoothing_fwhm_mm / np.sqrt(8.0 * np.log(2.0))
        sigma_voxels = tuple(sigma_mm / float(size) for size in voxel_size_mm) + (0.0,)
        bold = ndimage.gaussian_filter(bold, sigma=sigma_voxels, mode="constant")
    signals = np.empty((bold.shape[3], n_regions), dtype=float)
    for region in range(1, n_regions + 1):
        region_mask = atlas == region
        if not region_mask.any():
            raise ValueError(f"Region {region} has no voxels on this grid.")
        signals[:, region - 1] = np.nanmean(bold[region_mask, :], axis=0)
    return signals


def clean_run(bold_on_grid, framewise_displacement, events, repetition_time, space, white_matter_mask, csf_mask):
    """Denoise one run on the atlas grid (without dummy volumes) and compute the regional time series and the network
    weights.

    events: BIDS events of the run (onsets from the first acquired volume); space: atlas and template on the atlas grid
    (atlas_and_template); white_matter_mask, csf_mask: eroded tissue masks on the atlas grid.
    """
    n_volumes = bold_on_grid.shape[3]
    atlas, n_regions = space["atlas"], space["n_regions"]
    # Choice: CompCor is restricted to voxels within the EPI field of view, defined by a mean signal above 10% of the
    # robust maximum (99th percentile within the template brain mask); at least 100 voxels per tissue are required.
    mean_bold = bold_on_grid.mean(axis=3)
    in_field_of_view = mean_bold > 0.1 * np.percentile(mean_bold[space["brain_on_grid"]], 99)
    region_coverage = np.array([in_field_of_view[atlas == region].mean() for region in range(1, n_regions + 1)])

    white_matter_voxels, csf_voxels = white_matter_mask & in_field_of_view, csf_mask & in_field_of_view
    for name, mask in (("white matter", white_matter_voxels), ("CSF", csf_voxels)):
        if mask.sum() < 100:
            raise RuntimeError(f"{name} CompCor mask inside the EPI field of view has {int(mask.sum())} voxels")
    white_matter_compcor = compcor(bold_on_grid, white_matter_voxels)
    csf_compcor = compcor(bold_on_grid, csf_voxels)
    confounds = pd.DataFrame(np.column_stack([white_matter_compcor, csf_compcor]),
                             columns=[f"wm_compcor_{index:02d}" for index in range(N_COMPCOR)] +
                                     [f"csf_compcor_{index:02d}" for index in range(N_COMPCOR)])
    confounds["framewise_displacement"] = framewise_displacement

    events = events.copy()
    events["onset"] = events["onset"] - DUMMY_VOLUMES * repetition_time      # onsets relative to the first retained volume
    task_effects = nuisance_task_regressors(events, n_volumes, repetition_time)
    effect_names = ["task_session"] + [f"task_{phase.replace(' ', '')}" for phase in NUISANCE_EVENTS]
    for name, column in zip(effect_names, task_effects.T):
        confounds[name] = column

    voxel_size = space["atlas_image"].header.get_zooms()[:3]
    signals_smoothed = region_signals(bold_on_grid, atlas, voxel_size, n_regions)                        # 6 mm smoothing
    signals_unsmoothed = region_signals(bold_on_grid, atlas, voxel_size, n_regions, smoothing_fwhm_mm=0)  # CONN default
    design = nuisance_design(confounds, task_effects)
    clean_smoothed = band_pass(remove_nuisance(signals_smoothed, design), repetition_time)
    clean_unsmoothed = band_pass(remove_nuisance(signals_unsmoothed, design), repetition_time)

    all_weights, layer_weights = network_weights(events, n_volumes, repetition_time)
    temporal_snr = mean_bold / np.maximum(bold_on_grid.std(axis=3), 1e-6)
    return {"clean_unsmoothed": clean_unsmoothed, "clean_smoothed": clean_smoothed, "weights": all_weights,
            "layer_weights": layer_weights, "signals_unsmoothed": signals_unsmoothed, "signals_smoothed": signals_smoothed,
            "confounds": confounds, "region_coverage": region_coverage, "mean_bold": mean_bold,
            "temporal_snr": temporal_snr, "compcor_voxels": {"wm": int(white_matter_voxels.sum()), "csf": int(csf_voxels.sum())}}


# ================================================================ networks
def weighted_correlation(signal_a, signal_b, weights):
    """Compute the weighted Pearson correlation of two time series.

    Weights must be non-negative; NaN values and zero weights are ignored.
    """
    signal_a = np.asarray(signal_a, float)
    signal_b = np.asarray(signal_b, float)
    weights = np.asarray(weights, float)
    valid = np.isfinite(signal_a) & np.isfinite(signal_b) & np.isfinite(weights) & (weights > 0)
    signal_a, signal_b, weights = signal_a[valid], signal_b[valid], weights[valid]
    if signal_a.size < 3 or weights.sum() <= 0:
        return np.nan
    weights = weights / weights.sum()
    mean_a, mean_b = np.sum(weights * signal_a), np.sum(weights * signal_b)
    deviation_a, deviation_b = signal_a - mean_a, signal_b - mean_b
    variance_a, variance_b = np.sum(weights * deviation_a**2), np.sum(weights * deviation_b**2)
    if variance_a <= 0 or variance_b <= 0:
        return np.nan
    return np.sum(weights * deviation_a * deviation_b) / np.sqrt(variance_a * variance_b)


def network_from_signals(signals, weights):
    """Compute the region-by-region network of Fisher z-transformed weighted correlations, with an empty diagonal.

    Buergi et al.: "Fisher-transformed bivariate correlation coefficients from a weighted general linear model".
    Choice: correlations are clipped at +-0.999999 before the Fisher transform, which keeps z finite (|z| <= 7.25).
    """
    n_regions = signals.shape[1]
    network = np.zeros((n_regions, n_regions), dtype=float)
    np.fill_diagonal(network, np.nan)
    for region_a in range(n_regions):
        for region_b in range(region_a + 1, n_regions):
            correlation = weighted_correlation(signals[:, region_a], signals[:, region_b], weights)
            fisher_z = np.arctanh(np.clip(correlation, -0.999999, 0.999999)) if np.isfinite(correlation) else np.nan
            network[region_a, region_b] = network[region_b, region_a] = fisher_z
    return network


def network_from_runs(signals_per_run, weights_per_run):
    """Compute one network from several runs by concatenating their time series and weights.

    Choice: each run is denoised separately and the runs are then concatenated, so that all trials of a condition
    contribute equally; per-run networks are not averaged.
    """
    return network_from_signals(np.concatenate(signals_per_run, axis=0), np.concatenate(weights_per_run, axis=0))


def participant_networks(run_results, version):
    """Compute the B (per run), A (per payoff layer) and C networks of one participant as {file label: network}.

    version: 'clean_unsmoothed' (CONN default) or 'clean_smoothed'. A run without trials of a layer contributes
    zero weight to that layer.
    """
    runs = sorted(run_results)
    layers = sorted({layer for run in runs for layer in run_results[run]["layer_weights"]})
    signals_per_run = [run_results[run][version] for run in runs]
    networks = {}
    for run in runs:
        networks[f"network-B_run-{run}"] = network_from_signals(run_results[run][version], run_results[run]["weights"])
    for layer in layers:
        layer_weights = [run_results[run]["layer_weights"].get(layer, np.zeros(len(run_results[run]["weights"]))) for run in runs]
        networks[f"network-A_{layer_name(layer)}"] = network_from_runs(signals_per_run, layer_weights)
    networks["network-C"] = network_from_runs(signals_per_run, [run_results[run]["weights"] for run in runs])
    return networks


def parse_file_name(file_name):
    """Parse the key-value parts of a BIDS-style file name.

    Example: 'sub-07_ses-3t2_run-1_events.tsv' -> {'sub': '07', 'ses': '3t2', 'run': '1'}.
    """
    parts = {}
    for part in file_name.split("_"):
        key, dash, value = part.partition("-")
        if dash:
            parts[key] = value
    return parts


# ================================================================ processing of all participants
HERE = Path(__file__).resolve().parent
# (time series, output folder, file suffix); the smoothed networks are written last, so their network C marks a
# completed participant
NETWORK_VERSIONS = (("clean_unsmoothed", "networks_unsmoothed", "_connectivity-unsmoothed_z.csv"),
                    ("clean_smoothed", "networks", "_connectivity_z.csv"))


def log(message):
    """Print a message with the current time."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bids", required=True, type=Path, help="BIDS dataset folder (events; for option 1 also the images)")
    parser.add_argument("--fmriprep", type=Path, help="fMRIPrep output folder (option 2)")
    parser.add_argument("--out", required=True, type=Path, help="output folder")
    parser.add_argument("--participants", nargs="+", help="participant labels without 'sub-' (default: all participants)")
    parser.add_argument("--session", default=SESSIONS["3t"], help="3t2 (3T) or 7t2 (7T)")
    parser.add_argument("--atlas", type=Path, default=HERE / "atlas" / "risk_atlas.nii.gz", help="atlas image")
    parser.add_argument("--regions", type=Path, default=HERE / "atlas" / "risk_atlas_regions.csv", help="region table of the atlas")
    parser.add_argument("--templateflow-dir", type=Path, help="folder with the 1 mm TemplateFlow files (default: download)")
    parser.add_argument("--fsl-dir", type=Path, default=os.environ.get("FSLDIR"),
                        help="FSL folder with topup for distortion correction (option 1); without FSL, distortion "
                             "correction is skipped")
    parser.add_argument("--download", action="store_true", help="download missing participant files from OpenNeuro")
    parser.add_argument("--work", type=Path, default=Path("/tmp/task_weighted_networks_work"), help="folder for intermediate files")
    parser.add_argument("--save-bold", action="store_true", help="save the BOLD data on the atlas grid (brain voxels, float16)")
    return parser.parse_args()


def bids_run_files(bids_folder, participant, session, task, run):
    """Return the paths of the raw BOLD data, sidecar, events and reverse phase-encoding EPI of one run."""
    session_folder = bids_folder / f"sub-{participant}" / f"ses-{session}"
    stem = session_folder / "func" / f"sub-{participant}_ses-{session}_task-{task}_run-{run}"
    run_files = {"bold": Path(f"{stem}_bold.nii.gz"), "bold_json": Path(f"{stem}_bold.json"),
                 "events": Path(f"{stem}_events.tsv"), "fieldmap": None, "fieldmap_json": None}
    fieldmap_folder = session_folder / "fmap"
    fieldmaps = [fieldmap_folder / f"sub-{participant}_ses-{session}_dir-{direction}_run-{run}_epi.json"
                 for direction in FIELDMAP_DIRECTIONS]
    fieldmaps += sorted(fieldmap_folder.glob(f"sub-{participant}_ses-{session}_*run-{run}_epi.json"))
    fieldmaps = [path for path in fieldmaps if path.exists()]
    if fieldmaps:
        run_files["fieldmap_json"] = fieldmaps[0]
        run_files["fieldmap"] = Path(str(fieldmaps[0]).replace(".json", ".nii.gz"))
    return run_files


def download_participant(bids_folder, participant, session):
    """Download the files of one participant from OpenNeuro.

    Files: T1w image, BOLD runs, sidecars, events and reverse phase-encoding EPIs.
    """
    import urllib.request
    import urllib.error

    def fetch(relative_path, required=True):
        """Download one file unless it exists; return None if an optional file is not available."""
        destination = bids_folder / relative_path
        if destination.exists() and destination.stat().st_size > 0:
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(f"{OPENNEURO_URL}/{relative_path}", destination)
                return destination
            except urllib.error.HTTPError:
                destination.unlink(missing_ok=True)
                if required:
                    raise
                return None
            except Exception:
                destination.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(10)

    label = f"sub-{participant}"
    fetch(f"{label}/ses-{ANAT_SESSION}/anat/{label}_ses-{ANAT_SESSION}_T1w.nii.gz")
    for run in RUNS:
        stem = f"{label}/ses-{session}/func/{label}_ses-{session}_task-{TASK}_run-{run}"
        for ending in ("_bold.nii.gz", "_bold.json", "_events.tsv"):
            fetch(stem + ending)
        for direction in FIELDMAP_DIRECTIONS:
            fieldmap_stem = f"{label}/ses-{session}/fmap/{label}_ses-{session}_dir-{direction}_run-{run}_epi"
            if fetch(fieldmap_stem + ".json", required=False) is not None:
                fetch(fieldmap_stem + ".nii.gz")
                break


def write_run_outputs(prefix, run_result, events_path, space, save_bold):
    """Write the outputs of one run.

    Outputs: regional time series (raw and denoised, smoothed and unsmoothed), confounds with the weights, events,
    region coverage, and mean BOLD and temporal SNR images.
    """
    columns = [f"roi{region:02d}" for region in range(1, space["n_regions"] + 1)]
    for name, key in (("roi_timeseries", "signals_smoothed"), ("roi_timeseries_cleaned", "clean_smoothed"),
                      ("roi_timeseries_unsmoothed", "signals_unsmoothed"), ("roi_timeseries_unsmoothed_cleaned", "clean_unsmoothed")):
        pd.DataFrame(run_result[key], columns=columns).to_csv(f"{prefix}_{name}.tsv", sep="\t", index=False)
    run_result["confounds"].assign(weights_first_payoff=run_result["weights"]).to_csv(f"{prefix}_confounds.tsv", sep="\t", index=False)
    shutil.copy(events_path, f"{prefix}_events.tsv")       # original events, which allow other layer definitions
    pd.DataFrame({"roi": range(1, space["n_regions"] + 1), "coverage": run_result["region_coverage"]}).to_csv(
        f"{prefix}_roi_coverage.tsv", sep="\t", index=False)
    if save_bold:
        np.savez_compressed(f"{prefix}_bold_mni_brain_float16.npz",
                            data=run_result["bold_on_grid"][space["brain_on_grid"]].astype(np.float16),
                            mask_shape=np.array(space["atlas"].shape))
    affine = space["atlas_image"].affine
    nib.Nifti1Image((run_result["temporal_snr"] * space["brain_on_grid"]).astype(np.float32), affine).to_filename(f"{prefix}_tsnr_mni.nii.gz")
    nib.Nifti1Image(run_result["mean_bold"], affine).to_filename(f"{prefix}_mean_bold_mni.nii.gz")


def save_networks(label, run_results, participant_folder, space):
    """Write all networks of one participant in both versions."""
    region_names = [f"roi{region:02d}" for region in range(1, space["n_regions"] + 1)]
    for version, folder, suffix in NETWORK_VERSIONS:
        network_folder = participant_folder / folder
        network_folder.mkdir(parents=True, exist_ok=True)
        networks = participant_networks(run_results, version)
        for name, network in networks.items():
            pd.DataFrame(network, index=region_names, columns=region_names).to_csv(network_folder / f"{label}_{name}{suffix}")
    log(f"{label}: networks saved ({len(run_results)} runs)")


def process_participant(participant, args, space):
    """Preprocess and denoise the runs of one participant, compute the networks and write a summary."""
    session = args.session
    label = f"sub-{participant}_ses-{session}"
    participant_folder = args.out / label
    runs_folder = participant_folder / "runs"
    runs_folder.mkdir(parents=True, exist_ok=True)
    time_start = time.time()
    if args.download and not args.fmriprep:
        download_participant(args.bids, participant, session)
    if args.fmriprep:
        white_matter_mask, csf_mask = fmriprep_tissue_masks(args.fmriprep, participant, TEMPLATE, space)
        anatomy, anatomy_summary = None, {"source": "fMRIPrep", "wm_mask_voxels": int(white_matter_mask.sum()),
                                          "csf_mask_voxels": int(csf_mask.sum())}
    else:
        t1_path = args.bids / f"sub-{participant}" / f"ses-{ANAT_SESSION}" / "anat" / f"sub-{participant}_ses-{ANAT_SESSION}_T1w.nii.gz"
        anatomy = preprocess_anatomy(t1_path, space)
        white_matter_mask, csf_mask, anatomy_summary = anatomy["white_matter_mask"], anatomy["csf_mask"], anatomy["summary"]
    log(f"{label}: anatomical preprocessing finished; white matter mask {int(white_matter_mask.sum())} voxels, "
        f"CSF mask {int(csf_mask.sum())} voxels")

    run_results = {}
    for run in RUNS:
        run_start = time.time()
        run_files = bids_run_files(args.bids, participant, session, TASK, run)
        try:
            if args.fmriprep:
                bold_on_grid, framewise_displacement, repetition_time, preprocessing = read_fmriprep_run(
                    args.fmriprep, participant, session, TASK, run, TEMPLATE, space, DUMMY_VOLUMES)
            else:
                bold_on_grid, framewise_displacement, repetition_time, preprocessing = preprocess_run(
                    run_files, anatomy, space, args.work / label / f"run-{run}", DUMMY_VOLUMES, args.fsl_dir, log)
            run_result = clean_run(bold_on_grid, framewise_displacement, pd.read_csv(run_files["events"], sep="\t"),
                                   repetition_time, space, white_matter_mask, csf_mask)
            run_result["bold_on_grid"] = bold_on_grid
            write_run_outputs(runs_folder / f"{label}_run-{run}", run_result, run_files["events"], space, args.save_bold)
            run_result["summary"] = {"run": run, "volumes": int(bold_on_grid.shape[3]), "tr": repetition_time,
                                     "n_dummy_dropped": DUMMY_VOLUMES, **preprocessing,
                                     "mean_fd": float(framewise_displacement[1:].mean()),
                                     "max_fd": float(framewise_displacement.max()),
                                     "median_tsnr": float(np.median(run_result["temporal_snr"][space["brain_on_grid"]])),
                                     "min_roi_coverage": float(run_result["region_coverage"].min()),
                                     "compcor_voxels": run_result["compcor_voxels"],
                                     "minutes": round((time.time() - run_start) / 60, 1)}
            del run_result["bold_on_grid"], bold_on_grid
            run_results[run] = run_result
            summary = run_result["summary"]
            log(f"{label} run {run}: mean FD {summary['mean_fd']:.3f} mm, median tSNR {summary['median_tsnr']:.1f}, "
                f"lowest region coverage {summary['min_roi_coverage']:.2f}, {summary['minutes']} min")
        except Exception as error:
            log(f"{label} run {run} FAILED: {type(error).__name__}: {error}")
            traceback.print_exc()
        finally:
            shutil.rmtree(args.work / label / f"run-{run}", ignore_errors=True)
    if not run_results:
        raise RuntimeError("no run was processed successfully")
    save_networks(label, run_results, participant_folder, space)
    json.dump({"subject": participant, "session": session,
               "input": "fMRIPrep" if args.fmriprep else "raw BIDS", "anatomy": anatomy_summary,
               "runs": {run: run_results[run]["summary"] for run in sorted(run_results)},
               "minutes": round((time.time() - time_start) / 60, 1)},
              open(participant_folder / "subject_summary.json", "w"), indent=2, default=str)


def main():
    args = read_arguments()
    space = atlas_and_template(args.atlas, args.regions, TARGET_LABEL, TEMPLATE, args.templateflow_dir)
    log(f"atlas: {space['n_regions']} regions, target region {space['target_region']}; input: {'fMRIPrep' if args.fmriprep else 'raw BIDS'}")
    args.out.mkdir(parents=True, exist_ok=True)
    for participant in args.participants or PARTICIPANTS:
        finished = args.out / f"sub-{participant}_ses-{args.session}" / "networks" / \
            f"sub-{participant}_ses-{args.session}_network-C_connectivity_z.csv"
        if finished.exists():
            log(f"sub-{participant}: networks already exist; skipped")
            continue
        try:
            process_participant(participant, args, space)
        except Exception as error:
            log(f"sub-{participant} FAILED: {type(error).__name__}: {error}")
            traceback.print_exc()
        finally:
            shutil.rmtree(args.work / f"sub-{participant}_ses-{args.session}", ignore_errors=True)


if __name__ == "__main__":
    main()
