"""Risk network atlas: regions from a Neurosynth meta-analysis map, divided by spectral clustering.

The procedure follows Buergi et al. (2026), who derived their network from a Neurosynth map. Steps:
  1. Neurosynth uniformity-test map for "risky" (FDR 0.01; Yarkoni et al. 2011)
  2. clusters of connected suprathreshold voxels (6-connectivity) with more than 50 voxels
  3. spectral clustering of each cluster (spectral_clustering.py): number of clusters by eigengap, then KNSC-Ncut
  4. region labels from the AAL3 area with the largest overlap (Rolls et al. 2020); repeated labels are numbered
     _1, _2, ...
The spectral clusters are used as regions without manual recombination. With the default settings, the script
reproduces atlas/ voxel for voxel.

Any thresholded map can be used as input.
Outputs: <name>_atlas.nii.gz (regions numbered 1 to N) and <name>_atlas_regions.csv with one row per region: number,
label, size, centre (MNI, mm), overlapping AAL3 areas, source cluster and clustering results.

usage: python risk_brain_network_atlas.py --map risky_uniformity-test_z_FDR_0.01.nii.gz --aal3-zip AAL3v1_for_SPM12.zip
                                          --out atlas_rebuilt [--name risk] [--method knsc_ncut | knsc_rcut]
"""
import argparse
import csv
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from spectral_clustering import ALPHA, METHODS, MU, gaussian_kernel, knsc_clustering, number_of_parts_by_eigengap

MIN_CLUSTER_VOXELS = 50   # Clusters with more than 50 voxels, as in Buergi et al. ("k > 50")


def read_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", required=True, type=Path, help="thresholded meta-analysis map, e.g. Neurosynth 'risky'")
    parser.add_argument("--aal3-zip", required=True, type=Path, help="AAL3v1_for_SPM12.zip (used only for the region labels)")
    parser.add_argument("--out", required=True, type=Path, help="output folder")
    parser.add_argument("--name", default="risk", help="prefix of the output files <name>_atlas.nii.gz and <name>_atlas_regions.csv")
    parser.add_argument("--method", choices=sorted(METHODS), default="knsc_ncut", help="clustering method")
    parser.add_argument("--max-parts", type=int, default=10, help="maximum number of spectral clusters per cluster")
    parser.add_argument("--iterations", type=int, default=300, help="number of KNSC iterations")
    parser.add_argument("--tolerance", type=float, default=float("-inf"),
                        help="relative convergence tolerance; -inf runs all iterations")
    parser.add_argument("--seed", type=int, default=20260922, help="random seed; cluster c is initialised with seed + c")
    return parser.parse_args()


def read_aal3(aal3_zip, folder):
    """Read the AAL3 image and the area labels {number: label} from the AAL3 zip file.

    The labels include the area number, e.g. 'Insula_L_33'.
    """
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(aal3_zip, "r") as archive:
        archive.extractall(folder)
    images = list(folder.rglob("AAL3v1.nii.gz")) or list(folder.rglob("AAL3v1.nii"))
    name_files = list(folder.rglob("AAL3v1.nii.txt"))
    if len(images) != 1 or len(name_files) != 1:
        raise RuntimeError(f"expected one AAL3 image and one label file, found {images} and {name_files}")
    area_names = {0: "Outside_AAL3"}
    with open(name_files[0], encoding="utf-8", errors="replace") as lines:
        for line in lines:
            words = line.strip().replace(",", " ").split()
            for position, word in enumerate(words):
                if word.isdigit():
                    number = int(word)
                    area_names[number] = "_".join(words[position + 1:]) if position + 1 < len(words) else f"AAL3_{number}"
                    break
    return nib.load(str(images[0])), area_names


def aal3_overlap(region_voxels, aal3_data, area_names):
    """Return the AAL3 areas within a region, sorted by overlap, as 'area:voxels(fraction); ...' (at most 15)."""
    areas = aal3_data[tuple(region_voxels.T)].astype(int)
    areas = areas[areas > 0]
    if not len(areas):
        return ""
    numbers, counts = np.unique(areas, return_counts=True)
    largest_first = np.argsort(counts)[::-1]
    return "; ".join(f"{area_names.get(int(numbers[index]), f'AAL3_{int(numbers[index])}')}:{int(counts[index])}"
                     f"({counts[index] / counts.sum():.3f})" for index in largest_first[:15])


def region_names(overlaps):
    """Return the region labels: the AAL3 area with the largest overlap, without the area number.

    Repeated labels are numbered _1, _2, ...
    """
    base_names = []
    for overlap in overlaps:
        largest_area = overlap.split(";")[0].split(":")[0].strip()
        name, _, number = largest_area.rpartition("_")
        base_names.append(name if number.isdigit() else largest_area)
    names, seen = [], {}
    for name in base_names:
        if base_names.count(name) > 1:
            seen[name] = seen.get(name, 0) + 1
            names.append(f"{name}_{seen[name]}")
        else:
            names.append(name)
    return names


def write_table(path, rows):
    """Write a list of dictionaries as a CSV table."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = read_arguments()
    output_dir = args.out.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_analysis = nib.load(str(args.map.resolve()))
    aal3_image, area_names = read_aal3(args.aal3_zip.resolve(), output_dir / "_aal3_extracted")
    if meta_analysis.shape[:3] != aal3_image.shape[:3] or not np.allclose(meta_analysis.affine, aal3_image.affine, atol=1e-5):
        raise RuntimeError("The map and AAL3 are not on the same grid; resample one of them first.")
    map_data = np.asarray(meta_analysis.dataobj)
    aal3_data = np.asarray(aal3_image.dataobj)

    # steps 1-2: clusters of connected suprathreshold voxels (6-connectivity), sorted by size
    significant = np.isfinite(map_data) & (map_data != 0)
    cluster_image, n_clusters = ndimage.label(significant, structure=ndimage.generate_binary_structure(3, 1))
    cluster_sizes = np.bincount(cluster_image.ravel())
    retained = sorted((cluster for cluster in range(1, n_clusters + 1) if cluster_sizes[cluster] > MIN_CLUSTER_VOXELS),
                      key=lambda cluster: cluster_sizes[cluster], reverse=True)
    print(f"connected clusters > {MIN_CLUSTER_VOXELS} voxels: {len(retained)}", flush=True)

    # steps 3-4: spectral clustering of each cluster, then region labels and descriptions
    atlas = np.zeros(map_data.shape, dtype=np.int16)
    regions, region_number = [], 0
    for cluster_rank, cluster_label in enumerate(retained, 1):
        cluster_voxels = np.argwhere(cluster_image == cluster_label)
        sigma, kernel = gaussian_kernel(nib.affines.apply_affine(meta_analysis.affine, cluster_voxels))
        n_parts, eigenvalues, eigengaps = number_of_parts_by_eigengap(kernel, args.max_parts)
        if n_parts == 1:
            part_of_voxel, objective, iterations = np.zeros(len(cluster_voxels), dtype=int), None, 0
        else:
            part_of_voxel, objective, iterations = knsc_clustering(kernel, n_parts, seed=args.seed + cluster_rank,
                                                                   iterations=args.iterations, tolerance=args.tolerance,
                                                                   method=args.method)
        print(f"cluster {cluster_rank:02d}: {len(cluster_voxels)} voxels, sigma {sigma:.2f} mm, "
              f"{n_parts} spectral cluster(s)", flush=True)
        for part in range(n_parts):
            region_voxels = cluster_voxels[part_of_voxel == part]
            if not len(region_voxels):
                continue
            region_number += 1
            atlas[tuple(region_voxels.T)] = region_number
            centre = nib.affines.apply_affine(meta_analysis.affine, region_voxels.mean(axis=0))
            regions.append({"roi": region_number, "label": "", "voxels": len(region_voxels),
                            "center_x": round(float(centre[0]), 2), "center_y": round(float(centre[1]), 2),
                            "center_z": round(float(centre[2]), 2),
                            "aal3_regions": aal3_overlap(region_voxels, aal3_data, area_names),
                            "neurosynth_cluster": cluster_rank, "part_of_cluster": part + 1,
                            "cluster_voxels": len(cluster_voxels), "cluster_parts": n_parts, "kernel_sigma_mm": sigma,
                            "knsc_iterations": iterations, "knsc_objective": "" if objective is None else objective,
                            "cluster_eigenvalues": ";".join(f"{value:.8g}" for value in eigenvalues),
                            "cluster_eigengaps": ";".join(f"{value:.8g}" for value in eigengaps)})
    for region, name in zip(regions, region_names([region["aal3_regions"] for region in regions])):
        region["label"] = name

    nib.save(nib.Nifti1Image(atlas, meta_analysis.affine, meta_analysis.header), str(output_dir / f"{args.name}_atlas.nii.gz"))
    write_table(output_dir / f"{args.name}_atlas_regions.csv", regions)
    print(f"{region_number} regions written to {output_dir}; map {args.map.name}, clusters > {MIN_CLUSTER_VOXELS} voxels, "
          f"at most {args.max_parts} spectral clusters by eigengap, {args.method} (alpha {ALPHA}, mu {MU}, "
          f"{args.iterations} iterations, seed {args.seed})", flush=True)


if __name__ == "__main__":
    main()
