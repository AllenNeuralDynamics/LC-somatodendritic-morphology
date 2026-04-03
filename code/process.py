
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aind_exaspim_soma_detection import soma_proposal_generation as spg
from aind_exaspim_soma_detection.utils import img_util, util

### Subroutines
def get_soma_center(patch, n_cells=1):
    # Proposal generation
    proposals_1 = spg.detect_blobs(patch, 100, 16, 0)
    proposals_2 = spg.detect_blobs(patch, 100, 10, 0)
    proposals_3 = spg.detect_blobs(patch, 100, 5, 0)
    proposals = proposals_1 + proposals_2 + proposals_3

    # Filter proposals
    proposals = spg.spatial_filtering(proposals, 10)
    # print(f"{len(proposals)} proposals found")

    # pick brightest
    # top = spg.brightness_filtering(patch, proposals, n_cells)[0]

    # pick closest to guess
    top = proposals[np.argmin([np.linalg.norm(x - np.array(patch.shape)/2) for x in proposals ])]
    return top


def get_nbhd(patch, voxel, r=30):
    x0, y0, z0 = tuple(map(int, voxel))
    x_min, x_max = max(0, x0 - r), min(patch.shape[0], x0 + r + 1)
    y_min, y_max = max(0, y0 - r), min(patch.shape[1], y0 + r + 1)
    z_min, z_max = max(0, z0 - r), min(patch.shape[2], z0 + r + 1)
    return patch[x_min:x_max, y_min:y_max, z_min:z_max]


def fit_rotated_gaussian(nbhd):
    c = [s // 2 for s in nbhd.shape]
    voxels = np.stack(np.meshgrid(
        np.arange(nbhd.shape[0]),
        np.arange(nbhd.shape[1]),
        np.arange(nbhd.shape[2]),
        indexing='ij'
    ), -1).reshape(-1, 3)

    p0 = [c[0], c[1], c[2], 1e-2, 0, 0, 1e-2, 0, 1e-2, np.max(nbhd), np.min(nbhd)]

    try:
        params, _ = curve_fit(gaussian_3d_rotated, voxels, nbhd.ravel(), p0=p0)
    except RuntimeError:
        params = np.zeros((11,))
    return params, voxels


def gaussian_3d_rotated(coords, x0, y0, z0, a11, a12, a13, a22, a23, a33, A, B):
    # Refactor coordinates
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    dx = x - x0
    dy = y - y0
    dz = z - z0

    # Construct quadratic form
    quad = (
        a11*dx**2 + 2*a12*dx*dy + 2*a13*dx*dz +
        a22*dy**2 + 2*a23*dy*dz + a33*dz**2
    )
    return A * np.exp(-0.5 * quad) + B


def gaussian_fit_score(image_patch, fitted_patch, params, voxels):
    # Unpack parameters
    x0, y0, z0, a11, a12, a13, a22, a23, a33, A, B = params

    # Compute dx, dy, dz
    dx = voxels[:, 0] - x0
    dy = voxels[:, 1] - y0
    dz = voxels[:, 2] - z0

    # Compute quadratic form Q(x)
    quad = (
        a11 * dx**2 + 2 * a12 * dx * dy + 2 * a13 * dx * dz +
        a22 * dy**2 + 2 * a23 * dy * dz + a33 * dz**2
    )

    # Mask: only voxels within 2 std devs
    mask = quad <= 4

    # Flatten patches to align with voxels
    y_true = image_patch.ravel()[mask]
    y_pred = fitted_patch.ravel()[mask]

    # Compute R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0


def estimate_radii(params, anisotropy=(2.992, 2.992, 4.0)):
    # Compute precision matrix
    _, _, _, a11, a12, a13, a22, a23, a33, _, _ = params
    P_voxel = np.array([
        [a11, a12, a13],
        [a12, a22, a23],
        [a13, a23, a33]
    ])

    # Convert precision matrix from voxel to physical space
    S = np.diag(anisotropy)
    S_inv = np.linalg.inv(S)

    # Adjusted precision matrix in physical units
    # need to flip order
    P_physical = S_inv.T @ P_voxel @ S_inv
    try:
        cov_physical = np.linalg.inv(P_physical)
        # eigvals = np.linalg.eigvalsh(cov_physical)
        eigvals, eigvecs = np.linalg.eigh(cov_physical)
        radii = 2 * np.sqrt(np.abs(eigvals))
        return radii, eigvecs[::-1, :]
    except np.linalg.LinAlgError:
        return None


def run_shape_estimation(img, xyz, multiscale=2, patch_shape=(50, 50, 50)):
    # Get soma center
    voxel = img_util.to_voxels(xyz, multiscale)
    patch = img_util.get_patch(img, voxel, patch_shape)
    patch_filt = gaussian_filter(patch, sigma=1)
    soma_center = get_soma_center(patch_filt)
    voxel_offset = np.array(soma_center) - np.array(patch_shape)/2 
    soma_offset = np.array(img_util.to_physical(voxel_offset, multiscale))
    voxel_offset = np.rint(voxel_offset).astype(int)
    
    # Fit Gaussian
    # blob = get_nbhd(patch, soma_center, r=20)
    blob = img_util.get_patch(img, voxel+voxel_offset, patch_shape)
    params, voxels = fit_rotated_gaussian(blob)
    fitted_blob = gaussian_3d_rotated(voxels, *params).reshape(blob.shape)

    # Report results
    radii, eigvecs = estimate_radii(params)
    volume = np.prod(radii) * (4 / 3) * np.pi
    result = {
        "soma": xyz,
        "soma_offset": soma_offset,
        "radii": radii,
        "volume": volume,
        "rsquared": gaussian_fit_score(blob, fitted_blob, params, voxels),
        "primary_axis": eigvecs[:, 2],
    }
    return result, blob, fitted_blob
    
### Initializations
from glob import glob
records = {}
for swc in glob(f"/data/exaSPIM*reconstructions*/specimen_space_reconstructions/swc/*.swc"):
    id = swc.split("/")[-1][:11]
    soma = pd.read_csv(swc, sep="\s+", comment="#", nrows=1, names=["0","1","x","y","z","2","3"], usecols=["x","y","z"])
    record = soma.iloc[0].to_dict()
    # overwrite old with new
    record["id"] = id
    record["path"] = swc
    records[id] = record

somas_df = pd.DataFrame.from_records(list(records.values()), index="id")
somas_df["brain"] = somas_df.index.to_series().apply(lambda x: x.split("-")[-1])

### Images
import zarr

multiscale=2
imgs = dict()
for brain_id in somas_df["brain"].unique():
        # imgs[brain_id] = img_util.open_img(img_prefixes[brain_id] + str(multiscale))
    path = glob(f"/data/exaSPIM_{brain_id}*/fused.zarr/{multiscale}")[0]
    imgs[brain_id] = zarr.open(path, mode='r')

### Process Dataset

results = list()
for id in tqdm(somas_df.index[:]):
    # Extract info
    brain_id = id.split("-")[-1]
    xyz = somas_df.loc[id, ["x", "y", "z"]].values.astype("float")

    # Estimate soma shape
    # if brain_id in brains:
    res, _, _ = run_shape_estimation(imgs[brain_id], xyz)
    res["id"] = id
    results.append(res)


double_cells = [
    "N022-648434",
    "N018-685221",
    "N023-685221",
    "N040-685221",
    "N064-685221",
    "N065-685221",
    "N012-685222",
    "N013-685222",
]

df = pd.DataFrame(results).set_index("id").join(somas_df[["path", "brain"]])
df.loc[double_cells, ["radii", "volume", "rsquared", "primary_axis"]] = np.nan
df.to_csv("/scratch/LC_soma_shapes.csv")


#plots
import bipolarity
from utils import morphology_from_swc

names = ["aligned_bipolar_frac","aligned_bipolarity","abs_bipolarity", "cos_primary_axis"]
names_offset = [x+"_offset" for x in names]
df[names] = None
df[names_offset] = None
df["num_stems"] = None

for id in tqdm(df.index):
    swc = df.loc[id, "path"]
    morph = morphology_from_swc(swc)
    r2 = 50
    stems = bipolarity.intersection_points(morph, cutoff=r2)
    duplicates = []
    for i, stem in enumerate(stems[:len(stems)-1]):
        for j in range(i+1, len(stems)):
            if np.allclose(stem, stems[j], atol=2):
                duplicates.append(j)
    if len(duplicates) > 0:
        print(f"Removing {len(duplicates)} duplicate stems for {id}")
        stems = [s for i, s in enumerate(stems) if i not in duplicates]
        print(f"{len(stems)} stems remain after removing duplicates")
    df.loc[id, "num_stems"] = len(stems)

    soma = np.array(df.loc[id, "soma"])
    primary_axis = np.array(df.loc[id, "primary_axis"])
    df.loc[id, names] = bipolarity.get_bipolarity(soma, stems, primary_axis, cos2_cutoff=0.5)

ids = df.sort_values("abs_bipolarity", ascending=False).index.values
inds = [1, 40, 82, 129]
# inds=[1]
for i in inds:
    # print(i)
    bipolarity.plot_soma_all(df, ids[i])
    # plt.suptitle(f"{ids[i]}, b={df.loc[ids[i], 'abs_bipolarity']:.2f}")
    plt.suptitle(f"b={df.loc[ids[i], 'abs_bipolarity']:.2f}", ha="left", x=0.02)
    plt.savefig(f"/results/fig_s6b_bipolarity_image_{i}.svg")
    plt.show()