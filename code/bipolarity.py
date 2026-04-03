from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import zarr
import colorcet as cc
from glob import glob
import pandas as pd
from aind_exaspim_soma_detection.utils import img_util, util
from utils import read_swc


def plot_soma_all(somas_df, id):
    n = 250
    level = 0
    # Extract cell info
    brain_id = id.split("-")[-1]
    soma_xyz = np.array(somas_df.loc[id, "soma"])
    # soma_xyz = somas_df.loc[id, ["x", "y", "z"]].values.astype("float")

    soma_voxel = img_util.to_voxels(soma_xyz, level)
    path = glob(f"/data/exaSPIM_{brain_id}*/fused.zarr/{level}")[0]

    img = zarr.open(path, mode='r')
    lower = img_util.to_physical(soma_voxel-n/2, level)
    upper = img_util.to_physical(soma_voxel+n/2, level)

    patch = img_util.get_patch(img, soma_voxel, (n,n,n))
    vmax = np.percentile(patch, 99.9)
    # fig, axs = plt.subplots(1, 3, figsize=(10, 4))

    soma = np.array(somas_df.loc[id, "soma"])
    if np.isfinite(somas_df.loc[id, "volume"]):
        soma_fit = True
        soma_offset = soma + np.array(somas_df.loc[id, "soma_offset"])
        axis_offset = np.array(somas_df.loc[id, "primary_axis"]) * somas_df.loc[id, "radii"][2]/2
        primary = np.stack([soma-axis_offset, soma+axis_offset], -1)
        i = np.argmin(np.abs(somas_df.loc[id, "primary_axis"]))
    else:
        soma_fit = False
    i = 2

    swc = glob(f"/data/exaSPIM*reconstructions*/specimen_space_reconstructions/swc/{id}*.swc")[0]
    nodes = read_swc(swc, sep="\s+")
    nodes = nodes.query(
        f"x >= {lower[0]} & x <= {upper[0]} & "
        f"y >= {lower[1]} & y <= {upper[1]} & "
        f"z >= {lower[2]} & z <= {upper[2]}"
    ).to_numpy().T

    fig, axs = plt.subplots(1, 3, figsize=(8, 3), gridspec_kw={'width_ratios': [1, 1, 0.744]})
    for i in range(3):
        ax = axs[i]
        axes = list(range(3))
        axes.pop(i)
        axes = axes[::-1]

        # patch axes are reversed
        # remaining mip axes are reversed by imshow to match physical
        mip = np.max(patch, axis=2-i).T
        extent = [lower[axes[0]], upper[axes[0]], upper[axes[1]], lower[axes[1]]]
        ax.imshow(mip, vmax=vmax, extent=extent, cmap="cet_gray_r")
        # ax.plot(*soma[axes], "ko")
        ax.plot(*nodes[axes], "g.", alpha=1, markersize=3)
        if i==0:
            r = 8
            scale = 50
            x0 = (r*extent[0]+extent[1])/(r+1)
            y0 = (r*extent[2]+extent[3])/(r+1)
            ax.plot([x0, x0+scale],[y0, y0], "k-", linewidth=2, label=f"{scale} μm")
        if soma_fit:
            ax.plot(*soma[axes], "bo")
            # ax.plot(*soma_offset[axes], "yo")
            ax.plot(*primary[axes], "b-")
        # ax.set_title(" vs ".join("XYZ"[i] for i in axes[::-1]), fontsize=16)
        # ax.set_title("".join("XYZ"[i] for i in axes), fontsize=16)
        ax.set_aspect("equal")

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    # plt.show()

def intersection_points(m, cutoff = 100):
    root = m.get_root()
    hits = OrderedDict()
    record = {}
    # hits[root["id"]] = root
    def visit(node):
        # c = m.get_compartment_for_node(node)
        # if c is None:
        #     return
        # if (cutoff > m.euclidean_distance(root, c[0]) and cutoff < m.euclidean_distance(root, c[1])):
            # hits[node["id"]] = c[0]
        record["stop"] = False
        if node["type"]==1:
            return
        if node["type"]==3:
            is_second_branch = (len(m.get_children(node)) > 1) and (record["branch_root"] != 1)
            if cutoff < m.euclidean_distance(root, node) or is_second_branch:
                hits[node["id"]] = node
                record["stop"] = True
        else:
            # not dendrite or soma
            record["stop"] = True
        return
    def cb(node_id):
        if record["stop"]:
            # stop traversal
            return []
        else:
            nested_ids = m.child_ids([node_id])
            children = [nid for nids in nested_ids for nid in nids]
            if len(children) > 1:
                record["branch_root"] = node_id
            return children

    m.breadth_first_traversal(visit, cb)
    return [np.array([hit[x] for x in "xyz"]) for hit in hits.values()]

def get_bipolarity(soma, stems, primary_axis, cos2_cutoff=0.5):
    val = 0.0
    aligned_avg = 0.0
    avg0 = np.array([0,0,0])
    avg1 = np.array([0,0,0])
    avg2 = np.array([0,0,0])
    # cos2_vals = []
    n = len(stems)
    primary_axis = np.array(primary_axis)
    primary_axis = primary_axis / np.linalg.norm(primary_axis)
    vecs = []
    for stem in stems:
        vec = np.array(stem) - np.array(soma)
        vec = vec / np.linalg.norm(vec)
        vecs.append(vec.astype(float))

        avg0 = avg0 + (vec * np.sign(vec[0])/n)
        avg1 = avg1 + (vec * np.sign(vec[1])/n)
        avg2 = avg2 + (vec * np.sign(vec[2])/n)
        aligned = np.abs(np.dot(vec, primary_axis))
        aligned_avg += aligned/n
        # if aligned**2 > cos2_cutoff:
        #     val += 1.0/n
    vectors = np.array(vecs)
    S = vectors.T @ vectors / len(vectors)
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    avg, norm_avg = eigenvectors[:, 2], np.sqrt(eigenvalues[2])
    # avg_candidates = [avg0, avg1, avg2]
    # avg = max(avg_candidates, key=lambda x: np.linalg.norm(x))
    # norm_avg = np.linalg.norm(avg)
    # abs_cos = np.abs(np.dot(avg, primary_axis))/norm_avg
    abs_cos = np.abs(np.dot(avg, primary_axis))
    return val, aligned_avg, norm_avg, abs_cos


def get_dend_bipolarity(soma, stems):
    vecs = []
    for stem in stems:
        vec = np.array(stem) - np.array(soma)
        vec = vec / np.linalg.norm(vec)
        vecs.append(vec.astype(float))

    vectors = np.array(vecs)
    S = vectors.T @ vectors / len(vectors)
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    avg, norm_avg = eigenvectors[:, 2], np.sqrt(eigenvalues[2])
    return norm_avg