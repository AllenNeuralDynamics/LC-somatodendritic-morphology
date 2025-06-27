import pandas as pd
import json
import numpy as np

from joblib import Memory

from neuron_morphology.constants import AXON, BASAL_DENDRITE, APICAL_DENDRITE, SOMA
from neuron_morphology.morphology import Morphology

import tree_comparison.tree_compare as tc


def load_morphology_and_soma(swc_file):
    # check if json or swc
    if swc_file.endswith('.json'):
        # Load the morphology from a JSON file
        with open(swc_file, 'r') as f:
            data = json.load(f)
        neuron = data["neurons"][0]
        dendrites = pd.DataFrame(neuron["dendrite"]).replace(-1, 0).assign(type=BASAL_DENDRITE)
        soma = neuron["soma"]
        soma["sampleNumber"] = 0
        soma["parentNumber"] = -1
        columns = {
            "sampleNumber": "id",
            "parentNumber": "parent",
        }
        soma = pd.DataFrame([soma]).assign(type=SOMA)
        df = pd.concat([soma, dendrites])
        # mirror images of R hemisphere cells
        df["z"] = np.minimum(df["z"], 11400-df["z"])
        records = df.rename(columns=columns).to_dict(orient="records")
        return Morphology(records, node_id_cb=lambda node: node["id"], parent_id_cb=lambda node: node["parent"]), soma

def load_morphology(swc_file):
    # check if json or swc
    if swc_file.endswith('.json'):
        # Load the morphology from a JSON file
        with open(swc_file, 'r') as f:
            data = json.load(f)
        neuron = data["neurons"][0]
        dendrites = pd.DataFrame(neuron["dendrite"]).replace(-1, 0).assign(type=BASAL_DENDRITE)
        soma = neuron["soma"]
        soma["sampleNumber"] = 0
        soma["parentNumber"] = -1
        columns = {
            "sampleNumber": "id",
            "parentNumber": "parent",
        }
        soma = pd.DataFrame([soma]).assign(type=SOMA)
        df = pd.concat([soma, dendrites])
        # mirror images of R hemisphere cells
        df["z"] = np.minimum(df["z"], 11400-df["z"])
        records = df.rename(columns=columns).to_dict(orient="records")
        return Morphology(records, node_id_cb=lambda node: node["id"], parent_id_cb=lambda node: node["parent"])

    elif swc_file.endswith('.swc'):
        # Load the morphology from a SWC file
        return morphology_from_swc(swc_file)
    else:
        raise ValueError("Unsupported file format. Please provide a SWC or JSON file.")


# from neuron_morphology.swc_io import morphology_from_swc
# # monkey patch this function to use load_morphology
tc.morphology_from_swc = load_morphology


memory = Memory("/scratch/cache", verbose=0)

@memory.cache
def compare(paths_tuple, simFunc="convex"):
    """
    Compare two trees and return the result.
    """
    path1, path2 = paths_tuple  # Unpack the tuple of paths
    try:
        result = tc.compare_two_trees(
            path1, 
            path2, 
            simFunc=simFunc,
            maxDepth=1,
            orientation=0,
            compartments=[3],
            valid_set_dict="",
            partition_length=1/2000,
            angle_threshold=None, 
            segment_threshold=None,
            morphology_from_swc=load_morphology
        )
        distance, norm_distance = result[:2]
        return {
            "file1": path1,
            "file2": path2,
            "distance": distance,
            "norm_distance": norm_distance
        }
    except KeyError as e:
        print(f"Error comparing {path1} and {path2}: {e}")
        return {
            "file1": path1,
            "file2": path2,
            "distance": None,
            "norm_distance": None
        }

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np

def set_size(scale, ax):
    """ w, h: width, height in inches """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    w = x1-x0
    h = y1-y0
    figw = w/scale
    figh = h/scale
    ax.figure.set_size_inches(figw, figh)

def plot_morphology(morphology, ax, coords='xy', **kwargs):
    nodes = morphology.nodes()
    coord_arrays = [[node[dim] for node in nodes] for dim in coords]

    ax.scatter(*coord_arrays, s=0.1, **kwargs)
    ax.set_aspect('equal')

def create_morphology_scatter_plot(embedding, morphology_dict, files, **plot_args):
    # Create a large figure
    fig = plt.figure(figsize=(10, 10))
    main_ax = fig.add_subplot(111)
    
    # Hide axes and keep background
    main_ax.set_xticks([])
    main_ax.set_yticks([])
    
    # Calculate suitable size for the mini plots
    # points_spread = np.max(embedding, axis=0) - np.min(embedding, axis=0)
    # mini_size = min(points_spread) / 20000  # Reduced by a factor of 10 (was /15)
    
    # Normalize coordinates
    x_min, y_min = np.min(embedding, axis=0)
    x_max, y_max = np.max(embedding, axis=0)
    buffer = 0.05  # Buffer space around the edges (5%)
    x_range = x_max - x_min
    y_range = y_max - y_min
    main_ax.set_xlim(x_min - buffer*x_range, x_max + buffer*x_range)
    main_ax.set_ylim(y_min - buffer*y_range, y_max + buffer*y_range)
    
    # Plot each morphology at its embedding location
    for i, file in enumerate(files):
        # Create a color cycle
        colors = plt.cm.tab10(np.linspace(0, 1, len(files)))
                
        # Get the color for this morphology
        color = colors[i]
        # Create a small figure for this morphology
        mini_fig = plt.figure(figsize=(1, 1), dpi=100)
        mini_ax = mini_fig.add_subplot(111)
        
        # Plot the morphology 
        plot_morphology(morphology_dict[file], ax=mini_ax, color=color, **plot_args)
        mini_ax.axis('off')
        mini_fig.tight_layout(pad=0)
        set_size(1000, mini_ax)
        # remove background
        
        mini_ax.set_facecolor((1, 1, 1, 0))
        mini_fig.patch.set_facecolor((1, 1, 1, 0))
        
        # Convert the mini figure to an image
        mini_fig.canvas.draw()
        image_from_plot = np.frombuffer(mini_fig.canvas.buffer_rgba(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(mini_fig.canvas.get_width_height()[::-1] + (4,))
        plt.close(mini_fig)
        
        # Add the mini plot as an annotation at the embedding coordinates
        im = OffsetImage(image_from_plot, zoom=1/3)
        ab = AnnotationBbox(im, (embedding[i, 0], embedding[i, 1]), frameon=False)
        main_ax.add_artist(ab)
    
    # Add title
    # main_ax.set_title('Morphology Embedding with ZY Projections', fontsize=16)
    
    # Display the plot
    plt.tight_layout()
    return fig, main_ax


from mpl_toolkits.axes_grid1 import make_axes_locatable
def two_view_plot(soma_df, embedding, n, s=20, **kwargs):
    # create colorbar axis

    # two subplots, one for xy and one for zy
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    clim=(-1000, 1000)
    ax[1].scatter(soma_df["x"], -soma_df["y"], c=embedding[:, n], vmin=clim[0], vmax=clim[1], s=s, **kwargs)
    ax[0].scatter(soma_df["zz"], -soma_df["y"], c=embedding[:, n], vmin=clim[0], vmax=clim[1], s=s, **kwargs)

    # Create shared colorbar

    # Add colorbar to the right of the second subplot
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)

    plt.colorbar(ax[1].collections[0], cax=cax, label=f'Dendritic Structure Embedding Axis {n}',)
    for a in ax:
        a.set_axis_off()
        a.set_aspect('equal')
    ax[0].set_title('Coronal')
    ax[1].set_title('Sagittal')
