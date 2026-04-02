
import numpy as np
import utils as u

# %%
morphos, soma_df = u.load_all(axon_radius=0)

# %%
# df = soma_df.query("x>10000 and y<5500").set_index("file")
df = soma_df.query("x>9500 and y<6000 and z>2000").set_index("file")
df["z"] = df["zz"]

# %%
import matplotlib.pyplot as plt
import matplotlib.cm as cm

rand = np.random.default_rng(seed=42)
order = rand.permutation(df.index)

cmap = plt.get_cmap('viridis')
sm = cm.ScalarMappable(cmap=cmap)
data = df.loc[order,"y"]
sm.set_array(data)
clim = np.percentile(data, [5,95])
sm.set_clim(clim)
colors = sm.to_rgba(data)


print(f"{clim=}")



# %%
import vedo
# vedo.settings.default_backend= '2d'
vedo.settings.default_backend= 'k3d'
from vedo.file_io import load_obj


# 
# Load mesh
lc_mesh = load_obj('/root/capsule/data/LC_percentile_meshes/new_core_mesh.obj')[0]
# lc_mesh = load_obj('/root/capsule/data/lc_meshes/20250418_transformed_remesh_10.obj')

# %%
# Sagittal view (A-P vs D-V)
def plot_projection(ids, axes, ax, node_types=None, mesh=True, somas=True):
    if mesh:
        # proj_ax = next("xyz"[i] for i in range(3) if i not in axes)
        xx = lc_mesh.vertices[:,[1,2,0]]
        ax.scatter(xx[::10,axes[0]], xx[::10,axes[1]], c='gray', s=1, alpha=0.2)
    ax.set_aspect('equal')

    ax.invert_yaxis()
    # ax.set_xticks([])
    # ax.set_yticks([])
    # for spine in ax.spines.values():
    #     spine.set_visible(False)

    x, y = ["xyz"[i] for i in axes]
    for i, file in enumerate(ids):
        m = morphos[file]
        u.plot_morphology_lines(m, ax, x+y, c=colors[i], linewidth=0.5, node_types=node_types)
        # xx = [n[x] for n in m.nodes()]
        # yy = [n[y] for n in m.nodes()]
        # ax.plot(xx, yy, ',', c=colors[i])
        if somas:
            ax.plot(*df.loc[file, [x,y]], "o", markersize=7, markeredgecolor=colors[i], markerfacecolor='none')


# %%
# dendrites
fig, (ax1, ax2) = plt.subplots(
    1, 2,figsize=(15, 7),
)
node_types=[u.BASAL_DENDRITE]
ids = order[:]
plot_projection(ids, [0,1], ax1, node_types=node_types)
ylim=[6000, 3400]
ax1.set_ylim(ylim)
ax1.set_ylabel("D-V")
ax1.set_xlabel("A-P")
plot_projection(ids, [2,1], ax2, node_types=node_types)
ax2.set_ylim(ylim)
ax2.set_xlabel("L-M")

fig.savefig("/results/fig_s6a_dendrites_dv_colors.pdf")

# %%
morphos, soma_df = u.load_all(axon_radius=1e9)

# %%
# full axons
fig, (ax1, ax2) = plt.subplots(
    1, 2,figsize=(15, 7),
)
node_types=[ u.AXON]
ids = order[:]
plot_projection(ids, [0,1], ax1, node_types=node_types, mesh=False, somas=False)
ylim=[6000, 3400]
# ax1.set_ylim(ylim)
ax1.set_ylabel("D-V")
ax1.set_xlabel("A-P")
plot_projection(ids, [2,1], ax2, node_types=node_types, mesh=False, somas=False)
# ax2.set_ylim(ylim)
ax2.set_xlabel("L-M")

fig.savefig("/results/fig_2d_axons_dv_colors.pdf")

