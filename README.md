# LC somatodendritic morphology analysis

Github: https://github.com/AllenNeuralDynamics/LC-somatodendritic-morphology.git

Release capsule: https://codeocean.allenneuraldynamics.org/capsule/91b372af-1972-4b83-ab5e-1622d1eed31c/

This capsule analyses properties of the soma and dendrites of LC neurons, using both single-neuron reconstructions and images of the cells.

The run script (`code/run`) first runs `plot_3d_views` for morphology renderings in Fig 2, then `run_soma_quantification` to quantify soma shapes and cache results to `\scratch`. Finally `soma_asymmetry.ipynb` generates additional supplementary plots and additional context.

Input data:
- *exaSPIM-fused-images_LC-manuscript_2026-07-16*: whole-brain exaSPIM images for 4 brains with reconstructed neurons in LC
- *exaSPIM-reconstructions-snapshot_LC-manuscript_2026-04-15_17-09-39*: snapshots of all reconstructed neurons from each imaged brain, in SWC format
- *LC_percentile_meshes_2026-07-10_21-13-43*: Volumetric approximation of the bounds of LC, for visualization only