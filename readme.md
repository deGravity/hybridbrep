# HybridBrep

Code for [**Self-Supervised Representation Learning for CAD**](https://arxiv.org)

## Requirements

Depends on AutoMate, CMake Parasolod, and OpenCascade. To setup Parasolid dependency, set the `$PARASOLID_BASE` environmental variable as described [here](https://github.com/degravity/parasolid_frustrum.git).

## Datasets

We evaluate against Fusion 360 Segmentation, MFCAD, and FabWave datasets for segmentation and classification tasks.

Run the `download.py` script to obtain these datasets (they should not be unzipped). FabWave requires contacting the authors to ask for a download link.

## Experimental Data

The experimental data for all of our experiments is available [here](...). Plots from the paper can be reproduced with the `generate_figures.py` script.

## Citing

If you use our work, please cite us as:
```
TBD (In Submission)
```