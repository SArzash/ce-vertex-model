# Convergent Extension Example Files

This repository contains compact example C++ drivers and helper routines illustrating the main simulation protocols used in our convergent-extension study.

These files are intended as **minimal methodological examples**, not as a complete standalone software package.

## Contents

- `external_shear.cpp`  
  Example driver for **passive externally imposed pure shear** in a vertex-model tissue.

- `local_rule.cpp`  
  Example driver for **local edge-tension remodeling** based on geometric information at the edge level.

- `gd.cpp`  
  Example driver for **global gradient-descent-based tuning** of edge tensions toward a target pure-shear deformation.

- `auxiliary_functions.cpp`  
  Simplified helper/model routines used across the example drivers, including initialization of cell preferences and edge tensions, and measurement of edge-based observables.

## Purpose

The goal of this repository is to provide a small set of readable example files that show the main algorithmic structure of the methods used in the paper. In particular, these examples illustrate how we:

1. build initial tissue configurations,
2. tune initial edge-tension and cell-shape-related distributions to match experimental-like initial conditions,
3. evolve tissues under different deformation or tension-remodeling protocols, and
4. measure representative observables during the simulation.

## Dependency on `cellGPU`

These examples rely on the `cellGPU` codebase developed by Daniel Sussman, together with additional local model extensions used in this project.

- Upstream `cellGPU` repository:  
  https://github.com/sussmanLab/cellGPU

These files are **not expected to compile as-is** against a clean upstream `cellGPU` installation, because they assume project-specific extensions to the vertex-model codebase. They should therefore be viewed as example implementations that can be incorporated into an existing `cellGPU`-based workflow.

## What is project-specific here?

In addition to standard `cellGPU` functionality, these examples assume access to local extensions such as:

- active edge-tension fields,
- edge-based observables (length, angle, passive tension, susceptibility),
- local tension-remodeling routines,
- T1-reset conventions used in this project,
- box-strain degrees of freedom for quasistatic pure-shear minimization.

## Notes

- The code is intentionally simplified and shortened to highlight the main algorithmic ideas.
- Parameter values and output paths are passed through command-line arguments in each driver.
- These files are provided for transparency and methodology sharing in connection with the paper.

## Suggested citation/use

If you use or adapt these example files, please also cite the associated paper.

## Contact

For questions about these example files, please contact the repository author.