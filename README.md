# Convergent Extension as a Tuning Process — Example Simulation Files

This repository contains compact example C++ drivers and helper routines illustrating the simulation protocols used in

> S. Arzash, A. J. Liu, M. L. Manning, *Epithelial convergent extension as a tuning process*.

These files are intended as **readable methodological examples**, not as a standalone software package. They document how the simulations were set up and run, so that the protocols in the paper can be reproduced or adapted inside an existing `cellGPU`-based workflow.

## Contents

| File | What it shows |
| --- | --- |
| `external_shear.cpp` | Externally imposed pure shear (the passive baseline). Active tensions are held fixed and the box deformation is prescribed. |
| `local_rule.cpp` | Decentralized edge-tension remodeling. Selects among the local rules compared in the paper: **LO** (Eq. 11), **O** (Eq. 7)|
| `gd.cpp` | Global gradient descent on the edge tensions toward a target pure-shear strain. The gradient is obtained by central finite differences. |
| `auxiliary_functions.cpp` | Simplified reference implementations of the model-level routines the drivers call. |

## The model

Cells are polygons in a periodic box and the degrees of freedom are the vertex positions. The energy is

```
E = sum_i [ K_A (A_i - A_0)^2 + K_P (P_i - P_0)^2 ]  +  sum_<ij> Lambda_ij l_ij
```

Each junction also carries a *passive* tension from the perimeter elasticity of its two adjacent cells, `Lambda^P_ij = 2 K_P [(P_m - P_0) + (P_n - P_0)]`, so the total junctional tension is `T_ij = Lambda_ij + Lambda^P_ij`. The **physical** degrees of freedom are the vertex positions; the **tunable** degrees of freedom are the active tensions `Lambda_ij`. All three protocols alternate

1. one update of the tunable degrees of freedom (or one imposed strain increment), and
2. a full FIRE relaxation of the physical degrees of freedom,

so the tissue is always force balanced when observables are measured. The tension-update rules act on the *total* tension `T_ij`.

In `local_rule.cpp` and `gd.cpp` the box pure-shear strain is itself a relaxed degree of freedom: the tissue chooses its own aspect ratio in response to the tension pattern. In `external_shear.cpp` it is imposed instead. This single setting is the structural difference between the tuning protocols and the passive baseline.

## A note on cost

`gd.cpp` is far more expensive than the other two. Each gradient-descent step perturbs every junction in turn by `±h` and re-relaxes the tissue, i.e. `O(N_junctions)` full minimizations per tuning step. It is practical only for the small systems used here.

## Dependency on `cellGPU`

These examples build on the `cellGPU` code base developed by Daniel Sussman,

- upstream repository: https://github.com/sussmanLab/cellGPU
- D. M. Sussman, *cellGPU: Massively parallel simulations of dynamic vertex models*, Comput. Phys. Commun. **219**, 400 (2017),

together with project-specific extensions to the vertex-model classes. The routines in `auxiliary_functions.cpp` are simplified versions of those extensions, written with the same class scope as in the project code so that the correspondence with the drivers' `avm->...` calls is unambiguous.

For this reason the files are not expected to compile as-is against a clean upstream `cellGPU` installation. They are example implementations meant to be read, and to be incorporated into an existing `cellGPU`-based workflow.

## Citation

If you use or adapt these files, please cite the paper above.

## Contact

Questions about these example files: Sadjad Arzash — sarzash3@gatech.edu.

