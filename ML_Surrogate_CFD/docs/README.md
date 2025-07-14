# Documentation for Project4_ML_Surrogate_CFD
# Project 4: ML-Based Surrogate Modeling for CFD

## Objective

Train a fast and lightweight neural network model to predict aerodynamic coefficients (Cl, Cd) from airfoil geometry and flow parameters.

## Dataset Features

- Camber
- Thickness
- Angle of Attack (AoA)
- Mach number
- Reynolds number

## Targets

- Lift Coefficient (Cl)
- Drag Coefficient (Cd)

## How to Run

```bash
python scripts/train_model.py
python postprocessing/plot_predictions.py
