# Physics-Informed Neural Network (PINN) for a PDE Problem

This project implements a Physics-Informed Neural Network to solve a partial differential equation (PDE) over a bounded spatial-temporal domain. The training approach combines a neural network with the governing physics, using the PDE residual as part of the loss function instead of relying only on labeled data.

## Project Goal

The model learns an approximation to the analytical solution of the equation:

$$
\frac{\partial u}{\partial t} - \frac{\partial^2 u}{\partial x^2} + e^{-t}\left(\sin(\pi x) - \pi^2 \sin(\pi x)\right)=0
$$

with the exact solution:

$$
u(x, t) = e^{-t}\sin(\pi x)
$$

The code trains a feedforward neural network to satisfy both:
- the known boundary conditions, and
- the PDE residual at many collocation points.

## Included Files

- [pinn_hamiltonian.ipynb](pinn_hamiltonian.ipynb): interactive notebook version of the full experiment.

## Model Structure

The network is a fully connected neural network built with PyTorch:

- Input dimension: 2 (x, t)
- Hidden layers: 64, 64, 64, 64
- Output dimension: 1
- Activation: Tanh
- Optimizer: Adam followed by L-BFGS refinement

The architecture is defined in the `FCN` class and uses automatic differentiation to compute PDE derivatives via `torch.autograd`.

## Loss Function

The total loss combines:

1. Boundary condition loss
2. PDE residual loss

The model is trained to minimize:

$$
\mathcal{L} = \lambda_{BC} \cdot \mathcal{L}_{BC} + \mathcal{L}_{PDE}
$$

where the boundary loss ensures agreement with the known solution on the domain edges and the PDE loss enforces the physics at sampled collocation points.

## Training Workflow

The workflow is:

1. Construct a mesh over space and time.
2. Create collocation points and boundary points.
3. Sample training points from the domain.
4. Initialize the PINN.
5. Run Adam optimization for initial convergence.
6. Run L-BFGS for higher-precision optimization.
7. Evaluate MAE, RMSE, and relative L2 error against the analytical solution.
8. Plot predictions and ground truth.

## Dependencies

This project uses:

- Python 3.9+
- PyTorch
- NumPy
- Matplotlib

## How to Run

### Option 1: Notebook
Open [pinn_hamiltonian.ipynb](pinn_hamiltonian.ipynb) in Jupyter or VS Code and run all cells in order.

### Option 2: Script
Run:

```bash
python pinn_demo.py
```

from the workspace root.

## Expected Output

The script prints:

- device information (CPU or MPS)
- training loss progression
- final error metrics such as:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - Relative L2 error

It also creates a comparison plot between the neural network prediction and the analytical solution.

## Notes

- The project uses a PDE-informed residual to reduce the need for large labeled datasets.
- The choice of Tanh activation and residual-based training works well for smooth solutions.
- This is a compact demonstration of PINNs in scientific machine learning.

## Use Case

This code is useful for learning how PINNs are implemented in practice for solving forward problems with known governing equations and boundary conditions.
