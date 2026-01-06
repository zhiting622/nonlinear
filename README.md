# Nonlinear2026 Project

Project for window size estimation using linear programming and stochastic gradient descent.

## Project Structure

```
nonlinear2026/
├── data/              # Data files (.npy, .pkl.zip)
├── src/               # Source code modules
│   ├── __init__.py
│   ├── data_loader.py
│   ├── error_calculator.py
│   ├── error_matrix_generator.py
│   ├── H_constructor.py
│   ├── inflection_finder.py
│   ├── LP_solver.py
│   ├── piece_utils.py
│   ├── SGD_solver.py
│   └── y_generator.py
├── scripts/           # Main scripts and entry points
│   ├── main_func.py
│   ├── main_func_test.py
│   └── convert_npy.py
├── results/           # Output files (plots, etc.)
│   └── plots_comparison/
├── tests/             # Test files (to be added)
└── README.md
```

## Usage

### Run main pipeline
```bash
python3 scripts/main_func.py
```

### Run comparison tests
```bash
python3 scripts/main_func_test.py
```

### Convert data files
```bash
python3 scripts/convert_npy.py --in-dir data --out-dir data --fs 700
```

## Requirements

- Python 3.x
- numpy
- scipy
- matplotlib

