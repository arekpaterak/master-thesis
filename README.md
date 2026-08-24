# Learning Structurally Informed Destroy Operators for Large Neighbourhood Search

## Structure
The code is organised with the following structure:

```
├── core/                          # Core code: problems, models, LNS, evaluation
│   ├── models/                    # Neural network models
│   │   ├── attention_model/       # Attention Model (Kool et al., 2019)
│   │   ├── nds/                   # Neural Deconstruction Search (Hottung et al., 2025)  and my extensions
│   │   │   ├── models/            # Model variants (joint, split)
│   │   │   ├── encoders/          # Instance and solution encoders
│   │   │   ├── decoders/          # Destroy decoders (autoregressive, one-pass)
│   │   │   ├── training/          # Training loop, losses and schedules
│   │   │   └── pretrained/        # Loading of pretrained reference models
│   │   ├── repair/                # Repair model
│   │   ├── embeddings/            # Embedding layers and utilities
│   │   ├── layers/                # Custom neural network layers
│   │   └── trainer.py             # Model training utilities
│   ├── problems/                  # Combinatorial optimization problems
│   │   ├── tsp/                   # Traveling Salesman Problem
│   │   ├── cvrp/                  # Capacitated Vehicle Routing Problem
│   │   ├── */                     # more problems
│   │   ├── construction/          # Construction heuristics
│   │   ├── instance.py            # Problem instances
│   │   ├── solution.py            # Problem solutions
│   │   └── registry.py            # Problem registry
│   ├── states/                    # State representations (constructive, destroy)
│   ├── lns/                       # Large Neighborhood Search
│   │   ├── operators/             # Destroy and repair operators
│   │   ├── search.py              # Main LNS algorithm
│   │   └── acceptance_criterion.py # Acceptance criteria for LNS
│   ├── methods/                   # Solving methods compared in experiments
│   ├── eval/                      # Evaluation settings and configuration
│   └── results/                   # Result collection and aggregation
├── experiments/                   # Experiment scripts, configs, and results
│   ├── configs/
│   ├── runs/
│   ├── results/
│   ├── train.py
│   ├── eval.py
│   └── RUN.md                     # Instructions for running experiments
├── figures/                       # Scripts generating the thesis figures
├── notebooks/                     # Exploratory Jupyter/Marimo notebooks
├── tests/
│   ├── core/                      # Tests for core modules
│   ├── experiments/               # Tests for experiments
│   └── conftest.py
├── data/
│   ├── synthetic/                 # Generated instance sets
│   ├── cvrplib/                   # CVRPLib benchmark instances
│   └── tools/                     # Data generation scripts
├── pyproject.toml                 # Python project configuration
├── uv.lock                        # Dependency lock file
└── README.md
```

## Resources

Data:
- [CVRPLib CVRP Instance Sets](https://galgos.inf.puc-rio.br/cvrplib/en/instances/1)

NCO Methods:
- [Attention, Learn to Solve Routing Problems!](https://github.com/wouterkool/attention-learn-to-route)
- [NDS](https://github.com/ahottung/NDS)

Solvers:
- [PyVRP](https://github.com/PyVRP/PyVRP)

Tools:
- [VRPLIB](https://github.com/PyVRP/VRPLIB)

Miscellaneous:
- https://github.com/Thinklab-SJTU/awesome-ml4co
