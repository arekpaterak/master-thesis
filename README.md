## Structure
The project is organised with the following structure:

```
├── core/                          # Core code: problems, models, training, LNS
│   ├── models/                    # Neural network models
│   │   ├── attention_model/       # Attention Model (Kool et al., 2019)
│   │   ├── nds/                   # Neural Deconstruction Search (Hottung et al., 2025)
│   │   │   ├── models/
│   │   │   │   ├── ar.py          # AutoregressiveNDSModel
│   │   │   │   └── one_pass.py    # OnePassNDSModel
│   │   │   ├── decoders/          # Destroy decoders
│   │   │   │   ├── config.py      # Decoder config classes
│   │   │   │   ├── ar.py          # Autoregressive decoder
│   │   │   │   └── one_pass/      # One-pass decoder variants (gumbel, slot, proximity)
│   │   │   ├── encoder.py         # NDSEncoder (shared by all models)
│   │   │   ├── config.py          # Configuration
│   │   │   └── factory.py         # Model factory for creating NDS models
│   │   ├── repair/                # Repair model
│   │   ├── embeddings/            # Embedding layers and utilities
│   │   ├── layers/                # Custom neural network layers
│   │   ├── module.py              # Base model module
│   │   ├── trainer.py             # Model training utilities
│   │   ├── rollout.py
│   │   ├── entropy.py             # Entropy calculation
│   │   └── utils.py
│   ├── problems/                  # Combinatorial optimization problems
│   │   ├── tsp/                   # Traveling Salesman Problem
│   │   ├── cvrp/                  # Capacitated Vehicle Routing Problem
│   │   ├── instance.py
│   │   ├── solution.py
│   │   ├── sampler.py
│   │   └── features.py
│   ├── states/                    # State representations
│   │   ├── base.py
│   │   ├── constructive.py        # Constructive state operations
│   │   └── destroy.py             # Destroy state operations
│   ├── lns/                       # Large Neighborhood Search
│   │   ├── operators/             # LNS operators
│   │   ├── lns.py                 # Main LNS algorithm
│   │   └── acceptance_criterion.py # Acceptance criteria for LNS
│   ├── constants.py               # Global constants
│   └── __init__.py
├── experiments/                   # Experiment scripts (Marimo notebooks)
│   ├── configs/                   # Experiment base configurations (YAML)
│   ├── runs/
│   ├── results/
│   ├── attention-model.py
│   ├── nds.py
│   └── RUN.md                     # Instructions for running experiments
├── notebooks/                     # Exploratory Jupyter/Marimo notebooks
├── tests/
│   ├── core/                      # Tests for core modules
│   ├── experiments/               # Tests for experiments
│   ├── conftest.py                # Pytest configuration
│   └── NOTES.md
├── pyproject.toml                 # Python project configuration (uv)
├── uv.lock                        # Dependency lock file
└── README.md
```
