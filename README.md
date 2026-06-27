## Structure
The project is organised with the following structure:

```
├── core/                          # Core code: problems, models, training, LNS
│   ├── models/                    # Neural network models
│   │   ├── attention_model/       # Attention Model (Kool et al., 2019)
│   │   ├── nds/                   # Neural Deconstruction Search (Hottung et al., 2025)
│   │   │   ├── models/
│   │   │   │   ├── joint.py       # JointNDSModel (re-encodes each step; any decoder)
│   │   │   │   └── split.py       # SplitNDSModel (split encoder, cached instance)
│   │   │   ├── decoders/          # Destroy decoders
│   │   │   │   ├── config.py      # Decoder config classes
│   │   │   │   ├── factory.py     # build_decoder from a decoder config
│   │   │   │   ├── ar.py          # Autoregressive decoder
│   │   │   │   └── one_pass/      # One-pass decoder variants (simple, slot)
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
├── data/
│   ├── nds/
│   └── cvrlip/
├── pyproject.toml                 # Python project configuration (uv)
├── uv.lock                        # Dependency lock file
└── README.md
```

## Resources

Data:
- [CVRPLib CVRP Instance Sets](https://galgos.inf.puc-rio.br/cvrplib/en/instances/1)

Solvers:
- [PyVRP](https://github.com/PyVRP/PyVRP)

NCO Methods:
- [Attention, Learn to Solve Routing Problems!](https://github.com/wouterkool/attention-learn-to-route)
- [NDS](https://github.com/ahottung/NDS)

Tools:
- [VRPLIB](https://github.com/PyVRP/VRPLIB)

Miscellaneous:
- https://github.com/Thinklab-SJTU/awesome-ml4co
