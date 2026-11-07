# Training on variable problem sizes

Question: did any combinatorial optimisation paper train a model on variable problem sizes, and how?

Short answer: yes. Several papers train one model on a range of problem sizes. None of those below builds
mixed-size batches for an attention model; they either use one size per batch or a sparse graph
representation.

## Papers

| Paper | Training sizes | How sizes are mixed | Finding |
|---|---|---|---|
| Learning a SAT Solver from Single-Bit Supervision (Selsam et al., 2019) | SAT, n ~ U(10, 40) variables | Mixed sizes in one batch: "each batch containing up to 12,000 nodes (i.e. literals plus clauses)". This is a sparse graph, the same idea as a PyG `Batch`. | Size sampling was used "to accelerate the learning"; they evaluate only on SR(40). |
| Learning Combinatorial Optimization Algorithms over Graphs (Dai, Khalil et al., 2017) | MVC, MaxCut, TSP; ranges such as 50–100 | "we first sample the number of nodes uniformly at random from that range". The paper doesn't say how batches are built. | Models trained on 50–100 nodes were tested on graphs of up to 1,200 nodes. |
| Learning the Travelling Salesperson Problem Requires Rethinking Generalization (Joshi et al., 2022) | Variable TSP20-50, compared against fixed TSP20/50/100/200 | The paper doesn't say how variable-size instances are batched. | Variable sizes "help models retain performance across the range of graph sizes seen during training", but "somewhat counter-intuitively, does not enable better generalization to larger problems". |
| Evaluating Curriculum Learning Strategies in Neural Combinatorial Optimization (Lisicki et al., 2020) | The Attention Model from Attention, Learn to Solve Routing Problems! (Kool et al., 2019) on TSP 4–100 | The uniform baseline "selected a task randomly at each epoch", so there is one size at a time. Also compares a classic curriculum (sizes increasing monotonically) and an "adaptive staircase". | "balancing the problem sizes naïvely does not lead to an overall improvement in performance"; there is catastrophic forgetting. The adaptive staircase gave the best trade-off. |
| Towards Omni-generalizable Neural Methods for Vehicle Routing Problems (Zhou et al., 2023) | POMO on sizes [50, 200], uniform and Gaussian-mixture distributions (341 tasks) | A task is defined as instances "with the same size n". They use one task per meta-iteration, batch size 64 (32 above size 150), so each batch has one size. | Their plain baseline is POMO retrained on instances "randomly sampled from our training task set", a published comparison between plain multi-size training and meta-learning. |
| Enhancing the Cross-Size Generalization for Solving Vehicle Routing Problems via Continual Learning (Li et al., 2025, preprint) | POMO/ELG on sizes 60, 70, …, 150 | Sizes go in ascending order, 200 epochs each, with experience replay of earlier sizes. Batch size 64 (32 above 100). | Their baselines are POMO and ELG retrained per size, plus AMDKD-POMO and Omni-POMO. |

## What this means for NDS

- **NDS itself is trained per size.** Neural Deconstruction Search for Vehicle Routing Problems
  (Hottung et al., 2025): "For each problem and problem size, we perform a separate training run". They keep
  "15 customers … selected per deconstruction step across all problem sizes", and the 2000-customer model is
  warm-started from the 1000-customer checkpoint. A multi-size NDS model would therefore be a new
  contribution, and per-size NDS models are the natural baseline.
- **The usual approach is one size per batch** (per batch, per epoch, or as a curriculum). None of these works
  needed mixed-size batches. NeuroSAT mixes sizes within a batch only because its model is a sparse
  message-passing GNN.
- **Expect a trade-off.** Joshi et al. and Lisicki et al. both report that naive uniform size sampling keeps
  performance within the training range but doesn't by itself improve results overall or for larger sizes.
  Curriculum and replay schemes were introduced to address forgetting.
- **Normalisation matters.** Joshi et al. found that BatchNorm with learnt statistics "generalizes worse than
  not doing any normalization" across sizes, while batch statistics or LayerNorm cope better. Zhou et al.
  report something similar for meta-training: BatchNorm without running statistics and instance
  normalisation worked best. NDS uses instance norm by default, which fits.

## Multi-size training without mixed batches

The simple route is one size per batch, a different size each step:

- Every sampler already has `for_size` (`core/problems/sampler.py`), so the training loop could draw N from a
  set or range before each batch.
- Model code, solutions and numba repair stay exactly as they are.
- `torch.compile` recompiles for each new shape. Keep the set of sizes small and discrete, or try
  `dynamic=True`.
- Scale the batch size per size so the token count (B·N) stays roughly constant.
- `deconstruction_size` is a fixed count (15 in the configs). If it stays fixed across sizes, a removal of 15
  nodes means something different at N=50 than at N=500. Making it proportional to N is a design choice in its
  own right.
