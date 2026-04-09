A constructive (repair) model.

**Inspirations:**
- [*Neural Large Neighborhood Search for the Capacitated Vehicle Routing Problem* (Hottung et al., 2019)](https://arxiv.org/abs/1911.09539)

**Requirements:**

1. Applicable as repair to larger problem sizes than the size it was trained on.
2. When applied to a larger problem size, the model should only process the part of the input that is relevant to the repair, not all the nodes in the instance.

**Training:**

It is possible to train the model in different ways.

1. Constructive training: The model is trained to construct a solution from scratch for a small problem size matching the expected destruction size, e.g. 15 or 20. Then, it is applied as a repair to larger problem sizes, e.g. 100 or 200. The reward is the standard e.g. negative tour length for TSP.
2. Repair training: The instances of the target sizes are randomly destroyed, and the model is trained to repair them. The reward is the improvement in objective compared to the destroyed solution. This approach may be more effective for training the model to perform repairs, as it directly optimises for the repair task. NOTE: Not applicable to the TSP, as there is no feasible solution after the destruction. In the case of the CVRP, the removed nodes can be treated as single-node tours, which makes a feasible solution.

**Inputs:**
Only the nodes that are free to be reinserted are present in the input, e.g. not the nodes that are connected to two other in the TSP.

TSP:
- Node x coordinate.
- Node y coordinate.
- State of the solution: 1 if the node is not connected, 2 if the node is connected.

CVRP:
- Node x coordinate.
- Node y coordinate.
- Fulfilled demand of the node's tour. It is 0 (alternatively -1) for the depot.
- State of the solution: 0 (alternatively -1) for the depot; 1 if the node is not connected; 2 if the node is connected and the tour doesn't contain the depot, 3 otherwise.

