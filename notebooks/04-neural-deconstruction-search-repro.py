import marimo

__generated_with = "0.22.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Neural Deconstruction Search for Vehicle Routing Problems: Paper Reproduction

    **Paper:** *Neural Deconstruction Search for Vehicle Routing Problems* (Hottung et al., 2025)

    **Scope:**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _():
    import torch
    from torchinfo import summary

    from core.models.nds.models.ar import AutoregressiveNDSModel
    from core.models.nds.seed_sampler import sample_seed
    from core.problems.cvrp import CVRPConfig, CVRPSampler
    from core.problems.cvrp.solve import one_customer_per_route

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {DEVICE}")
    return (
        CVRPConfig,
        CVRPSampler,
        DEVICE,
        AutoregressiveNDSModel,
        one_customer_per_route,
        sample_seed,
        summary,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data
    """)
    return


@app.cell
def _(CVRPConfig, CVRPSampler):
    _example_sampler = CVRPSampler(CVRPConfig.for_n(20))
    example_batch = _example_sampler(4)
    print(example_batch)
    return (example_batch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Architecture
    """)
    return


@app.cell
def _():
    HIDDEN_DIM = 128
    NUM_HEADS = 8
    FF_DIM = 512
    SEED_DIM = 10
    POLY_DIM = 256
    _M = 15
    return FF_DIM, HIDDEN_DIM, POLY_DIM, NUM_HEADS, SEED_DIM


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Encoder
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Initial Node Embedding**
    """)
    return


@app.cell
def _(HIDDEN_DIM, example_batch):
    from core.models.embeddings.cvrp import CVRPInitialEmbedding

    depot_location = example_batch.depot_location
    customer_features = example_batch.customer_features

    print(f"Depot location shape: {depot_location.shape}")
    print(f"Customer features shape: {customer_features.shape}")

    initial_embedding = CVRPInitialEmbedding(hidden_dim=HIDDEN_DIM)
    initial_node_embeddings = initial_embedding(
        depot_features=depot_location, customer_features=customer_features
    )
    print(f"Initial node embeddings shape: {initial_node_embeddings.shape}")
    return (initial_node_embeddings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Attention Layer**

    Same as in "Attention, Learn to Solve Routing Problems!" (Kool et al., 2019).
    """)
    return


@app.cell
def _(FF_DIM, HIDDEN_DIM, NUM_HEADS, initial_node_embeddings):
    from core.models.nds.encoder import AttentionLayer

    att_layer = AttentionLayer(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
    )
    refined_embeddings = att_layer(initial_node_embeddings)

    print(f"Refined node embeddings shape: {refined_embeddings.shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Message Passing Layer**
    """)
    return


@app.cell
def _(FF_DIM, HIDDEN_DIM):
    from core.models.nds.encoder import NeighboursEncodingLayer

    NeighboursEncodingLayer(
        hidden_dim=HIDDEN_DIM,
        ff_dim=FF_DIM,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Tour Encoding Layer**
    """)
    return


@app.cell
def _(FF_DIM, HIDDEN_DIM):
    from core.models.nds.encoder import RouteEncodingLayer

    RouteEncodingLayer(
        hidden_dim=HIDDEN_DIM,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **All sub-layers combined into the encoder**
    """)
    return


@app.cell
def _():
    from core.models.nds.encoder import NDSEncoder

    return (NDSEncoder,)


@app.cell
def _(
    CVRPConfig,
    CVRPSampler,
    DEVICE,
    NDSEncoder,
    one_customer_per_route,
    summary,
):
    encoder = NDSEncoder(problem_name="cvrp").to(DEVICE)
    _example_sampler = CVRPSampler(CVRPConfig(num_customers=5, vehicle_capacity_raw=9), seed=42)
    batch = _example_sampler(2).to(DEVICE)
    depot = batch.depot_location
    customers = batch.customer_features
    solution = one_customer_per_route(batch)
    neighbours_index, tour_index = (
        solution.neighbours_per_customer(),
        solution.route_per_customer(),
    )
    _ = encoder(
        depot_features=depot,
        customer_features=customers,
        neighbours_index=neighbours_index,
        tour_index=tour_index,
    )
    # routes = [[[1,3,2], [4,5]], [[2,1,4], [3,5]]]
    # neighbours_index, tour_index = build_encoder_input_indexes_from_routes(routes, num_customers=batch.num_customers)
    print(
        summary(
            encoder,
            input_data={
                "depot_features": depot,
                "customer_features": customers,
                "neighbours_index": neighbours_index,
                "tour_index": tour_index,
            },
            col_names=["input_size", "output_size", "num_params"],
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Decoder
    """)
    return


@app.cell
def _(DEVICE, HIDDEN_DIM, POLY_DIM, NUM_HEADS, SEED_DIM, sample_seed, torch):
    from core.models.nds.decoders.ar import AutoregressiveNDSDecoder, DecoderState

    B, rollout_size, N = (2, 10, 4)
    seed_dim = 10
    M = 4
    decoder = AutoregressiveNDSDecoder(
        hidden_dim=HIDDEN_DIM, seed_dim=SEED_DIM, poly_dim=POLY_DIM, num_heads=NUM_HEADS
    ).to(DEVICE)
    h_nodes = torch.randn(B, N + 1, HIDDEN_DIM).to(DEVICE)
    z = sample_seed(batch_size=B, rollout_size=rollout_size, seed_dim=seed_dim, device=DEVICE)
    print(f"z: {z}")
    state = decoder.build_initial_state(batch_size=B, rollout_size=rollout_size, num_nodes=N + 1)
    cache = decoder.build_cache(h_nodes=h_nodes, seed=z)
    selected = []
    for m in range(M):
        log_p, gru_hidden_state = decoder(cache=cache, state=state)
        idx = torch.distributions.Categorical(logits=log_p).sample()
        selected.append(idx)
        new_mask = state.selected_mask.clone()
        new_mask.scatter_(2, idx.unsqueeze(-1), True)
        new_h_last_node = cache.h_nodes_expanded.gather(
            dim=2, index=idx.unsqueeze(-1).unsqueeze(-1).expand(B, rollout_size, 1, HIDDEN_DIM)
        ).squeeze(2)
        state = DecoderState(
            h_last_node=new_h_last_node, gru_hidden=gru_hidden_state, selected_mask=new_mask
        )
    selected = torch.stack(selected, dim=-1)
    print(f"selected.shape: {selected.shape}")
    print(f"selected:\n{selected}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Model
    """)
    return


@app.cell
def _(DEVICE, NDSModel):
    model = NDSModel(problem_name="cvrp").to(DEVICE)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training
    """)
    return


@app.cell
def _(CVRPConfig, CVRPSampler, DEVICE, model, one_customer_per_route):
    import torch.optim as optim

    from core.lns.operators.repair import NoopRepair
    from core.models.nds.config import TrainConfig, TrainerParams
    from core.models.nds.functions import only_expanding_apply_rollout
    from core.models.nds.module import NDSModule
    from core.models.trainer import Trainer

    cfg = CVRPConfig.for_n(20)
    sampler = CVRPSampler(cfg, seed=123)

    nds_module = NDSModule(
        config=TrainConfig(
            trainer=TrainerParams(
                num_main_improvement_steps=2,
                rollouts_per_instance=4,
                num_initial_improvement_steps=0,
                deconstruction_size=5,
            )
        ),
        model=model,
        generate_start_solution=one_customer_per_route,
        apply_rollout=only_expanding_apply_rollout,
        repair=NoopRepair(),
    )
    optimizer = optim.Adam(nds_module.parameters(), lr=1e-4)

    Trainer(
        max_epochs=1,
        training_batches_per_epoch=1,
        batch_size=32,
    ).fit(
        module=nds_module,
        optimizer=optimizer,
        instance_sampler=lambda batch_size: sampler(batch_size=batch_size).to(DEVICE),
    )
    return


if __name__ == "__main__":
    app.run()
