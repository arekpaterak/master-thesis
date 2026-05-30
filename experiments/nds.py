import marimo

__generated_with = "0.23.6"
app = marimo.App(width="columns")

with app.setup(hide_code=True):
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import torch.nn
    import yaml
    from dotenv import load_dotenv

    import wandb
    from core.models.nds.config import TrainConfig

    load_dotenv(".env")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device
    return (device,)


@app.cell(hide_code=True)
def _():
    from wigglystuff import EnvConfig

    is_script_mode = mo.app_meta().mode == "script"

    env_config_widget = mo.ui.anywidget(
        EnvConfig(
            {
                "WANDB_API_KEY": lambda k: wandb.login(key=k, verify=True),
            }
        )
    )

    yaml_data = {}
    if is_script_mode:
        _cli_args = mo.cli_args()

        if not _cli_args:
            from rich.console import Console
            from rich.table import Table

            _usage = Table(title="NDS Training — CLI Usage")
            _usage.add_column("Flag", style="cyan")
            _usage.add_column("Description")
            _usage.add_row("--config <path>", "Path to YAML config file")
            _usage.add_row(
                "--<section>.<field> <value>",
                "Override a nested config field, e.g. --trainer.epochs 5",
            )
            Console().print(_usage)

            _defaults = TrainConfig()
            _fields = Table(title="Config fields and defaults")
            _fields.add_column("Key", style="cyan")
            _fields.add_column("Default", style="yellow")
            for _section, _params in _defaults.model_dump(exclude={"run_name", "logging"}).items():
                for _field, _value in _params.items():
                    _fields.add_row(f"{_section}.{_field}", str(_value))
            Console().print(_fields)

            sys.exit(0)

        _resume_path = str(_cli_args.get("resume"))
        resume_run_dir = Path(_resume_path) if _resume_path else None

        _config_path = str(_cli_args.get("config")) or (
            str(resume_run_dir / "config.yml") if resume_run_dir else None
        )
        if _config_path:
            with open(_config_path) as _f:
                yaml_data = yaml.safe_load(_f) or {}

        for _key, _value in _cli_args.items():
            if _key in ("config", "resume"):
                continue
            _parts = _key.split(".")
            _node = yaml_data
            for _part in _parts[:-1]:
                _node = _node.setdefault(_part, {})
            _node[_parts[-1]] = _value
    else:
        resume_run_dir = None
    return env_config_widget, is_script_mode, resume_run_dir, yaml_data


@app.cell(hide_code=True)
def _(env_config_widget, is_script_mode):
    env_config_widget if not is_script_mode and not wandb.login() else None
    return


@app.cell(column=1, hide_code=True)
def _():
    mo.md(r"""
    ## Config
    """)
    return


@app.cell(hide_code=True)
def _():
    defaults = TrainConfig()
    return (defaults,)


@app.cell(hide_code=True)
def _(defaults):
    problem_name = mo.ui.dropdown(
        ["cvrp", "tsp"], value=defaults.problem.name, label="Problem name"
    )
    problem_size = mo.ui.dropdown(
        [20, 50, 100, 250, 500, 1000], value=defaults.problem.size, label="Problem size"
    )
    mo.vstack([mo.md("### Problem"), problem_name, problem_size])
    return problem_name, problem_size


@app.cell(hide_code=True)
def _(defaults):
    model_hidden_dim = mo.ui.dropdown(
        [64, 128, 256, 512], value=defaults.model.hidden_dim, label="Hidden dim"
    )
    model_num_heads = mo.ui.slider(1, 16, value=defaults.model.num_heads, step=1, label="Num heads")
    model_ff_dim = mo.ui.dropdown(
        [128, 256, 512, 1024], value=defaults.model.ff_dim, label="FF dim"
    )
    model_poly_dim = mo.ui.dropdown(
        [64, 128, 256, 512], value=defaults.model.poly_dim, label="Poly dim"
    )
    model_seed_dim = mo.ui.dropdown(
        [4, 8, 10, 16, 32], value=defaults.model.seed_dim, label="Seed dim"
    )
    model_with_neighbours_encoding = mo.ui.checkbox(
        value=defaults.model.with_neighbours_encoding, label="With neighbours encoding"
    )
    mo.vstack(
        [
            mo.md("### Model"),
            model_hidden_dim,
            model_num_heads,
            model_ff_dim,
            model_poly_dim,
            model_seed_dim,
            model_with_neighbours_encoding,
        ]
    )
    return (
        model_ff_dim,
        model_hidden_dim,
        model_num_heads,
        model_poly_dim,
        model_seed_dim,
        model_with_neighbours_encoding,
    )


@app.cell(hide_code=True)
def _(defaults):
    from core.models.nds.decoders.config import (
        AutoregressiveDecoderConfig as _AutoregressiveDecoderConfig,
    )
    from core.models.nds.decoders.config import (
        ProximityDecoderConfig as _ProximityDecoderConfig,
    )

    _default_decoder = defaults.model.decoder

    decoder_type_widget = mo.ui.dropdown(
        ["ar", "simple", "slot", "proximity"],
        value=_default_decoder.type,
        label="Decoder type",
    )
    decoder_use_hidden_widget = mo.ui.checkbox(
        value=_default_decoder.use_hidden_decoding_state
        if isinstance(_default_decoder, _AutoregressiveDecoderConfig)
        else True,
        label="Use hidden decoding state",
    )
    decoder_sigma_widget = mo.ui.number(
        0.01,
        2.0,
        value=_default_decoder.sigma
        if isinstance(_default_decoder, _ProximityDecoderConfig)
        else 0.1,
        step=0.01,
        label="Sigma",
    )
    return decoder_sigma_widget, decoder_type_widget, decoder_use_hidden_widget


@app.cell(hide_code=True)
def _(decoder_sigma_widget, decoder_type_widget, decoder_use_hidden_widget):
    _items = [mo.md("### Decoder"), decoder_type_widget]
    if decoder_type_widget.value == "ar":
        _items.append(decoder_use_hidden_widget)
    elif decoder_type_widget.value == "proximity":
        _items.append(decoder_sigma_widget)
    mo.vstack(_items)
    return


@app.cell(hide_code=True)
def _(defaults):
    _batch_sizes = [8, 16, 32, 64, 128, 256, 512]
    _rollout_sizes = [64, 128, 256]
    trainer_epochs = mo.ui.number(1, 100, value=defaults.trainer.epochs, step=1, label="Epochs")
    trainer_instances_per_epoch = mo.ui.number(
        100, 1500, value=defaults.trainer.instances_per_epoch, step=100, label="Instances per epoch"
    )
    trainer_batch_size = mo.ui.dropdown(
        _batch_sizes, value=defaults.trainer.batch_size, label="Batch size"
    )
    trainer_rollouts_per_instance = mo.ui.dropdown(
        _rollout_sizes, value=defaults.trainer.rollouts_per_instance, label="Rollouts/instance"
    )
    trainer_num_initial_improvement_steps = mo.ui.slider(
        0, 10, value=defaults.trainer.num_initial_improvement_steps, step=5, label="Initial steps"
    )
    trainer_num_main_improvement_steps = mo.ui.slider(
        10, 200, value=defaults.trainer.num_main_improvement_steps, step=10, label="Main steps"
    )
    trainer_deconstruction_size = mo.ui.slider(
        5, 25, step=5, value=defaults.trainer.deconstruction_size, label="Deconstruction size"
    )
    trainer_checkpoint_every_n_epochs = mo.ui.number(
        1,
        100,
        value=defaults.trainer.checkpoint_every_n_epochs,
        step=1,
        label="Checkpoint every N epochs",
    )
    mo.vstack(
        [
            mo.md("### Trainer"),
            trainer_epochs,
            trainer_instances_per_epoch,
            trainer_batch_size,
            trainer_rollouts_per_instance,
            trainer_num_initial_improvement_steps,
            trainer_num_main_improvement_steps,
            trainer_deconstruction_size,
            trainer_checkpoint_every_n_epochs,
        ]
    )
    return (
        trainer_batch_size,
        trainer_checkpoint_every_n_epochs,
        trainer_deconstruction_size,
        trainer_epochs,
        trainer_instances_per_epoch,
        trainer_num_initial_improvement_steps,
        trainer_num_main_improvement_steps,
        trainer_rollouts_per_instance,
    )


@app.cell(hide_code=True)
def _(defaults):
    trainer_init_weights = mo.ui.text(
        value=defaults.trainer.init_weights or "",
        label="Own checkpoint path (empty = none)",
    )
    _pretrained = defaults.trainer.original_pretrained_weights
    _pretrained_dir = Path("experiments/pretrained/nds")
    _pretrained_options = {"(none)": ""}
    for _pt in sorted(_pretrained_dir.glob("**/*.pt")):
        _pretrained_options[f"{_pt.parent.name}/{_pt.name}"] = str(_pt)
    _pretrained_default_path = _pretrained.path if _pretrained else ""
    _pretrained_default_key = next(
        (k for k, v in _pretrained_options.items() if v == _pretrained_default_path),
        "(none)",
    )
    trainer_original_pretrained_path = mo.ui.dropdown(
        options=_pretrained_options,
        value=_pretrained_default_key,
        label="Original pretrained checkpoint",
    )
    trainer_original_pretrained_load_decoder = mo.ui.checkbox(
        value=_pretrained.load_decoder if _pretrained else False,
        label="Load decoder from original pretrained",
    )
    trainer_freeze_encoder = mo.ui.checkbox(
        value=defaults.trainer.freeze_encoder,
        label="Freeze encoder",
    )
    mo.vstack(
        [
            mo.md("### Weights"),
            trainer_original_pretrained_path,
            trainer_original_pretrained_load_decoder,
            trainer_init_weights,
            trainer_freeze_encoder,
        ]
    )
    return (
        trainer_freeze_encoder,
        trainer_init_weights,
        trainer_original_pretrained_load_decoder,
        trainer_original_pretrained_path,
    )


@app.cell(hide_code=True)
def _(defaults):
    optimizer_learning_rate = mo.ui.slider(
        1e-5, 5e-4, value=defaults.optimizer.learning_rate, step=1e-5, label="Learning rate"
    )
    mo.vstack([mo.md("### Optimizer"), optimizer_learning_rate])
    return (optimizer_learning_rate,)


@app.cell(hide_code=True)
def _(defaults):
    logging_wandb_project = mo.ui.text(
        value=defaults.logging.wandb_project, label="W&B project (empty to skip)"
    )
    mo.vstack([mo.md("### Logging"), logging_wandb_project])
    return (logging_wandb_project,)


@app.cell(hide_code=True)
def _(defaults):
    _batch_sizes = [8, 16, 32, 64, 128, 256, 512]
    validation_num_instances = mo.ui.number(
        32, 5000, value=defaults.validation.num_instances, label="Num instances"
    )
    validation_batch_size = mo.ui.dropdown(
        _batch_sizes, value=defaults.validation.batch_size, label="Batch size"
    )
    validation_num_improvement_steps = mo.ui.number(
        1, 200, value=defaults.validation.num_improvement_steps, step=1, label="Improvement steps"
    )
    validation_every_n_epochs = mo.ui.number(
        1, 100, value=defaults.validation.every_n_epochs, step=1, label="Every N epochs"
    )
    validation_seed = mo.ui.number(0, 99999, value=defaults.validation.seed, step=1, label="Seed")
    validation_plot_enabled = mo.ui.checkbox(
        value=defaults.validation.plot_enabled, label="Plot enabled"
    )
    validation_plot_every_n_improvement_steps = mo.ui.number(
        1,
        20,
        value=defaults.validation.plot_every_n_improvement_steps,
        step=1,
        label="Plot every N steps",
    )
    validation_plot_save_to_disk = mo.ui.checkbox(
        value=defaults.validation.plot_save_to_disk, label="Save to disk"
    )
    validation_plot_show_in_marimo = mo.ui.checkbox(
        value=defaults.validation.plot_show_in_marimo, label="Show in marimo"
    )
    mo.vstack(
        [
            mo.md("### Validation"),
            validation_num_instances,
            validation_batch_size,
            validation_num_improvement_steps,
            validation_every_n_epochs,
            validation_seed,
            validation_plot_enabled,
            validation_plot_every_n_improvement_steps,
            validation_plot_save_to_disk,
            validation_plot_show_in_marimo,
        ]
    )
    return (
        validation_batch_size,
        validation_every_n_epochs,
        validation_num_improvement_steps,
        validation_num_instances,
        validation_plot_enabled,
        validation_plot_every_n_improvement_steps,
        validation_plot_save_to_disk,
        validation_plot_show_in_marimo,
        validation_seed,
    )


@app.cell(hide_code=True)
def _(is_script_mode):
    run_button = mo.ui.run_button(label="Start Training")
    noop_apply_repair_checkbox = mo.ui.checkbox(
        value=False, label="Noop apply+repair (timing benchmark)"
    )
    mo.vstack([run_button, noop_apply_repair_checkbox]) if not is_script_mode else None
    return noop_apply_repair_checkbox, run_button


@app.cell(hide_code=True)
def _(
    decoder_sigma_widget,
    decoder_type_widget,
    decoder_use_hidden_widget,
    is_script_mode,
    logging_wandb_project,
    model_ff_dim,
    model_hidden_dim,
    model_num_heads,
    model_poly_dim,
    model_seed_dim,
    model_with_neighbours_encoding,
    optimizer_learning_rate,
    problem_name,
    problem_size,
    run_button,
    trainer_batch_size,
    trainer_checkpoint_every_n_epochs,
    trainer_deconstruction_size,
    trainer_epochs,
    trainer_freeze_encoder,
    trainer_init_weights,
    trainer_instances_per_epoch,
    trainer_num_initial_improvement_steps,
    trainer_num_main_improvement_steps,
    trainer_original_pretrained_load_decoder,
    trainer_original_pretrained_path,
    trainer_rollouts_per_instance,
    validation_batch_size,
    validation_every_n_epochs,
    validation_num_improvement_steps,
    validation_num_instances,
    validation_plot_enabled,
    validation_plot_every_n_improvement_steps,
    validation_plot_save_to_disk,
    validation_plot_show_in_marimo,
    validation_seed,
    yaml_data,
):
    from core.models.nds.config import (
        LoggingConfig,
        ModelParams,
        OptimizerParams,
        OriginalPretrainedConfig,
        ProblemParams,
        TrainerParams,
        ValidationConfig,
    )
    from core.models.nds.decoders.config import (
        AutoregressiveDecoderConfig as _AutoregressiveDecoderConfig,
    )
    from core.models.nds.decoders.config import (
        ProximityDecoderConfig as _ProximityDecoderConfig,
    )
    from core.models.nds.decoders.config import (
        SimpleOnePassDecoderConfig as _SimpleDecoderConfig,
    )
    from core.models.nds.decoders.config import (
        SlotDecoderConfig as _SlotDecoderConfig,
    )

    if is_script_mode:
        config = TrainConfig.model_validate(yaml_data)
    else:
        mo.stop(
            not run_button.value,
            mo.md("Click 'Start Training' to begin."),
        )
        match decoder_type_widget.value:
            case "ar":
                _decoder_config = _AutoregressiveDecoderConfig(
                    use_hidden_decoding_state=decoder_use_hidden_widget.value
                )
            case "simple":
                _decoder_config = _SimpleDecoderConfig()
            case "slot":
                _decoder_config = _SlotDecoderConfig()
            case "proximity":
                _decoder_config = _ProximityDecoderConfig(sigma=decoder_sigma_widget.value)

        _init_weights = trainer_init_weights.value.strip() or None
        _pretrained_path = trainer_original_pretrained_path.value
        _original_pretrained = (
            OriginalPretrainedConfig(
                path=_pretrained_path,
                load_decoder=trainer_original_pretrained_load_decoder.value,
            )
            if _pretrained_path
            else None
        )

        config = TrainConfig(
            problem=ProblemParams(
                name=problem_name.value,
                size=problem_size.value,
            ),
            model=ModelParams(
                hidden_dim=model_hidden_dim.value,
                num_heads=model_num_heads.value,
                ff_dim=model_ff_dim.value,
                poly_dim=model_poly_dim.value,
                seed_dim=model_seed_dim.value,
                with_neighbours_encoding=model_with_neighbours_encoding.value,
                decoder=_decoder_config,
            ),
            optimizer=OptimizerParams(
                learning_rate=optimizer_learning_rate.value,
            ),
            trainer=TrainerParams(
                epochs=trainer_epochs.value,
                instances_per_epoch=trainer_instances_per_epoch.value,
                batch_size=trainer_batch_size.value,
                rollouts_per_instance=trainer_rollouts_per_instance.value,
                num_initial_improvement_steps=trainer_num_initial_improvement_steps.value,
                num_main_improvement_steps=trainer_num_main_improvement_steps.value,
                deconstruction_size=trainer_deconstruction_size.value,
                checkpoint_every_n_epochs=trainer_checkpoint_every_n_epochs.value,
                init_weights=_init_weights,
                original_pretrained_weights=_original_pretrained,
                freeze_encoder=trainer_freeze_encoder.value,
            ),
            logging=LoggingConfig(
                wandb_project=logging_wandb_project.value,
            ),
            validation=ValidationConfig(
                num_instances=validation_num_instances.value,
                batch_size=validation_batch_size.value,
                num_improvement_steps=validation_num_improvement_steps.value,
                every_n_epochs=validation_every_n_epochs.value,
                seed=validation_seed.value,
                plot_enabled=validation_plot_enabled.value,
                plot_every_n_improvement_steps=validation_plot_every_n_improvement_steps.value,
                plot_save_to_disk=validation_plot_save_to_disk.value,
                plot_show_in_marimo=validation_plot_show_in_marimo.value,
            ),
        )
    return (config,)


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    ## Problem Setup
    """)
    return


@app.cell
def _(config):
    from core.problems.cvrp import CVRPConfig, CVRPSampler
    from core.problems.tsp import TSPConfig, TSPSampler

    match config.problem.name:
        case "cvrp":
            sampler = CVRPSampler(CVRPConfig.for_n(config.problem.size))
            val_sampler = CVRPSampler(
                CVRPConfig.for_n(config.problem.size), seed=config.validation.seed
            )
        case "tsp":
            sampler = TSPSampler(TSPConfig(num_locations=config.problem.size))
            val_sampler = TSPSampler(
                TSPConfig(num_locations=config.problem.size), seed=config.validation.seed
            )
        case _:
            raise ValueError(f"Unsupported problem name: {config.problem.name}")
    return sampler, val_sampler


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Model Setup
    """)
    return


@app.cell
def _(config, device):
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    from core.models.nds.factory import build_model

    model = build_model(config)

    if config.trainer.original_pretrained_weights:
        from core.models.nds.pretrained import (
            load_original_pretrained_decoder_weights,
            load_original_pretrained_encoder_weights,
        )

        _pretrained = config.trainer.original_pretrained_weights
        load_original_pretrained_encoder_weights(model.encoder, _pretrained.path)
        print(f"Loaded pretrained encoder weights from {_pretrained.path}")
        if _pretrained.load_decoder:
            load_original_pretrained_decoder_weights(model.decoder, _pretrained.path)
            print(f"Loaded pretrained decoder weights from {_pretrained.path}")

    elif config.trainer.init_weights:
        from core.models.utils import load_checkpoint

        load_checkpoint(config.trainer.init_weights, model)
        print(f"Initiated model with weights from {config.trainer.init_weights}")

    if config.trainer.freeze_encoder:
        model.encoder.requires_grad_(False)
        print("Frozen encoder weights")

    model = model.to(device)
    model
    return (model,)


@app.cell(column=3, hide_code=True)
def _():
    mo.md(r"""
    ## Training
    """)
    return


@app.cell(hide_code=True)
def _(
    config,
    is_script_mode,
    model,
    noop_apply_repair_checkbox,
    resume_run_dir,
    run_button,
):
    if not is_script_mode:
        mo.stop(not run_button.value)

    from core.lns.operators.repair import NoopRepair
    from core.lns.operators.repair.greedy_reconnect import (
        CVRPGreedyReconnectRepair,
        TSPGreedyReconnectRepair,
    )
    from core.models.nds.functions import (
        only_expanding_apply_rollout,
    )
    from core.problems.cvrp.solve import one_customer_per_route_graph
    from core.problems.tsp.solve import random_permutation_graph

    def find_latest_checkpoint(run_dir: Path):
        checkpoints = sorted((run_dir / "checkpoints").glob("epoch_*.pt"))
        if not checkpoints:
            return None, 0
        latest = checkpoints[-1]
        epoch = int(latest.stem.split("_")[1])
        return latest, epoch

    run_dir = resume_run_dir or (Path(__file__).parent / "runs" / f"nds-{config.run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)

    initial_epoch = 0
    if resume_run_dir:
        _checkpoint_path, _last_epoch = find_latest_checkpoint(resume_run_dir)
        if _checkpoint_path:
            _ckpt = torch.load(_checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(_ckpt["model"])
            initial_epoch = _last_epoch + 1
    else:
        with open(run_dir / "config.yml", "w") as _f:
            yaml.safe_dump(config.model_dump(), _f, sort_keys=False)

    if config.logging.wandb_project:
        _wandb_id_path = run_dir / "wandb_run_id"
        if resume_run_dir and _wandb_id_path.exists():
            wandb.init(
                project=config.logging.wandb_project,
                resume="must",
                id=_wandb_id_path.read_text().strip(),
            )
        else:
            wandb.init(
                project=config.logging.wandb_project,
                name=config.run_name,
                config=config.model_dump(),
            )
            _wandb_id_path.write_text(wandb.run.id)
        wandb.define_metric("val/*", step_metric="epoch")

    match config.problem.name:
        case "cvrp":
            generate_start_solution = one_customer_per_route_graph
            repair_fn = CVRPGreedyReconnectRepair()
        case "tsp":
            generate_start_solution = random_permutation_graph
            repair_fn = TSPGreedyReconnectRepair()

    if noop_apply_repair_checkbox.value:
        apply_rollout_fn = only_expanding_apply_rollout
        repair_fn = NoopRepair()
    else:
        apply_rollout_fn = None
    return (
        apply_rollout_fn,
        find_latest_checkpoint,
        generate_start_solution,
        initial_epoch,
        repair_fn,
        run_dir,
    )


@app.cell(hide_code=True)
def _(
    apply_rollout_fn,
    config,
    find_latest_checkpoint,
    generate_start_solution,
    initial_epoch,
    model,
    repair_fn,
    resume_run_dir,
    run_dir,
    sampler,
    val_sampler,
):
    import torch.optim as optim

    from core.models.nds.evaluation import EpochValidationSummary, plot_lns_progress
    from core.models.nds.module import NDSModule
    from core.models.nds.training import BatchTrainSummary, EpochTrainSummary
    from core.models.trainer import Trainer

    with (
        mo.status.progress_bar(total=config.trainer.epochs, title="Epochs") as epoch_bar,
        mo.status.progress_bar(
            total=config.trainer.epochs * config.trainer.batches_per_epoch, title="Batches"
        ) as step_bar,
        mo.status.progress_bar(
            total=config.trainer.epochs
            * config.trainer.batches_per_epoch
            * config.trainer.num_main_improvement_steps,
            title="Improvement Steps",
        ) as improvement_bar,
    ):

        def on_improvement_step(_step: int, loss: float) -> None:
            improvement_bar.update(subtitle=f"Loss: {loss:.4f}")

        nds_module = NDSModule(
            config=config,
            model=model,
            generate_start_solution=generate_start_solution,
            apply_rollout=apply_rollout_fn,
            repair=repair_fn,
            on_improvement_step=on_improvement_step,
        )

        def on_batch_train_end(summary: BatchTrainSummary):
            step_bar.update(subtitle=f"Loss: {summary.loss:.4f}")
            if config.logging.wandb_project:
                wandb.log({f"train/{k}": v for k, v in summary.model_dump().items()})

        def on_epoch_train_end(summary: EpochTrainSummary):
            epoch_bar.update(subtitle=f"Avg. Loss: {summary.loss:.4f}")
            is_last = summary.epoch == config.trainer.epochs - 1
            is_checkpoint = (summary.epoch + 1) % config.trainer.checkpoint_every_n_epochs == 0
            if is_last or is_checkpoint:
                checkpoints_path = run_dir / "checkpoints"
                checkpoints_path.mkdir(exist_ok=True)
                torch.save(
                    {"model": nds_module.model.state_dict(), "optimizer": optimizer.state_dict()},
                    checkpoints_path / f"epoch_{summary.epoch:04d}.pt",
                )

        def on_epoch_validation_end(val_summary: EpochValidationSummary):
            if config.logging.wandb_project:
                wandb.log(
                    {
                        "val/objective_initial": val_summary.objective_initial,
                        "val/objective_final": val_summary.objective_final,
                        "val/improvement": val_summary.improvement,
                        "val/relative_improvement": val_summary.relative_improvement,
                        "epoch": val_summary.epoch,
                    }
                )
            if config.validation.plot_enabled:
                fig = plot_lns_progress(
                    val_summary.snapshots, val_summary.step_labels, epoch=val_summary.epoch
                )
                if config.validation.plot_save_to_disk:
                    plot_path = run_dir / "plots"
                    plot_path.mkdir(parents=True, exist_ok=True)
                    fig.savefig(
                        plot_path / f"epoch_{val_summary.epoch:04d}.png",
                        dpi=100,
                        bbox_inches="tight",
                    )
                if config.validation.plot_show_in_marimo:
                    pass
                plt.close(fig)

        optimizer = optim.AdamW(
            nds_module.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
        )

        if resume_run_dir:
            _checkpoint_path, _ = find_latest_checkpoint(resume_run_dir)
            if _checkpoint_path:
                _ckpt = torch.load(_checkpoint_path, map_location="cpu", weights_only=True)
                optimizer.load_state_dict(_ckpt["optimizer"])

        try:
            Trainer(
                max_epochs=config.trainer.epochs,
                training_instances=config.trainer.instances_per_epoch,
                batch_size=config.trainer.batch_size,
                on_batch_train_end=on_batch_train_end,
                on_epoch_train_end=[on_epoch_train_end],
                initial_epoch=initial_epoch,
                validation_instances=config.validation.num_instances,
                validation_batch_size=config.validation.batch_size,
                validate_every_n_epochs=config.validation.every_n_epochs,
                on_epoch_validation_end=[on_epoch_validation_end],
            ).fit(
                module=nds_module,
                optimizer=optimizer,
                instance_sampler=sampler,
                validation_instance_sampler=val_sampler,
            )
        except BaseException:
            if config.logging.wandb_project:
                wandb.finish(exit_code=1)
            raise
        else:
            if config.logging.wandb_project:
                wandb.finish()

    mo.md("**Training complete.**")
    return


@app.cell(column=4, hide_code=True)
def _():
    mo.md(r"""
    ## Evaluation
    """)
    return


if __name__ == "__main__":
    app.run()
