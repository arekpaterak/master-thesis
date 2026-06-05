import marimo

__generated_with = "0.23.6"
app = marimo.App(width="columns")

with app.setup(hide_code=True):
    import gc
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import torch.nn
    import torch.optim as optim
    import yaml
    from dotenv import load_dotenv
    from rich.console import Console
    from rich.table import Table
    from tqdm import tqdm
    from wigglystuff import EnvConfig

    import wandb
    from core.lns.acceptance_criterion import GreedyAcceptanceConfig
    from core.lns.config import SearchParams
    from core.models.nds.config import (
        AmpConfig,
        CompileConfig,
        LoggingConfig,
        ModelParams,
        OptimizerParams,
        OriginalPretrainedConfig,
        ProblemParams,
        TrainConfig,
        TrainerParams,
        ValidationConfig,
    )
    from core.models.nds.decoders.config import (
        AutoregressiveDecoderConfig,
        ProximityDecoderConfig,
        SimpleOnePassDecoderConfig,
        SlotDecoderConfig,
    )
    from core.models.nds.evaluation import EpochValidationSummary
    from core.models.nds.factory import build_model
    from core.models.nds.module import NDSModule
    from core.models.nds.pretrained import (
        load_original_pretrained_decoder_weights,
        load_original_pretrained_encoder_weights,
    )
    from core.models.nds.training import BatchTrainSummary, EpochTrainSummary
    from core.models.trainer import Trainer
    from core.models.utils import load_checkpoint
    from core.problems.setup import make_sampler
    from core.results.plots import plot_lns_progress
    from core.results.wandb_artifacts import upload_final_checkpoint, upload_validation_plots

    load_dotenv(".env")

    TRAIN_CONFIGS_DIR = Path(__file__).parent / "configs" / "train"

    nds_model_holder: list = []


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Neural Deconstruction Search
    """)
    return


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

        if not _cli_args or "help" in _cli_args or "h" in _cli_args:
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

        _resume_path = _cli_args.get("resume")
        resume_run_dir = Path(str(_resume_path)) if _resume_path is not None else None

        _config_path = str(_cli_args.get("config")) or (
            str(resume_run_dir / "config.yml") if resume_run_dir else None
        )
        if _config_path:
            with open(_config_path) as _f:
                yaml_data = yaml.safe_load(_f) or {}

        for _key, _value in _cli_args.items():
            if _key in ("config", "resume"):
                continue
            _parts = [_part.replace("-", "_") for _part in _key.split(".")]
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
    presets = (
        ["(defaults)"] + sorted(p.name for p in TRAIN_CONFIGS_DIR.glob("*.y*ml"))
        if TRAIN_CONFIGS_DIR.exists()
        else ["(defaults)"]
    )
    return (presets,)


@app.cell(hide_code=True)
def _(is_script_mode, presets):
    if is_script_mode:
        preset_dropdown = None
    else:
        preset_dropdown = mo.ui.dropdown(presets, value=presets[0], label="Training preset")
    preset_dropdown
    return (preset_dropdown,)


@app.cell(hide_code=True)
def _(is_script_mode, preset_dropdown):
    if is_script_mode or preset_dropdown.value == "(defaults)":
        defaults = TrainConfig()
    else:
        with open(TRAIN_CONFIGS_DIR / preset_dropdown.value) as _f:
            defaults = TrainConfig.model_validate(yaml.safe_load(_f) or {})
    return (defaults,)


@app.cell(hide_code=True)
def _(defaults):
    problem_name = mo.ui.dropdown(
        ["cvrp", "tsp"], value=defaults.problem.name, label="Problem name"
    )
    problem_size = mo.ui.dropdown([20, 50, 100, 250, 500, 1000], value=100, label="Problem size")
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
    _default_decoder = defaults.model.decoder

    decoder_type_widget = mo.ui.dropdown(
        ["ar", "simple", "slot", "proximity"],
        value=_default_decoder.type,
        label="Decoder type",
    )
    decoder_use_hidden_widget = mo.ui.checkbox(
        value=_default_decoder.use_hidden_decoding_state
        if isinstance(_default_decoder, AutoregressiveDecoderConfig)
        else True,
        label="Use hidden decoding state",
    )
    decoder_sigma_widget = mo.ui.number(
        0.01,
        2.0,
        value=_default_decoder.sigma
        if isinstance(_default_decoder, ProximityDecoderConfig)
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
    model_init_weights = mo.ui.text(
        value=defaults.model.init_weights or "",
        label="Own checkpoint path (empty = none)",
    )
    _pretrained = defaults.model.original_pretrained_weights
    _pretrained_dir = Path("experiments/pretrained/nds")
    _pretrained_options = {"(none)": ""}
    for _pt in sorted(_pretrained_dir.glob("**/*.pt")):
        _pretrained_options[f"{_pt.parent.name}/{_pt.name}"] = _pt.as_posix()
    _pretrained_default_path = Path(_pretrained.path).as_posix() if _pretrained else ""
    _pretrained_default_key = next(
        (k for k, v in _pretrained_options.items() if v == _pretrained_default_path),
        "(none)",
    )
    model_original_pretrained_path = mo.ui.dropdown(
        options=_pretrained_options,
        value=_pretrained_default_key,
        label="Original pretrained checkpoint",
    )
    model_original_pretrained_load_decoder = mo.ui.checkbox(
        value=_pretrained.load_decoder if _pretrained else False,
        label="Load decoder from original pretrained",
    )
    model_freeze_encoder = mo.ui.checkbox(
        value=defaults.model.freeze_encoder,
        label="Freeze encoder",
    )
    mo.vstack(
        [
            mo.md("### Weights"),
            model_original_pretrained_path,
            model_original_pretrained_load_decoder,
            model_init_weights,
            model_freeze_encoder,
        ]
    )
    return (
        model_freeze_encoder,
        model_init_weights,
        model_original_pretrained_load_decoder,
        model_original_pretrained_path,
    )


@app.cell(hide_code=True)
def _(defaults):
    _batch_sizes = [8, 16, 32, 64, 128, 256, 512]
    trainer_epochs = mo.ui.number(1, 100, value=defaults.trainer.epochs, step=1, label="Epochs")
    trainer_instances_per_epoch = mo.ui.number(
        0, 1500, value=defaults.trainer.instances_per_epoch, step=100, label="Instances per epoch"
    )
    trainer_batch_size = mo.ui.dropdown(
        _batch_sizes, value=defaults.trainer.batch_size, label="Batch size"
    )
    trainer_num_main_improvement_steps = mo.ui.slider(
        10, 200, value=defaults.trainer.num_main_improvement_steps, step=10, label="Main steps"
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
            trainer_num_main_improvement_steps,
            trainer_checkpoint_every_n_epochs,
        ]
    )
    return (
        trainer_batch_size,
        trainer_checkpoint_every_n_epochs,
        trainer_epochs,
        trainer_instances_per_epoch,
        trainer_num_main_improvement_steps,
    )


@app.cell(hide_code=True)
def _(defaults):
    _rollout_sizes = [64, 128, 256]
    repair_type_dropdown = mo.ui.dropdown(
        options=["greedy_reconnect", "greedy_insert", "noop"],
        value=defaults.search.repair_type,
        label="Repair type",
    )
    search_num_improvement_steps = mo.ui.slider(
        0,
        100,
        value=defaults.search.num_improvement_steps,
        step=1,
        label="Initial steps",
    )
    search_num_rollouts = mo.ui.dropdown(
        _rollout_sizes, value=defaults.search.num_rollouts, label="Rollouts/instance"
    )
    search_deconstruction_size = mo.ui.slider(
        5, 25, step=5, value=defaults.search.deconstruction_size, label="Deconstruction size"
    )
    search_n_repairs = mo.ui.number(
        1, 20, value=defaults.search.n_repairs, step=1, label="n_repairs"
    )
    search_improvement_step_type = mo.ui.dropdown(
        ["sequential", "parallel"],
        value=defaults.search.improvement_step_type,
        label="Improvement step type",
    )
    mo.vstack(
        [
            mo.md("### Search"),
            repair_type_dropdown,
            search_improvement_step_type,
            search_num_improvement_steps,
            search_num_rollouts,
            search_deconstruction_size,
            search_n_repairs,
        ]
    )
    return (
        repair_type_dropdown,
        search_deconstruction_size,
        search_improvement_step_type,
        search_n_repairs,
        search_num_improvement_steps,
        search_num_rollouts,
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
    amp_enabled = mo.ui.checkbox(value=defaults.amp.enabled, label="AMP enabled")
    amp_dtype = mo.ui.dropdown(["float16", "bfloat16"], value=defaults.amp.dtype, label="AMP dtype")
    mo.vstack([mo.md("### Mixed precision (AMP)"), amp_enabled, amp_dtype])
    return amp_dtype, amp_enabled


@app.cell(hide_code=True)
def _(defaults):
    compile_enabled = mo.ui.checkbox(value=defaults.compile.enabled, label="Compile enabled")
    compile_mode = mo.ui.dropdown(
        ["default", "reduce-overhead", "max-autotune"],
        value=defaults.compile.mode,
        label="Compile mode",
    )
    mo.vstack([mo.md("### torch.compile"), compile_enabled, compile_mode])
    return compile_enabled, compile_mode


@app.cell(hide_code=True)
def _(defaults):
    logging_wandb_project = mo.ui.text(
        value=defaults.logging.wandb_project, label="W&B project (empty to skip)"
    )
    mo.vstack([mo.md("### Logging"), logging_wandb_project])
    return (logging_wandb_project,)


@app.cell(hide_code=True)
def _(defaults):
    validation_num_instances = mo.ui.number(
        0, 5000, value=defaults.validation.num_instances, label="Num instances"
    )
    validation_batch_size = mo.ui.number(value=defaults.validation.batch_size, label="Batch size")
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
    mo.vstack(
        [
            mo.md("### Validation"),
            validation_num_instances,
            validation_batch_size,
            validation_every_n_epochs,
            validation_seed,
            validation_plot_enabled,
            validation_plot_every_n_improvement_steps,
            validation_plot_save_to_disk,
        ]
    )
    return (
        validation_batch_size,
        validation_every_n_epochs,
        validation_num_instances,
        validation_plot_enabled,
        validation_plot_every_n_improvement_steps,
        validation_plot_save_to_disk,
        validation_seed,
    )


@app.cell(hide_code=True)
def _(defaults):
    _search = defaults.validation.search
    validation_search_num_improvement_steps = mo.ui.number(
        1, 200, value=_search.num_improvement_steps, step=1, label="Improvement steps"
    )
    validation_search_num_rollouts = mo.ui.number(
        value=_search.num_rollouts, label="Rollouts per improvement step"
    )
    validation_search_deconstruction_size = mo.ui.slider(
        5, 25, step=5, value=_search.deconstruction_size, label="Deconstruction size"
    )
    validation_search_n_repairs = mo.ui.number(
        1, 20, value=_search.n_repairs, step=1, label="n_repairs"
    )
    validation_search_improvement_step_type = mo.ui.dropdown(
        ["sequential", "parallel"],
        value=_search.improvement_step_type,
        label="Improvement step type",
    )
    mo.vstack(
        [
            mo.md("### Validation: Search"),
            validation_search_improvement_step_type,
            validation_search_num_improvement_steps,
            validation_search_num_rollouts,
            validation_search_deconstruction_size,
            validation_search_n_repairs,
        ]
    )
    return (
        validation_search_deconstruction_size,
        validation_search_improvement_step_type,
        validation_search_n_repairs,
        validation_search_num_improvement_steps,
        validation_search_num_rollouts,
    )


@app.cell(hide_code=True)
def _(is_script_mode):
    run_button = mo.ui.run_button(label="Start Training")
    run_button if not is_script_mode else None
    return (run_button,)


@app.cell(hide_code=True)
def _(
    amp_dtype,
    amp_enabled,
    compile_enabled,
    compile_mode,
    decoder_sigma_widget,
    decoder_type_widget,
    decoder_use_hidden_widget,
    is_script_mode,
    logging_wandb_project,
    model_ff_dim,
    model_freeze_encoder,
    model_hidden_dim,
    model_init_weights,
    model_num_heads,
    model_original_pretrained_load_decoder,
    model_original_pretrained_path,
    model_poly_dim,
    model_seed_dim,
    model_with_neighbours_encoding,
    optimizer_learning_rate,
    problem_name,
    problem_size,
    repair_type_dropdown,
    run_button,
    search_deconstruction_size,
    search_improvement_step_type,
    search_n_repairs,
    search_num_improvement_steps,
    search_num_rollouts,
    trainer_batch_size,
    trainer_checkpoint_every_n_epochs,
    trainer_epochs,
    trainer_instances_per_epoch,
    trainer_num_main_improvement_steps,
    validation_batch_size,
    validation_every_n_epochs,
    validation_num_instances,
    validation_plot_enabled,
    validation_plot_every_n_improvement_steps,
    validation_plot_save_to_disk,
    validation_search_deconstruction_size,
    validation_search_improvement_step_type,
    validation_search_n_repairs,
    validation_search_num_improvement_steps,
    validation_search_num_rollouts,
    validation_seed,
    yaml_data,
):
    if is_script_mode:
        config = TrainConfig.model_validate(yaml_data)
    else:
        mo.stop(
            not run_button.value,
            mo.md("Click 'Start Training' to begin."),
        )
        match decoder_type_widget.value:
            case "ar":
                _decoder_config = AutoregressiveDecoderConfig(
                    use_hidden_decoding_state=decoder_use_hidden_widget.value
                )
            case "simple":
                _decoder_config = SimpleOnePassDecoderConfig()
            case "slot":
                _decoder_config = SlotDecoderConfig()
            case "proximity":
                _decoder_config = ProximityDecoderConfig(sigma=decoder_sigma_widget.value)

        _init_weights = model_init_weights.value.strip() or None
        _pretrained_path = model_original_pretrained_path.value
        _original_pretrained = (
            OriginalPretrainedConfig(
                path=_pretrained_path,
                load_decoder=model_original_pretrained_load_decoder.value,
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
                init_weights=_init_weights,
                original_pretrained_weights=_original_pretrained,
                freeze_encoder=model_freeze_encoder.value,
            ),
            optimizer=OptimizerParams(
                learning_rate=optimizer_learning_rate.value,
            ),
            amp=AmpConfig(
                enabled=amp_enabled.value,
                dtype=amp_dtype.value,
            ),
            compile=CompileConfig(
                enabled=compile_enabled.value,
                mode=compile_mode.value,
            ),
            trainer=TrainerParams(
                epochs=trainer_epochs.value,
                instances_per_epoch=trainer_instances_per_epoch.value,
                batch_size=trainer_batch_size.value,
                num_main_improvement_steps=trainer_num_main_improvement_steps.value,
                checkpoint_every_n_epochs=trainer_checkpoint_every_n_epochs.value,
            ),
            search=SearchParams(
                repair_type=repair_type_dropdown.value,
                num_improvement_steps=search_num_improvement_steps.value,
                num_rollouts=search_num_rollouts.value,
                deconstruction_size=search_deconstruction_size.value,
                n_repairs=search_n_repairs.value,
                improvement_step_type=search_improvement_step_type.value,
            ),
            logging=LoggingConfig(
                wandb_project=logging_wandb_project.value,
            ),
            validation=ValidationConfig(
                num_instances=validation_num_instances.value,
                batch_size=validation_batch_size.value,
                every_n_epochs=validation_every_n_epochs.value,
                seed=validation_seed.value,
                plot_enabled=validation_plot_enabled.value,
                plot_every_n_improvement_steps=validation_plot_every_n_improvement_steps.value,
                plot_save_to_disk=validation_plot_save_to_disk.value,
                search=SearchParams(
                    num_improvement_steps=validation_search_num_improvement_steps.value,
                    num_rollouts=validation_search_num_rollouts.value,
                    deconstruction_size=validation_search_deconstruction_size.value,
                    n_repairs=validation_search_n_repairs.value,
                    improvement_step_type=validation_search_improvement_step_type.value,
                    acceptance=GreedyAcceptanceConfig(),
                ),
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
    sampler = make_sampler(config.problem.name, config.problem.size)
    val_sampler = make_sampler(
        config.problem.name, config.problem.size, seed=config.validation.seed
    )
    return sampler, val_sampler


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Model Setup
    """)
    return


@app.cell
def _(config, device):
    if nds_model_holder:
        del nds_model_holder[0]
        nds_model_holder.clear()
        gc.collect()
        torch.cuda.empty_cache()

    model = build_model(config)

    if config.model.original_pretrained_weights:
        _pretrained = config.model.original_pretrained_weights
        load_original_pretrained_encoder_weights(model.encoder, _pretrained.path)
        print(f"Loaded pretrained encoder weights from {_pretrained.path}")
        if _pretrained.load_decoder:
            load_original_pretrained_decoder_weights(model.decoder, _pretrained.path)
            print(f"Loaded pretrained decoder weights from {_pretrained.path}")

    elif config.model.init_weights:
        load_checkpoint(config.model.init_weights, model)
        print(f"Initiated model with weights from {config.model.init_weights}")

    if config.model.freeze_encoder:
        model.encoder.requires_grad_(False)
        print("Frozen encoder weights")

    model = model.to(device)
    nds_model_holder.append(model)
    model
    return (model,)


@app.cell(column=3, hide_code=True)
def _():
    mo.md(r"""
    ## Training
    """)
    return


@app.cell(hide_code=True)
def _(config, is_script_mode, model, resume_run_dir, run_button):
    if not is_script_mode:
        mo.stop(not run_button.value)

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
    return find_latest_checkpoint, initial_epoch, run_dir


@app.cell(hide_code=True)
def _(
    config,
    find_latest_checkpoint,
    initial_epoch,
    is_script_mode,
    model,
    resume_run_dir,
    run_dir,
    sampler,
    val_sampler,
):
    if is_script_mode:
        epoch_progress = lambda it: tqdm(it, desc="Epochs", total=config.trainer.epochs)
        batch_progress = lambda it: tqdm(it, desc="Batches", leave=False)
        improvement_step_progress = lambda it: tqdm(it, desc="Improvement steps", leave=False)
        val_batch_progress = lambda it: tqdm(it, desc="Validation batches", leave=False)
        val_improvement_step_progress = lambda it: tqdm(
            it, desc="Validation improvement steps", leave=False
        )
    else:
        epoch_progress = lambda it: mo.status.progress_bar(
            it, title="Epochs", completion_title="Training done", total=config.trainer.epochs
        )
        batch_progress = lambda it: mo.status.progress_bar(it, title="Batches", remove_on_exit=True)
        improvement_step_progress = lambda it: mo.status.progress_bar(
            it, title="Improvement steps", remove_on_exit=True
        )
        val_batch_progress = lambda it: mo.status.progress_bar(
            it, title="Validation batches", remove_on_exit=True
        )
        val_improvement_step_progress = lambda it: mo.status.progress_bar(
            it, title="Validation improvement steps", remove_on_exit=True
        )

    nds_module = NDSModule(
        config=config,
        model=model,
        improvement_step_progress=improvement_step_progress,
        validation_improvement_step_progress=val_improvement_step_progress,
    )

    def on_batch_train_end(summary: BatchTrainSummary):
        if config.logging.wandb_project:
            wandb.log({f"train/{k}": v for k, v in summary.model_dump().items()})

    def on_epoch_train_end(summary: EpochTrainSummary):
        mo.output.append(
            mo.md(f"**Epoch {summary.epoch + 1}** - Obj: {summary.objective_after:.4f}")
        )
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
                    plot_path / f"epoch_{val_summary.epoch:04d}.pdf",
                    bbox_inches="tight",
                )
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
            epoch_progress=epoch_progress,
            batch_progress=batch_progress,
            validation_batch_progress=val_batch_progress,
            strict_number_of_instances=config.trainer.strict_number_of_instances,
        ).fit(
            module=nds_module,
            optimizer=optimizer,
            instance_sampler=sampler,
            validation_instance_sampler=val_sampler,
        )
    except BaseException:
        gc.collect()
        torch.cuda.empty_cache()
        if config.logging.wandb_project:
            wandb.finish(exit_code=1)
        raise
    else:
        if config.logging.wandb_project:
            _final_checkpoint, _final_epoch = find_latest_checkpoint(run_dir)
            if _final_checkpoint is not None:
                upload_final_checkpoint(
                    run_name=config.run_name,
                    checkpoint_path=_final_checkpoint,
                    epoch=_final_epoch,
                )
            upload_validation_plots(
                run_name=config.run_name,
                plots_dir=run_dir / "plots",
            )
            wandb.finish()

    mo.md("**Training complete.**")
    return


if __name__ == "__main__":
    app.run()
