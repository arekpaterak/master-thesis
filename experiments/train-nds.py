import marimo

__generated_with = "0.23.6"
app = marimo.App(width="columns")

with app.setup(hide_code=True):
    import gc
    import sys
    import traceback
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import torch.nn
    import yaml
    from dotenv import load_dotenv
    from rich.console import Console
    from rich.table import Table
    from tqdm import tqdm
    from wigglystuff import EnvConfig

    import wandb
    from core.lns.acceptance_criterion import AcceptanceCriterionConfig
    from core.lns.config import SearchParams, StartSolutionParams
    from core.models.nds.baseline import BaselineConfig
    from core.models.nds.config import (
        JointModelConfig,
        LoggingConfig,
        OriginalPretrainedConfig,
        PositionalEncodingConfig,
        SplitModelConfig,
        TrainConfig,
        TrainerParams,
        ValidationConfig,
    )
    from core.models.nds.decoders.config import DecoderConfig
    from core.models.nds.encoders.config import JointEncoderConfig, SplitEncoderConfig
    from core.models.nds.factory import build_model
    from core.models.nds.module import NDSModule
    from core.models.nds.policy_loss import PolicyLossConfig
    from core.models.nds.pretrained.load import (
        load_original_pretrained_decoder_weights,
        load_original_pretrained_encoder_weights,
    )
    from core.models.nds.reward import RewardConfig
    from core.models.nds.training import BatchTrainSummary, EpochTrainSummary
    from core.models.nds.validation import EpochValidationSummary
    from core.models.optimizer import OptimizerConfig, build_optimizer
    from core.models.runtime import AmpConfig, CompileConfig
    from core.models.trainer import Trainer
    from core.models.utils import load_checkpoint, save_checkpoint
    from core.problems import make_problem_config
    from core.problems.dataset import DatasetSampler
    from core.problems.sampler import Sampler
    from core.results.plots import plot_search_progress
    from core.results.wandb_artifacts import upload_final_checkpoint, upload_validation_plots
    from core.ui.forms import build_mo_ui_dict, build_union_selector, collect_union_value

    load_dotenv(".env")

    TRAIN_CONFIGS_DIR = Path(__file__).parent / "configs" / "train"


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

        # marimo's CLI parser only keeps --flag args, so a bare config path passed
        # without --config is read from sys.argv. Only the first argument is treated
        # as positional, so it is not confused with a flag's value.
        _positional_config = (
            sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else None
        )

        if (not _cli_args and not _positional_config) or "help" in _cli_args or "h" in _cli_args:
            _usage = Table(title="NDS Training — CLI Usage")
            _usage.add_column("Flag", style="cyan")
            _usage.add_column("Description")
            _usage.add_row("<path>", "Path to YAML config file (positional)")
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

        _config_arg = _cli_args.get("config")
        _config_path = (str(_config_arg) if _config_arg is not None else _positional_config) or (
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
    preset_dropdown = mo.ui.dropdown(presets, value=presets[0], label="Training preset")
    preset_dropdown if not is_script_mode else None
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
    encoder_mode_widget = mo.ui.dropdown(
        ["joint", "split"], value=defaults.model.encoder_mode, label="Encoder mode"
    )
    mo.vstack([mo.md("### Encoder mode"), encoder_mode_widget])
    return (encoder_mode_widget,)


@app.cell(hide_code=True)
def _(defaults):
    _skip = {
        "encoder_mode",
        "positional_encoding",
        "encoder",
        "decoder",
        "init_weights",
        "original_pretrained_weights",
        "freeze_encoder",
    }
    model_form = build_mo_ui_dict(
        JointModelConfig,
        defaults.model if isinstance(defaults.model, JointModelConfig) else JointModelConfig(),
        skip=_skip,
        hints={
            "hidden_dim": {"options": [64, 128, 256, 512]},
            "ff_dim": {"options": [128, 256, 512, 1024]},
            "num_heads": {"start": 1, "stop": 16},
        },
    )
    mo.vstack([mo.md("### Model"), *model_form.values()])
    return (model_form,)


@app.cell(hide_code=True)
def _(defaults, encoder_mode_widget):
    # The split (LNS-native) encoder relies on positional encoding as its only
    # solution signal, so enabling it is mandatory and the toggle is locked on.
    _split = encoder_mode_widget.value == "split"
    positional_enabled_widget = mo.ui.checkbox(
        value=True if _split else defaults.model.positional_encoding is not None,
        label="Enable positional encoding (required for the split encoder)"
        if _split
        else "Enable positional encoding",
        disabled=_split,
    )
    positional_form = build_mo_ui_dict(PositionalEncodingConfig, defaults.model.positional_encoding)
    return positional_enabled_widget, positional_form


@app.cell(hide_code=True)
def _(positional_enabled_widget, positional_form):
    _items = [mo.md("### Positional encoding"), positional_enabled_widget]
    if positional_enabled_widget.value:
        _items.extend(positional_form.values())
    mo.vstack(_items)
    return


@app.cell(hide_code=True)
def _(defaults, encoder_mode_widget):
    _encoder_config_cls = (
        SplitEncoderConfig if encoder_mode_widget.value == "split" else JointEncoderConfig
    )
    _default_encoder = (
        defaults.model.encoder
        if isinstance(defaults.model.encoder, _encoder_config_cls)
        else _encoder_config_cls()
    )
    encoder_form = build_mo_ui_dict(_encoder_config_cls, _default_encoder)
    mo.vstack([mo.md("### Encoder"), *encoder_form.values()])
    return (encoder_form,)


@app.cell(hide_code=True)
def _(defaults):
    # Setup: create the widgets. No .value access here -- that must happen downstream.
    decoder_type, decoder_forms = build_union_selector(DecoderConfig, defaults.model.decoder)
    return decoder_forms, decoder_type


@app.cell(hide_code=True)
def _(decoder_forms, decoder_type):
    # Display: reads decoder_type.value to show the active variant's fields. Spreading
    # .values() renders nothing for a no-field variant (e.g. the simple decoder),
    # matching the acceptance selectors.
    mo.vstack(
        [
            mo.md("### Decoder"),
            decoder_type,
            *decoder_forms[decoder_type.value].values(),
        ]
    )
    return


@app.cell(hide_code=True)
def _(defaults, encoder_mode_widget):
    _pretrained_dir = Path("experiments/pretrained/nds")
    _pretrained_options = {"(none)": ""}
    for _pt in sorted(_pretrained_dir.glob("**/*.pt")):
        _pretrained_options[f"{_pt.parent.name}/{_pt.name}"] = _pt.as_posix()
    # Pretrained-NDS init and encoder freezing are joint-only (getattr guards a split default).
    _pretrained = getattr(defaults.model, "original_pretrained_weights", None)
    _pretrained_path = Path(_pretrained.path).as_posix() if _pretrained else ""
    _pretrained_key = next(
        (k for k, v in _pretrained_options.items() if v == _pretrained_path), "(none)"
    )
    pretrained_path_widget = mo.ui.dropdown(
        options=_pretrained_options,
        value=_pretrained_key,
        label="Original pretrained checkpoint",
    )
    pretrained_load_decoder_widget = mo.ui.checkbox(
        value=_pretrained.load_decoder if _pretrained else False,
        label="Load decoder from original pretrained",
    )
    init_weights_widget = mo.ui.text(
        value=defaults.model.init_weights or "",
        label="Own checkpoint path (empty = none)",
    )
    freeze_encoder_widget = mo.ui.checkbox(
        value=getattr(defaults.model, "freeze_encoder", False),
        label="Freeze encoder",
    )
    _items = [mo.md("### Weights"), init_weights_widget]
    if encoder_mode_widget.value == "joint":
        _items[1:1] = [pretrained_path_widget, pretrained_load_decoder_widget]
        _items.append(freeze_encoder_widget)
    mo.vstack(_items)
    return (
        freeze_encoder_widget,
        init_weights_widget,
        pretrained_load_decoder_widget,
        pretrained_path_widget,
    )


@app.cell(hide_code=True)
def _(defaults):
    trainer_form = build_mo_ui_dict(
        TrainerParams,
        defaults.trainer,
        skip={"training_acceptance"},
        hints={
            "batch_size": {"options": [8, 16, 32, 64, 128, 256, 512]},
            "num_main_improvement_steps": {"start": 10, "stop": 200, "step": 10},
        },
    )
    trainer_acceptance_type, trainer_acceptance_forms = build_union_selector(
        AcceptanceCriterionConfig, defaults.trainer.training_acceptance
    )
    return trainer_acceptance_forms, trainer_acceptance_type, trainer_form


@app.cell(hide_code=True)
def _(trainer_acceptance_forms, trainer_acceptance_type, trainer_form):
    mo.vstack(
        [
            mo.md("### Trainer"),
            *trainer_form.values(),
            mo.md("#### Training acceptance"),
            trainer_acceptance_type,
            *trainer_acceptance_forms[trainer_acceptance_type.value].values(),
        ]
    )
    return


@app.cell(hide_code=True)
def _(defaults):
    loss_type, loss_forms = build_union_selector(PolicyLossConfig, defaults.loss)
    return loss_forms, loss_type


@app.cell(hide_code=True)
def _(defaults):
    baseline_type, baseline_forms = build_union_selector(BaselineConfig, defaults.baseline)
    return baseline_forms, baseline_type


@app.cell(hide_code=True)
def _(loss_forms, loss_type):
    mo.vstack([mo.md("### Policy loss"), loss_type, *loss_forms[loss_type.value].values()])
    return


@app.cell(hide_code=True)
def _(baseline_forms, baseline_type):
    mo.vstack([mo.md("### Baseline"), baseline_type, *baseline_forms[baseline_type.value].values()])
    return


@app.cell(hide_code=True)
def _(defaults):
    reward_type, reward_forms = build_union_selector(RewardConfig, defaults.reward)
    return reward_forms, reward_type


@app.cell(hide_code=True)
def _(reward_forms, reward_type):
    mo.vstack([mo.md("### Reward"), reward_type, *reward_forms[reward_type.value].values()])
    return


@app.cell(hide_code=True)
def _(defaults):
    search_form = build_mo_ui_dict(
        SearchParams,
        defaults.search,
        skip={"acceptance", "start_solution_params"},
        hints={
            "num_improvement_steps": {"label": "Initial steps"},
            "deconstruction_size": {"start": 5, "stop": 25, "step": 5},
        },
    )
    start_solution_form = build_mo_ui_dict(
        StartSolutionParams, defaults.search.start_solution_params
    )
    acceptance_type, acceptance_forms = build_union_selector(
        AcceptanceCriterionConfig, defaults.search.acceptance
    )
    return acceptance_forms, acceptance_type, search_form, start_solution_form


@app.cell(hide_code=True)
def _(acceptance_forms, acceptance_type, search_form, start_solution_form):
    mo.vstack(
        [
            mo.md("### Search"),
            *search_form.values(),
            mo.md("#### Start solution warmup"),
            *start_solution_form.values(),
            mo.md("#### Acceptance"),
            acceptance_type,
            *acceptance_forms[acceptance_type.value].values(),
        ]
    )
    return


@app.cell(hide_code=True)
def _(defaults):
    optimizer_form = build_mo_ui_dict(OptimizerConfig, defaults.optimizer)
    mo.vstack([mo.md("### Optimizer"), *optimizer_form.values()])
    return (optimizer_form,)


@app.cell(hide_code=True)
def _(defaults):
    amp_form = build_mo_ui_dict(AmpConfig, defaults.amp)
    mo.vstack([mo.md("### Mixed precision (AMP)"), *amp_form.values()])
    return (amp_form,)


@app.cell(hide_code=True)
def _(defaults):
    compile_form = build_mo_ui_dict(CompileConfig, defaults.compile)
    mo.vstack([mo.md("### torch.compile"), *compile_form.values()])
    return (compile_form,)


@app.cell(hide_code=True)
def _(defaults):
    logging_form = build_mo_ui_dict(LoggingConfig, defaults.logging)
    mo.vstack([mo.md("### Logging"), *logging_form.values()])
    return (logging_form,)


@app.cell(hide_code=True)
def _(defaults):
    val_form = build_mo_ui_dict(
        ValidationConfig, defaults.validation, skip={"search", "dataset_path"}
    )
    _initial_path = (
        str(Path(defaults.validation.dataset_path).parent)
        if defaults.validation.dataset_path
        else str(Path(__file__).parent.parent / "data")
    )
    val_dataset_path_widget = mo.ui.file_browser(
        initial_path=_initial_path,
        multiple=False,
        label="Validation dataset (override; preset default used if none selected)",
    )
    _active = defaults.validation.dataset_path or "(none -> sample randomly)"
    mo.vstack(
        [
            mo.md("### Validation"),
            *val_form.values(),
            mo.md(f"Current validation dataset: `{_active}`"),
            val_dataset_path_widget,
        ]
    )
    return val_dataset_path_widget, val_form


@app.cell(hide_code=True)
def _(defaults):
    val_search_form = build_mo_ui_dict(
        SearchParams,
        defaults.validation.search,
        skip={"acceptance", "start_solution_params"},
        hints={"num_rollouts": {"label": "Rollouts per improvement step"}},
    )
    val_start_solution_form = build_mo_ui_dict(
        StartSolutionParams, defaults.validation.search.start_solution_params
    )
    val_acceptance_type, val_acceptance_forms = build_union_selector(
        AcceptanceCriterionConfig, defaults.validation.search.acceptance
    )
    return (
        val_acceptance_forms,
        val_acceptance_type,
        val_search_form,
        val_start_solution_form,
    )


@app.cell(hide_code=True)
def _(
    val_acceptance_forms,
    val_acceptance_type,
    val_search_form,
    val_start_solution_form,
):
    mo.vstack(
        [
            mo.md("### Validation: Search"),
            *val_search_form.values(),
            mo.md("#### Start solution warmup"),
            *val_start_solution_form.values(),
            mo.md("#### Acceptance"),
            val_acceptance_type,
            *val_acceptance_forms[val_acceptance_type.value].values(),
        ]
    )
    return


@app.cell(hide_code=True)
def _(is_script_mode):
    run_button = mo.ui.run_button(label="Start Training")
    run_button if not is_script_mode else None
    return (run_button,)


@app.cell(hide_code=True)
def _(
    acceptance_forms,
    acceptance_type,
    amp_form,
    baseline_forms,
    baseline_type,
    compile_form,
    decoder_forms,
    decoder_type,
    defaults,
    encoder_form,
    encoder_mode_widget,
    freeze_encoder_widget,
    init_weights_widget,
    is_script_mode,
    logging_form,
    loss_forms,
    loss_type,
    model_form,
    optimizer_form,
    positional_enabled_widget,
    positional_form,
    pretrained_load_decoder_widget,
    pretrained_path_widget,
    problem_name,
    problem_size,
    reward_forms,
    reward_type,
    run_button,
    search_form,
    start_solution_form,
    trainer_acceptance_forms,
    trainer_acceptance_type,
    trainer_form,
    val_acceptance_forms,
    val_acceptance_type,
    val_dataset_path_widget,
    val_form,
    val_search_form,
    val_start_solution_form,
    yaml_data,
):
    if is_script_mode:
        config = TrainConfig.model_validate(yaml_data)
    else:
        mo.stop(
            not run_button.value,
            mo.md("Click 'Start Training' to begin."),
        )
        _pe_enabled = positional_enabled_widget.value or encoder_mode_widget.value == "split"
        _encoder_config_cls = (
            SplitEncoderConfig if encoder_mode_widget.value == "split" else JointEncoderConfig
        )
        _shared_model_kwargs = dict(
            **model_form.value,
            positional_encoding=(
                PositionalEncodingConfig(**positional_form.value) if _pe_enabled else None
            ),
            encoder=_encoder_config_cls(**encoder_form.value),
            decoder=collect_union_value(DecoderConfig, decoder_type, decoder_forms),
            init_weights=init_weights_widget.value.strip() or None,
        )
        if encoder_mode_widget.value == "split":
            _model = SplitModelConfig(**_shared_model_kwargs)
        else:
            _pretrained_path = pretrained_path_widget.value
            _model = JointModelConfig(
                **_shared_model_kwargs,
                original_pretrained_weights=(
                    OriginalPretrainedConfig(
                        path=_pretrained_path,
                        load_decoder=pretrained_load_decoder_widget.value,
                    )
                    if _pretrained_path
                    else None
                ),
                freeze_encoder=freeze_encoder_widget.value,
            )
        config = TrainConfig(
            problem=make_problem_config(
                name=problem_name.value,
                size=problem_size.value,
            ),
            model=_model,
            optimizer=OptimizerConfig(**optimizer_form.value),
            loss=collect_union_value(PolicyLossConfig, loss_type, loss_forms),
            baseline=collect_union_value(BaselineConfig, baseline_type, baseline_forms),
            reward=collect_union_value(RewardConfig, reward_type, reward_forms),
            amp=AmpConfig(**amp_form.value),
            compile=CompileConfig(**compile_form.value),
            trainer=TrainerParams(
                **trainer_form.value,
                training_acceptance=collect_union_value(
                    AcceptanceCriterionConfig, trainer_acceptance_type, trainer_acceptance_forms
                ),
            ),
            search=SearchParams(
                **search_form.value,
                start_solution_params=StartSolutionParams(**start_solution_form.value),
                acceptance=collect_union_value(
                    AcceptanceCriterionConfig, acceptance_type, acceptance_forms
                ),
            ),
            logging=LoggingConfig(**logging_form.value),
            validation=ValidationConfig(
                **val_form.value,
                dataset_path=(
                    val_dataset_path_widget.value[0].path
                    if val_dataset_path_widget.value
                    else defaults.validation.dataset_path
                ),
                search=SearchParams(
                    **val_search_form.value,
                    start_solution_params=StartSolutionParams(**val_start_solution_form.value),
                    acceptance=collect_union_value(
                        AcceptanceCriterionConfig, val_acceptance_type, val_acceptance_forms
                    ),
                ),
            ),
        )
    return (config,)


@app.cell(hide_code=True)
def _(encoder_form, is_script_mode):
    if not is_script_mode:
        _enc = encoder_form.value
        mo.vstack(
            [
                mo.md("### Encoder config (live preview)"),
                mo.md(
                    f"norm=**`{_enc.get('norm_type', '?')}`** &nbsp; "
                    f"pre_norm=**`{_enc.get('pre_norm', '?')}`** &nbsp; "
                    f"ff=**`{_enc.get('ff_activation', '?')}`**"
                ),
            ]
        )
    return


@app.cell(hide_code=True)
def _(config):
    _config_yaml = yaml.safe_dump(config.model_dump(exclude={"run_name"}), sort_keys=False)
    print(_config_yaml)
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    ## Problem Setup
    """)
    return


@app.cell
def _(config):
    sampler = Sampler.from_config(config.problem)
    if config.validation.dataset_path is None:
        val_sampler = Sampler.from_config(config.problem, seed=config.validation.seed)
    else:
        val_sampler = DatasetSampler.from_file(config.problem.name, config.validation.dataset_path)
        _val_dataset_size = val_sampler.num_nodes - 1
        if _val_dataset_size != config.problem.size:
            raise ValueError(
                f"Validation dataset instances have size {_val_dataset_size} but "
                f"problem.size is {config.problem.size}."
            )
        if config.validation.num_instances > len(val_sampler):
            raise ValueError(
                f"validation.num_instances={config.validation.num_instances} exceeds "
                f"the {len(val_sampler)} instances in {config.validation.dataset_path}."
            )
    return sampler, val_sampler


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Model Setup
    """)
    return


@app.cell
def _(config):
    model = build_model(config.model, config.problem.name)

    # original_pretrained_weights / freeze_encoder are joint-only; the split model
    # cannot initialise from a pretrained NDS encoder.
    _is_joint = isinstance(config.model, JointModelConfig)
    if _is_joint and config.model.original_pretrained_weights:
        _pretrained = config.model.original_pretrained_weights
        load_original_pretrained_encoder_weights(model.encoder, _pretrained.path)
        print(f"Loaded pretrained encoder weights from {_pretrained.path}")
        if _pretrained.load_decoder:
            load_original_pretrained_decoder_weights(model.decoder, _pretrained.path)
            print(f"Loaded pretrained decoder weights from {_pretrained.path}")

    elif config.model.init_weights:
        load_checkpoint(config.model.init_weights, model)
        print(f"Initiated model with weights from {config.model.init_weights}")

    if _is_joint and config.model.freeze_encoder:
        model.encoder.requires_grad_(False)
        print("Frozen encoder weights")

    # Built on CPU; the training cell moves it to the GPU for the run and back to CPU
    # afterwards, so the GPU is only occupied while training is actually running.
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
        wandb.config.update(
            {
                "model/num_params": sum(p.numel() for p in model.parameters()),
                "model/num_params_trainable": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "model/num_params_decoder": sum(p.numel() for p in model.decoder.parameters()),
            },
            allow_val_change=True,
        )
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("train/epoch_time_sec", step_metric="epoch")
    return find_latest_checkpoint, initial_epoch, run_dir


@app.cell(hide_code=True)
def _(
    config,
    device,
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

    model.to(device)
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
        if config.logging.wandb_project:
            wandb.log({"train/epoch_time_sec": summary.epoch_time_sec, "epoch": summary.epoch})
        is_last = summary.epoch == config.trainer.epochs - 1
        is_checkpoint = (summary.epoch + 1) % config.trainer.checkpoint_every_n_epochs == 0
        if is_last or is_checkpoint:
            checkpoints_path = run_dir / "checkpoints"
            checkpoints_path.mkdir(exist_ok=True)
            save_checkpoint(
                path=checkpoints_path / f"epoch_{summary.epoch:04d}.pt",
                model=nds_module.model,
                optimizer=optimizer,
                epoch=summary.epoch,
                config=config,
            )

    def on_epoch_validation_end(val_summary: EpochValidationSummary):
        fig = None
        if config.validation.plot_enabled:
            fig = plot_search_progress(
                val_summary.snapshots, val_summary.step_labels, epoch=val_summary.epoch
            )
            if config.validation.plot_save_to_disk:
                plot_path = run_dir / "plots"
                plot_path.mkdir(parents=True, exist_ok=True)
                _stem = plot_path / f"epoch_{val_summary.epoch:04d}"
                fig.savefig(f"{_stem}.pdf", bbox_inches="tight")
                fig.savefig(f"{_stem}.png", bbox_inches="tight", dpi=150)
        if config.logging.wandb_project:
            _log = {
                "val/objective_initial": val_summary.objective_initial,
                "val/objective_final": val_summary.objective_final,
                "val/improvement": val_summary.improvement,
                "val/relative_improvement": val_summary.relative_improvement,
                "val/batch_time_sec": val_summary.batch_time_sec,
                "val/epoch_time_sec": val_summary.epoch_time_sec,
                "val/instances_per_sec": val_summary.instances_per_sec,
                "val/steps_to_best": val_summary.steps_to_best,
                "val/diversity_overlap": val_summary.diversity_overlap,
                "val/unique_rollouts": val_summary.unique_rollouts,
                "val/objective_curve": wandb.plot.line_series(
                    xs=val_summary.step_labels,
                    ys=[val_summary.objective_curve],
                    keys=["objective"],
                    title=f"Validation objective (epoch {val_summary.epoch})",
                    xname="Improvement step",
                ),
                "epoch": val_summary.epoch,
            }
            if val_summary.objective_final_augmented is not None:
                _log["val/objective_final_augmented"] = val_summary.objective_final_augmented
            if fig is not None:
                _log["val/search_progress"] = wandb.Image(fig)
            wandb.log(_log)
        if fig is not None:
            plt.close(fig)

    optimizer = build_optimizer(config.optimizer, nds_module.parameters())

    if resume_run_dir:
        _checkpoint_path, _ = find_latest_checkpoint(resume_run_dir)
        if _checkpoint_path:
            _ckpt = torch.load(_checkpoint_path, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(_ckpt["optimizer"])

    _error: Exception | None = None
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
            validate_before_training=config.validation.validate_before_training,
        ).fit(
            module=nds_module,
            optimizer=optimizer,
            instance_sampler=sampler,
            validation_instance_sampler=val_sampler,
        )
    except Exception as exc:
        # Surface the error instead of silently ending the run, then re-raise below so
        # the process exits non-zero (after the cleanup in `finally`).
        traceback.print_exc()
        _error = exc
    finally:
        if config.logging.wandb_project:
            _final_checkpoint, _final_epoch = find_latest_checkpoint(run_dir)
            if _final_checkpoint is not None:
                upload_final_checkpoint(
                    run_name=config.run_name,
                    checkpoint_path=_final_checkpoint,
                    epoch=_final_epoch,
                )
            if _error is None:
                upload_validation_plots(
                    run_name=config.run_name,
                    plots_dir=run_dir / "plots",
                )
            wandb.finish(exit_code=1 if _error is not None else 0)
        # Free the GPU whether training finished or errored: the trained weights are
        # persisted in checkpoints and the in-memory model only needs to live on CPU.
        model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    if _error is not None:
        raise _error

    mo.md("**Training completed.**")
    return


if __name__ == "__main__":
    app.run()
