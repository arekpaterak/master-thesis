Train the NDS model:

```bash
python experiments/train.py --config experiments/configs/train/cvrp/nds_cvrp_100.yml
```

Evaluate a method:

```bash
python experiments/eval.py --config experiments/configs/eval/cvrp/random_destroy.yml
```

`--config` also accepts a bare name resolved under `experiments/configs/eval/`
(e.g. `--config cvrp/random_destroy.yml`), `--device cpu|cuda` overrides the device,
and `--setting <label>` (repeatable) runs only some settings (e.g. `--setting cvrp200`).

One eval config per method lives at `configs/eval/<problem>/<method>.yml`. Results are
keyed by method and setting: a run writes to `results/eval/<problem>/<method>/<setting>/`
(e.g. `results/eval/cvrp/random_destroy/cvrp100/`), holding `summary.csv`, `steps.csv`,
`instances.csv`, and `plots/`. Re-running a config only recomputes settings that have no
`summary.csv` yet (use `--force` to redo them), so partial runs accumulate in place.
Pass `--tmp` to rerun a config into a `tmp/` subdir of the method dir instead
(e.g. `results/eval/cvrp/random_destroy/tmp/cvrp100/`), leaving the real per-setting
results untouched -- useful for a throwaway rerun without overwriting results.

An eval config binds one method to settings grouped by role:

- `configs/eval/<problem>/<method>.yml` -- the method (a `runs/<problem>/featured/...` run,
  a pretrained checkpoint, or random-destroy) plus its `search`, and `settings` grouped
  as `{main: [...], generalisation: [...]}` (in-distribution vs out-of-distribution sizes
  and CVRPLIB). The group is recorded on each result row.
- `configs/eval/<problem>/problems/*.yml` -- reusable settings.
- `configs/eval/<problem>/search/*.yml` -- reusable search settings.

`method.search` and each setting entry may be a string path (relative to the config) or
an inline block. Worthy training runs are moved by hand into
`experiments/runs/<problem>/featured/` and referenced by `method.run_dir`.
