Train the NDS model:

```bash
python experiments/train.py --config experiments/configs/train/nds_cvrp_100.yml
```

Evaluate a method:

```bash
python experiments/eval.py --config experiments/configs/eval/cvrp/random_destroy.yml
```

`--config` also accepts a bare name resolved under `experiments/configs/eval/`
(e.g. `--config cvrp/random_destroy.yml`), and `--device cpu|cuda` overrides the device.

Eval configs are grouped by problem, e.g. `configs/eval/cvrp/`. The config's location
under `configs/eval/` is mirrored in the results tree, so
`cvrp/random_destroy.yml` writes to `results/eval/cvrp/random_destroy/<eval_id>/`.

An eval config binds one method to a list of problem settings:

- `configs/eval/<problem>/<name>.yml` -- the method (a `runs/featured/...` run, a
  pretrained checkpoint, or random-destroy) plus its `search`, and the `problems` to
  evaluate on (in-distribution, out-of-distribution sizes, CVRPLIB).
- `configs/eval/<problem>/problems/*.yml` -- reusable problem settings.
- `configs/eval/<problem>/search/*.yml` -- reusable search settings.

`method.search` and each `problems` entry may be a string path (relative to
`configs/eval/`) or an inline block. 
Worthy training runs are moved by hand into
`experiments/runs/featured/` and referenced by `method.run_dir`.
