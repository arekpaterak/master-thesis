Train the NDS model:

```bash
python experiments/train-nds.py --config experiments/configs/train/nds_cvrp_100.yml
```

Evaluate a method:

```bash
python experiments/eval.py --config experiments/configs/eval/random_destroy.yml
```

`--config` also accepts a bare name resolved under `experiments/configs/eval/`
(e.g. `--config random_destroy.yml`), and `--device cpu|cuda` overrides the device.

An eval config binds one method to a list of problem settings:

- `configs/eval/<name>.yml` -- the method (a `runs/featured/...` run, a pretrained
  checkpoint, or random-destroy) plus its `search`, and the `problems` to evaluate on
  (in-distribution, out-of-distribution sizes, CVRPLIB).
- `configs/eval/problems/*.yml` -- reusable problem settings.
- `configs/eval/search/*.yml` -- reusable search settings.

`method.search` and each `problems` entry may be a string path (relative to
`configs/eval/`) or an inline block. 
Worthy training runs are moved by hand into
`experiments/runs/featured/` and referenced by `method.run_dir`.
