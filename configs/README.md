# Training configs

Every `TrainConfig` lives here as YAML, one file per config. The file's **stem is the config's
name** — the argument to `train.py`:

```bash
uv run scripts/train.py pi05droid-full-d100 --exp-name=run1
uv run scripts/compute_norm_stats.py --config-name pi05droid-full-d100
```

The subdirectories (`pi05base/`, `pi05droid/`, `pi05polaris/`, `aloha/`, …) only group the files;
they are not part of a config's name, and names must be unique across the whole tree.

`src/openpi/training/config_yaml.py` loads these into the same `_CONFIGS` registry that used to
hold ~2k lines of Python `TrainConfig(...)` literals, so `get_config()`, `train.py`'s tyro CLI
(including `--data.repo-id=…` style overrides) and `serve_policy.py` all work unchanged.

## Writing one

Only what you want to differ from the dataclass defaults needs to be written. A field whose type is
polymorphic (`model`, `data`, `weight_loader`, `lr_schedule`, `optimizer`) carries a `type:` tag
naming its implementation:

```yaml
model:
  type: pi0                 # -> Pi0Config
  pi05: true
  action_dim: 32
  action_horizon: 16
weight_loader:
  type: checkpoint          # -> CheckpointWeightLoader
  params_path: gs://openpi-assets/checkpoints/pi05_droid/params
data:
  type: lerobot_droid       # -> LeRobotDROIDDataConfig
  repo_id: SamratSahoo/d100
  assets:
    # Reuse the pretrained checkpoint's norm stats instead of computing new ones.
    assets_dir: gs://openpi-assets/checkpoints/pi05_droid/assets
    asset_id: droid
  base_config:
    prompt_from_task: true
num_train_steps: 20000
batch_size: 32
```

The registries of `type:` names (models, data pipelines, weight loaders, optimizers, transforms)
are at the top of `config_yaml.py`.

## Starting from an existing config

`extends:` inherits every field of another config, and anything you write overrides it:

```yaml
extends: pi05droid-full-d100
num_train_steps: 50000
batch_size: 64
```

`extends` is a **shallow replace** (it is `dataclasses.replace` on `TrainConfig`). Overriding
`data:` therefore replaces the *whole* data block rather than merging into it — so a partial `data:`
in a child cannot silently keep the base's `repo_id`. Write the block out in full when you change it.

The TPU launcher (`tpu/`) has a UI for this: pick a base, edit the fields, and it writes the file.

## Two things that are not plain data

`freeze_filter` holds an `nnx` filter tree, always derived from the model, so it is written as
`freeze_filter: from_model` (pair it with the LoRA model variants — see
`pi05droid/pi05droid-lora-d100+toys.yaml`).

Transform groups may need values from the model config, which is not known until the data pipeline
is built. `${model.<field>}` defers the read:

```yaml
data:
  type: simple
  data_transforms:
    inputs:
      - {type: droid_inputs, model_type: PI05}
    outputs:
      - {type: droid_outputs}
```

## Regenerating / testing

`scripts/export_configs_to_yaml.py` is what performed the original migration — it introspects the
live Python objects and emits this tree, checking that each file rebuilds the config it came from.
It stays in the repo for re-exporting after a schema change.

`src/openpi/training/config_yaml_test.py` asserts every config here still builds, that names are
unique, that each survives an encode/decode round-trip, and that every transform group is actually
constructible.

```bash
uv run pytest src/openpi/training/config_yaml_test.py
```
