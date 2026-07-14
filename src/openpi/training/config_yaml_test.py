"""Tests for the YAML config registry.

The one-time Python->YAML migration was verified by comparing every config against its Python
original (see scripts/export_configs_to_yaml.py). These are the guards that keep the tree honest
afterwards: every file still builds, names stay unique, and -- the important one -- every config
still survives an encode/decode round-trip, so a schema change that the codec cannot express
fails here rather than silently dropping a field at train time.
"""

import dataclasses
import pathlib
import textwrap

import pytest

import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.training.config_yaml as config_yaml
import openpi.training.weight_loaders as weight_loaders


def _all_configs():
    return list(_config._CONFIGS)  # noqa: SLF001


def test_registry_is_not_empty():
    assert len(_all_configs()) > 0, "no configs were discovered under configs/"


def test_every_yaml_file_builds_a_config():
    files = config_yaml.iter_config_files()
    assert len(files) == len(_all_configs())


def test_config_names_are_unique_and_match_their_filename():
    for path in config_yaml.iter_config_files():
        body = config_yaml.read_file(path)
        expected = body.get("name") or config_yaml.name_for(path)
        assert _config.get_config(expected).name == expected, path


@pytest.mark.parametrize("config", _all_configs(), ids=lambda c: c.name)
def test_round_trips_through_yaml(config):
    """encode -> decode must reproduce the config. Catches a codec that silently drops a field."""
    body = config_yaml.to_yaml_dict(config)
    rebuilt = config_yaml.build_config(body, name=config.name)
    assert config_yaml.configs_equal(rebuilt, config)


@pytest.mark.parametrize("config", _all_configs(), ids=lambda c: c.name)
def test_transform_groups_are_constructible(config):
    """Every transform factory must actually build its group.

    This is the check the Python registry never had: five roboarena configs passed an
    `action_dim=` kwarg that `DroidInputs` does not accept, so they raised TypeError the moment
    anything touched their data pipeline -- and nothing did until the migration called them.
    """
    for attr in ("data_transforms", "model_transforms"):
        factory = getattr(config.data, attr, None)
        if callable(factory):
            factory(config.model)


# ---- the codec itself ----


def _build(text: str, name: str = "t"):
    import yaml

    return config_yaml.build_config(yaml.safe_load(textwrap.dedent(text)), name=name)


def test_unknown_field_is_rejected():
    with pytest.raises(config_yaml.ConfigError, match="unknown field"):
        _build("batch_size: 8\nnot_a_field: 3\n")


def test_unknown_type_tag_is_rejected():
    with pytest.raises(config_yaml.ConfigError, match="unknown type 'nope'"):
        _build("model:\n  type: nope\n")


def test_unknown_nested_field_is_rejected():
    with pytest.raises(config_yaml.ConfigError, match="unknown field"):
        _build("model:\n  type: pi0\n  nope: 1\n")


def test_enum_is_decoded_by_name():
    cfg = _build("""
        data:
          type: rlds_droid
          repo_id: x
          action_space: JOINT_POSITION
        """)
    assert cfg.data.action_space.name == "JOINT_POSITION"


def test_bad_enum_member_is_rejected():
    with pytest.raises(config_yaml.ConfigError, match="not a member"):
        _build("data:\n  type: rlds_droid\n  repo_id: x\n  action_space: SIDEWAYS\n")


def test_extends_inherits_then_overrides():
    base = _config.get_config("pi05droid-full-d100")
    cfg = _build("extends: pi05droid-full-d100\nbatch_size: 999\n", name="child")

    assert cfg.name == "child"
    assert cfg.batch_size == 999
    # Everything not named is inherited.
    assert cfg.num_train_steps == base.num_train_steps
    assert cfg.weight_loader.params_path == base.weight_loader.params_path
    assert cfg.data.repo_id == base.data.repo_id


def test_extends_unknown_base_is_rejected():
    with pytest.raises(config_yaml.ConfigError, match="extends"):
        _build("extends: no-such-config\n")


def test_freeze_filter_from_model():
    cfg = _build("""
        model:
          type: pi0
          pi05: true
          paligemma_variant: gemma_2b_lora
          action_expert_variant: gemma_300m_lora
        freeze_filter: from_model
        """)
    expected = pi0_config.Pi0Config(
        pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
    ).get_freeze_filter()
    assert repr(cfg.freeze_filter) == repr(expected)


def test_model_interpolation_is_resolved_against_the_model():
    """`${model.x}` is what replaced the `lambda model: Group(...)` closures."""
    cfg = _build("""
        model:
          type: pi0_fast
          action_dim: 8
        data:
          type: simple
          repo_id: x
          data_transforms:
            outputs:
              - type: extract_fast_actions
                tokenizer: fast_tokenizer
                action_dim: "${model.action_dim}"
                action_horizon: "${model.action_horizon}"
        """)
    group = cfg.data.data_transforms(cfg.model)
    assert group.outputs[0].action_dim == cfg.model.action_dim == 8
    assert group.outputs[0].action_horizon == cfg.model.action_horizon


def test_weight_loader_shorthand():
    assert isinstance(_build("weight_loader: none\n").weight_loader, weight_loaders.NoOpWeightLoader)


def test_post_init_validation_still_applies():
    with pytest.raises(config_yaml.ConfigError, match="resume and overwrite"):
        _build("resume: true\noverwrite: true\n")


def test_configs_equal_ignores_list_vs_tuple():
    """Sequence fields are spelled both ways in openpi; the container type carries no meaning."""
    import openpi.transforms as transforms

    a = transforms.Group(inputs=[], outputs=())
    b = transforms.Group(inputs=(), outputs=[])
    assert config_yaml._same(a, b)  # noqa: SLF001


def test_configs_equal_still_catches_a_real_difference():
    a = _config.get_config("pi05droid-full-d100")
    b = dataclasses.replace(a, batch_size=a.batch_size + 1)
    assert not config_yaml.configs_equal(a, b)


def test_loading_a_directory_of_configs(tmp_path: pathlib.Path):
    (tmp_path / "a.yaml").write_text("batch_size: 4\n")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.yaml").write_text("extends: a\nbatch_size: 8\n")
    (tmp_path / "_ignored.yaml").write_text("!!! not even yaml\n")

    configs = {c.name: c for c in config_yaml.load_configs(tmp_path, register=False)}

    assert set(configs) == {"a", "b"}, "a leading underscore marks a file as not-a-config"
    assert configs["a"].batch_size == 4
    assert configs["b"].batch_size == 8, "b extends a, which is defined in a different directory"


def test_extends_resolves_regardless_of_file_order(tmp_path: pathlib.Path):
    """`z` is loaded before `a` alphabetically, but extends it -- the loader must defer it."""
    (tmp_path / "z.yaml").write_text("extends: a\nseed: 7\n")
    (tmp_path / "a.yaml").write_text("batch_size: 4\n")

    configs = {c.name: c for c in config_yaml.load_configs(tmp_path, register=False)}
    assert configs["z"].batch_size == 4
    assert configs["z"].seed == 7


def test_extends_cycle_is_reported_as_a_cycle(tmp_path: pathlib.Path):
    """Not as a bogus "'a' not found. Did you mean...?" about a config that plainly exists."""
    (tmp_path / "a.yaml").write_text("extends: b\n")
    (tmp_path / "b.yaml").write_text("extends: a\n")
    with pytest.raises(config_yaml.ConfigError, match="cycle"):
        config_yaml.load_configs(tmp_path, register=False, strict=True)


def test_extends_missing_base_is_still_reported_as_missing(tmp_path: pathlib.Path):
    (tmp_path / "a.yaml").write_text("extends: nope\n")
    with pytest.raises(config_yaml.ConfigError, match="not found"):
        config_yaml.load_configs(tmp_path, register=False, strict=True)


# ---- overriding the model must re-derive freeze_filter ----


def test_extends_with_a_model_override_rederives_freeze_filter():
    """The base's filter is an nnx tree built from the BASE's model.

    Inheriting it verbatim after changing the model freezes parameters that no longer exist: extend
    the LoRA config with a full model and the whole LLM stays frozen, so a "full finetune" trains
    almost nothing -- silently, with only a flat loss curve on the TPU to show for it.
    """
    lora = _config.get_config("pi05droid-lora-d100+toys")
    assert not _same_filter(lora.freeze_filter, pi0_config.Pi0Config().get_freeze_filter())

    child = config_yaml.build_config(
        {
            "extends": "pi05droid-lora-d100+toys",
            "model": {"type": "pi0", "pi05": True, "action_dim": 32, "action_horizon": 16},
        },
        name="full_child",
    )
    assert _same_filter(child.freeze_filter, child.model.get_freeze_filter())


def test_extends_without_a_model_override_keeps_the_base_freeze_filter():
    child = config_yaml.build_config(
        {"extends": "pi05droid-lora-d100+toys", "batch_size": 8}, name="lora_child"
    )
    base = _config.get_config("pi05droid-lora-d100+toys")
    assert _same_filter(child.freeze_filter, base.freeze_filter)


def _same_filter(a, b) -> bool:
    return repr(a) == repr(b)


# ---- scalars are coerced to their declared type ----
#
# PyYAML follows YAML 1.1, where `1e-5` is NOT a float (it wants `1.0e-5`). Untyped, that arrives as
# the string '1e-5', builds a TrainConfig happily, and dies inside the optimizer AFTER the TPU has
# been allocated. js-yaml (which the launcher uses) reads the same text as a number, so the two
# would silently disagree about what the config says.


@pytest.mark.parametrize(
    ("body", "path", "expected"),
    [
        ({"lr_schedule": {"type": "cosine_decay", "peak_lr": "1e-5"}}, ("lr_schedule", "peak_lr"), 1e-5),
        ({"batch_size": "32"}, ("batch_size",), 32),
        ({"ema_decay": "0.99"}, ("ema_decay",), 0.99),
    ],
)
def test_scalars_are_coerced_to_their_declared_type(body, path, expected):
    cfg = _build_dict(body)
    got = cfg
    for key in path:
        got = getattr(got, key)
    assert got == expected
    assert type(got) is type(expected)


@pytest.mark.parametrize(
    "body",
    [
        {"num_train_steps": "notanint"},
        {"batch_size": True},  # bool is an int subclass; `batch_size: true` is a mistake
        {"num_train_steps": 1.5},
        {"pytorch_training_precision": "float64"},  # not in the Literal
    ],
)
def test_a_scalar_of_the_wrong_type_is_rejected(body):
    with pytest.raises(config_yaml.ConfigError):
        _build_dict(body)


def _build_dict(body):
    return config_yaml.build_config(body, name="t")


@pytest.mark.parametrize(
    ("repo_id", "want"),
    [
        ("me/one", "me/one"),
        (["lerobot/droid_1.0.1", "me/toys"], ["lerobot/droid_1.0.1", "me/toys"]),
    ],
)
def test_repo_id_keeps_its_shape(repo_id, want):
    """`repo_id` is `str | Sequence[str]`, so a union member has to be chosen by the value's shape.

    Picking the first member instead would coerce a mixture's LIST of repo ids into the string
    "['lerobot/droid_1.0.1', 'me/toys']" -- one nonsense repo name, and a multi-dataset run silently
    training on nothing.
    """
    cfg = _build_dict({"data": {"type": "lerobot_droid", "repo_id": repo_id}})
    assert cfg.data.repo_id == want
    assert type(cfg.data.repo_id) is type(want)


# ---- one bad file must not take down the registry ----
#
# load_configs runs at `import openpi.training.config` time, so a raise there would mean one typo in
# one config file bricks the whole package: you could not train a DIFFERENT config, serve a policy,
# or even collect the test suite.


def test_a_broken_file_does_not_stop_the_others(tmp_path: pathlib.Path):
    (tmp_path / "good.yaml").write_text("batch_size: 4\n")
    (tmp_path / "broken.yaml").write_text("model:\n  type: pi0\n  bogus_field: 1\n")
    (tmp_path / "unparseable.yaml").write_text("key: [unclosed\n")

    configs = config_yaml.load_configs(tmp_path, register=False)

    assert [c.name for c in configs] == ["good"], "the good config still loads"


def test_strict_mode_raises_on_a_broken_file(tmp_path: pathlib.Path):
    (tmp_path / "broken.yaml").write_text("model:\n  type: pi0\n  bogus_field: 1\n")
    with pytest.raises(config_yaml.ConfigError, match="bogus_field"):
        config_yaml.load_configs(tmp_path, register=False, strict=True)


def test_asking_for_a_broken_config_reports_its_actual_error(monkeypatch, tmp_path: pathlib.Path):
    """`get_config` must not say "not found -- did you mean...?" when the file is simply wrong."""
    (tmp_path / "typo_cfg.yaml").write_text("model:\n  type: pi0\n  bogus_field: 1\n")
    monkeypatch.setattr(config_yaml, "BROKEN", {})
    config_yaml.load_configs(tmp_path, register=False)
    # register=False leaves BROKEN alone, so seed it the way an import-time load would.
    try:
        config_yaml.load_file(tmp_path / "typo_cfg.yaml")
    except config_yaml.ConfigError as e:
        config_yaml.BROKEN["typo_cfg"] = e

    with pytest.raises(ValueError, match="failed to load.*bogus_field"):
        _config.get_config("typo_cfg")
