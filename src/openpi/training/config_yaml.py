"""YAML-defined training configs.

Every `configs/**/*.yaml` file under the repo root becomes a first-class
`TrainConfig`, selectable exactly like a Python one:

    uv run scripts/train.py pi05droid-full-d100 --exp-name=run1
    uv run scripts/compute_norm_stats.py --config-name pi05droid-full-d100

The directory a file sits in is purely organizational (`configs/pi05droid/foo.yaml`
and `configs/pi05base/foo.yaml` are two different configs only because their
`name:` differs) -- names, not paths, are the identity, and they must be unique.

# Why YAML at all

The Python registry had grown to ~2k lines of near-duplicate `TrainConfig(...)`
literals that differed only in a repo id, a weights path and a couple of scalars.
Those are data, not code. Expressing them as data also lets the TPU launcher's
config builder read, diff and write configs without executing Python.

# The encoding

Every object in the config tree is a frozen dataclass, so the codec is generic:
a dataclass becomes a mapping of its fields, and a field whose declared type is
polymorphic (`model`, `data`, `weight_loader`, `optimizer`, ...) carries a `type:`
tag naming its concrete class in the registry below.

    model:
      type: pi0                 # -> Pi0Config
      pi05: true
      action_dim: 32
    weight_loader:
      type: checkpoint          # -> CheckpointWeightLoader
      params_path: gs://openpi-assets/checkpoints/pi05_base/params

Only non-default fields need to be written; anything omitted keeps the dataclass
default. `scripts/export_configs_to_yaml.py` emits exactly this form, and
`tests/config_yaml_test.py` asserts every exported config rebuilds a TrainConfig
equal to the Python original it replaced.

# Two things that are not plain data

`freeze_filter` holds an `nnx` filter tree built by `<ModelConfig>.get_freeze_filter()`.
It is always derived from the model, so it is written as `freeze_filter: from_model`
(or a full `{from_model: {<model spec>}}` when it must be derived from a *different*
model spec than the config's own).

`SimpleDataConfig.data_transforms` / `model_transforms` are callables of the model
config -- in Python they were lambdas closing over `model`. In YAML they are literal
transform groups whose arguments may interpolate the model config:

    data_transforms:
      inputs:
        - {type: droid_inputs, action_dim: "${model.action_dim}", model_type: "${model.model_type}"}
      outputs:
        - {type: droid_outputs}

`${model.<attr>}` is resolved when the data pipeline is created and the model config
is in hand, which is precisely the lambda's semantics.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import enum
import functools
import logging
import os
import pathlib
import types
import typing
from typing import Any, Literal

import flax.nnx as nnx
import tyro
import yaml

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

# The one place that defines where YAML configs live. Everything else derives from it.
CONFIG_DIR_NAME = "configs"
CONFIG_DIR_ENV = "OPENPI_CONFIG_DIR"


def repo_root() -> pathlib.Path:
    """The openpi checkout root (this file is at <root>/src/openpi/training/config_yaml.py)."""
    return pathlib.Path(__file__).resolve().parents[3]


def config_dir() -> pathlib.Path:
    """Where the YAML configs live.

    Resolution order, so this works in every shape openpi actually runs in:

    1. `$OPENPI_CONFIG_DIR` -- the explicit escape hatch (and how to point a job at a config tree
       that is not inside the checkout).
    2. `<cwd>/configs` -- the TPU worker untars the repo and runs `uv run scripts/train.py` from its
       root, and local runs are from the root too.
    3. `<repo root inferred from this file>/configs` -- correct for an editable/`uv sync` install,
       which is what openpi uses. It is the only one of the three that would silently come up empty
       if openpi were ever installed non-editable into site-packages, hence the two fallbacks above.
    """
    if env := os.environ.get(CONFIG_DIR_ENV):
        return pathlib.Path(env).expanduser().resolve()
    cwd_configs = pathlib.Path.cwd() / CONFIG_DIR_NAME
    if cwd_configs.is_dir():
        return cwd_configs
    return repo_root() / CONFIG_DIR_NAME


# --------------------------------------------------------------------------------------
# Registry: short YAML `type:` names -> concrete classes.
#
# Keep these names stable; they are a user-facing API (they appear in every YAML file
# and in the launcher's config builder).
# --------------------------------------------------------------------------------------


def _data_config_classes() -> dict[str, type]:
    # Deferred: openpi.training.config imports *this* module, so it cannot be imported at
    # module scope. By the time any of these functions run, config.py's module body has
    # finished and the classes exist.
    import openpi.training.config as _config

    return {
        "fake": _config.FakeDataConfig,
        "simple": _config.SimpleDataConfig,
        "lerobot_aloha": _config.LeRobotAlohaDataConfig,
        "lerobot_libero": _config.LeRobotLiberoDataConfig,
        "lerobot_droid": _config.LeRobotDROIDDataConfig,
        "rlds_droid": _config.RLDSDroidDataConfig,
    }


def _group_factory_classes() -> dict[str, type]:
    import openpi.training.config as _config

    return {"model_transform_factory": _config.ModelTransformFactory}


MODEL_TYPES: dict[str, type] = {
    "pi0": pi0_config.Pi0Config,
    "pi0_fast": pi0_fast.Pi0FASTConfig,
}

WEIGHT_LOADER_TYPES: dict[str, type] = {
    "none": weight_loaders.NoOpWeightLoader,
    "checkpoint": weight_loaders.CheckpointWeightLoader,
    "paligemma": weight_loaders.PaliGemmaWeightLoader,
}

LR_SCHEDULE_TYPES: dict[str, type] = {
    "cosine_decay": _optimizer.CosineDecaySchedule,
    "rsqrt_decay": _optimizer.RsqrtDecaySchedule,
}

OPTIMIZER_TYPES: dict[str, type] = {
    "adamw": _optimizer.AdamW,
    "sgd": _optimizer.SGD,
}

# Data/model transforms usable inside a transform group.
TRANSFORM_TYPES: dict[str, type] = {
    "repack": _transforms.RepackTransform,
    "inject_default_prompt": _transforms.InjectDefaultPrompt,
    "resize_images": _transforms.ResizeImages,
    "subsample_actions": _transforms.SubsampleActions,
    "delta_actions": _transforms.DeltaActions,
    "absolute_actions": _transforms.AbsoluteActions,
    "prompt_from_lerobot_task": _transforms.PromptFromLeRobotTask,
    "pad_states_and_actions": _transforms.PadStatesAndActions,
    "tokenize_prompt": _transforms.TokenizePrompt,
    "tokenize_fast_inputs": _transforms.TokenizeFASTInputs,
    "extract_fast_actions": _transforms.ExtractFASTActions,
    "droid_inputs": droid_policy.DroidInputs,
    "droid_outputs": droid_policy.DroidOutputs,
    "aloha_inputs": aloha_policy.AlohaInputs,
    "aloha_outputs": aloha_policy.AlohaOutputs,
    "libero_inputs": libero_policy.LiberoInputs,
    "libero_outputs": libero_policy.LiberoOutputs,
}

# Classes referenced *as values* (not instantiated by the codec), e.g.
# Pi0FASTConfig(fast_model_tokenizer=BinningTokenizer).
CLASS_REFS: dict[str, type] = {
    "paligemma_tokenizer": _tokenizer.PaligemmaTokenizer,
    "fast_tokenizer": _tokenizer.FASTTokenizer,
    "binning_tokenizer": _tokenizer.BinningTokenizer,
    "fsq_tokenizer": _tokenizer.FSQTokenizer,
}

ENUM_TYPES: dict[str, type[enum.Enum]] = {
    "ModelType": _model.ModelType,
    "DroidActionSpace": droid_rlds_dataset.DroidActionSpace,
}


def _registry_for(cls: type) -> dict[str, type] | None:
    """The registry that owns `cls`'s concrete implementations, if it is a tagged field."""
    import openpi.training.config as _config

    if cls is _model.BaseModelConfig:
        return MODEL_TYPES
    if cls is weight_loaders.WeightLoader:
        return WEIGHT_LOADER_TYPES
    if cls is _optimizer.LRScheduleConfig:
        return LR_SCHEDULE_TYPES
    if cls is _optimizer.OptimizerConfig:
        return OPTIMIZER_TYPES
    if cls is _config.DataConfigFactory:
        return _data_config_classes()
    if cls is _transforms.DataTransformFn:
        return TRANSFORM_TYPES
    return None


def _tag_of(obj: Any, registry: Mapping[str, type]) -> str:
    for tag, cls in registry.items():
        if type(obj) is cls:
            return tag
    raise ValueError(
        f"{type(obj).__module__}.{type(obj).__qualname__} is not in the YAML registry. "
        f"Add it to config_yaml.py (known: {sorted(registry)})."
    )


class ConfigError(ValueError):
    """A YAML config could not be parsed or built. Carries the offending file."""

    def __init__(self, message: str, path: pathlib.Path | None = None):
        self.path = path
        super().__init__(f"{path}: {message}" if path else message)


# --------------------------------------------------------------------------------------
# `${model.<attr>}` interpolation
# --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _ModelRef:
    """A deferred read of one attribute off the model config (`${model.action_dim}`)."""

    attr: str

    def resolve(self, model_config: _model.BaseModelConfig) -> Any:
        try:
            return getattr(model_config, self.attr)
        except AttributeError as e:
            raise ConfigError(
                f"${{model.{self.attr}}} is not a field of {type(model_config).__name__}"
            ) from e


def _parse_interpolation(value: str) -> _ModelRef | str:
    if value.startswith("${model.") and value.endswith("}"):
        return _ModelRef(value[len("${model.") : -1])
    return value


def _resolve_refs(value: Any, model_config: _model.BaseModelConfig) -> Any:
    """Replace every _ModelRef in an already-built object tree with its concrete value."""
    if isinstance(value, _ModelRef):
        return value.resolve(model_config)
    if isinstance(value, list):
        return [_resolve_refs(v, model_config) for v in value]
    if isinstance(value, tuple):
        return tuple(_resolve_refs(v, model_config) for v in value)
    if isinstance(value, dict):
        return {k: _resolve_refs(v, model_config) for k, v in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        changed = {
            f.name: _resolve_refs(getattr(value, f.name), model_config)
            for f in dataclasses.fields(value)
            if f.init
        }
        return type(value)(**changed)
    return value


@dataclasses.dataclass(frozen=True)
class YamlGroupFactory:
    """A `GroupFactory` built from a literal YAML transform group.

    Stands in for the `lambda model: Group(...)` closures the Python configs used: the
    transforms are constructed up-front with `_ModelRef` placeholders, and the placeholders
    are resolved against the real model config when the data pipeline is created.
    """

    group: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        return _resolve_refs(self.group, model_config)


# --------------------------------------------------------------------------------------
# Decoding: YAML value -> Python object, guided by the declared field type
# --------------------------------------------------------------------------------------


def _unwrap(hint: Any) -> Any:
    """Strip tyro/typing wrappers down to the useful runtime type."""
    # tyro.conf.Suppress[X] and friends are Annotated[X, ...].
    while typing.get_origin(hint) is typing.Annotated:
        hint = typing.get_args(hint)[0]
    return hint


def _resolve_union(value: Any, hint: Any) -> Any:
    """Pick the union member that fits `value`'s shape.

    Taking the first member would be wrong for a genuine multi-type union: `repo_id` is declared
    `str | Sequence[str]`, so "first member" turns a list of repo ids into the *string*
    "['lerobot/droid_1.0.1', ...]" -- a mixture silently collapsed into one nonsense repo name.
    Choose by what the value actually is.
    """
    origin = typing.get_origin(hint)
    if origin not in (typing.Union, types.UnionType):
        return hint
    members = [m for m in typing.get_args(hint) if m is not type(None)]
    if not members:
        return hint

    for member in members:
        inner = _unwrap(member)
        kind = typing.get_origin(inner) or inner
        if isinstance(value, str) and kind is str:
            return member
        if isinstance(value, (list, tuple)) and kind in (list, tuple, Sequence, typing.Sequence):
            return member
        if isinstance(value, Mapping) and kind in (dict, Mapping, typing.Mapping):
            return member
        if isinstance(value, bool) and kind is bool:
            return member
        if isinstance(value, (int, float)) and not isinstance(value, bool) and kind in (int, float):
            return member
        if isinstance(value, Mapping) and isinstance(kind, type) and dataclasses.is_dataclass(kind):
            return member
    return members[0]


@functools.lru_cache(maxsize=None)
def _field_hints(cls: type) -> dict[str, Any]:
    """Resolved type hints for a dataclass.

    Falls back to the raw `field.type` when a hint cannot be evaluated (a few openpi
    annotations reference runtime-only aliases). Every field we actually decode has a
    resolvable hint; the fallback just keeps one bad annotation from poisoning the rest.
    """
    try:
        return typing.get_type_hints(cls, include_extras=True)
    except Exception:  # noqa: BLE001
        return {f.name: f.type for f in dataclasses.fields(cls)}


def _decode_value(value: Any, hint: Any, *, path: str) -> Any:
    """Build the Python object for one YAML `value` declared as `hint`."""
    hint = _unwrap(hint)

    if value is None:
        return None

    # Scalars that may carry a `${model.*}` interpolation.
    if isinstance(value, str):
        ref = _parse_interpolation(value)
        if isinstance(ref, _ModelRef):
            return ref

    base = _unwrap(_resolve_union(value, hint))
    origin = typing.get_origin(base)

    # Enums, by member name ("PI05", "JOINT_POSITION").
    if isinstance(base, type) and issubclass(base, enum.Enum):
        return _decode_enum(value, base, path=path)

    # Sequence[T] / list[T] / tuple[T, ...]
    if origin in (list, Sequence, typing.Sequence, tuple):
        args = typing.get_args(base)
        item_hint = args[0] if args else Any
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        items = [_decode_value(v, item_hint, path=f"{path}[{i}]") for i, v in enumerate(value)]
        return tuple(items) if origin is tuple else items

    # Mapping[K, V] / dict[K, V]
    if origin in (dict, Mapping, typing.Mapping):
        args = typing.get_args(base)
        val_hint = args[1] if len(args) == 2 else Any
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        return {k: _decode_value(v, val_hint, path=f"{path}.{k}") for k, v in value.items()}

    # A transform group (used by DataConfig.repack_transforms and friends).
    if base is _transforms.Group:
        return _decode_group(value, path=path)

    # A GroupFactory: either a registered factory class, or a literal group.
    import openpi.training.config as _config

    if base is _config.GroupFactory:
        return _decode_group_factory(value, path=path)

    # A polymorphic (tagged) field: `type:` names the concrete class.
    registry = _registry_for(base) if isinstance(base, type) else None
    if registry is not None:
        return _decode_tagged(value, registry, path=path)

    # A class reference held as a value (e.g. fast_model_tokenizer: binning_tokenizer).
    if isinstance(value, str) and value in CLASS_REFS and base in (Any, type, None):
        return CLASS_REFS[value]

    # A concrete nested dataclass whose type is fully determined by the field.
    if isinstance(base, type) and dataclasses.is_dataclass(base):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping for {base.__name__}, got {type(value).__name__}")
        return _build_dataclass(base, value, path=path)

    # `Any`-typed field carrying a registered class name (Pi0FASTConfig.fast_model_tokenizer).
    if isinstance(value, str) and value in CLASS_REFS:
        return CLASS_REFS[value]

    return _decode_scalar(value, base, path=path)


def _decode_scalar(value: Any, hint: Any, *, path: str) -> Any:
    """Coerce a scalar to its declared type, or say so.

    Nothing else checks these, and the failure is nasty: PyYAML follows YAML 1.1, where `1e-5` is
    NOT a float (it wants `1.0e-5`), so `peak_lr: 1e-5` arrives as the *string* '1e-5'. Untyped, it
    sails through import and dies deep inside the optimizer with "unsupported operand /: 'str' and
    'int'" -- after the TPU has been allocated. js-yaml, which the launcher uses, reads the same
    text as the number 1e-05, so the config preview would show a perfectly good float while training
    got a string. Coerce here so the two agree and a bad value fails in the browser instead.
    """
    if hint in (int, float, bool, str) and value is not None:
        if hint is bool:
            if isinstance(value, bool):
                return value
            raise ConfigError(f"{path}: expected true/false, got {value!r}")
        if hint is str:
            return value if isinstance(value, str) else str(value)
        if isinstance(value, bool):  # bool is an int subclass; `batch_size: true` is a mistake.
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        try:
            coerced = hint(value)
        except (TypeError, ValueError) as e:
            raise ConfigError(f"{path}: expected {hint.__name__}, got {value!r}") from e
        if hint is int and isinstance(value, float) and value != coerced:
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return coerced

    if typing.get_origin(hint) is Literal:
        allowed = typing.get_args(hint)
        if value not in allowed:
            raise ConfigError(f"{path}: expected one of {list(allowed)}, got {value!r}")

    return value


def _decode_enum(value: Any, cls: type[enum.Enum], *, path: str) -> enum.Enum:
    if isinstance(value, cls):
        return value
    try:
        return cls[str(value)]
    except KeyError as e:
        raise ConfigError(
            f"{path}: '{value}' is not a member of {cls.__name__} (expected one of {[m.name for m in cls]})"
        ) from e


def _decode_tagged(value: Any, registry: Mapping[str, type], *, path: str) -> Any:
    """`{type: <tag>, ...fields}` -> an instance of the registered class."""
    if isinstance(value, str):
        # Shorthand for a no-argument implementation: `weight_loader: none`.
        value = {"type": value}
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: expected a mapping with a 'type' key, got {type(value).__name__}")
    body = dict(value)
    tag = body.pop("type", None)
    if tag is None:
        raise ConfigError(f"{path}: missing 'type' (one of {sorted(registry)})")
    cls = registry.get(tag)
    if cls is None:
        raise ConfigError(f"{path}: unknown type '{tag}' (expected one of {sorted(registry)})")
    return _build_dataclass(cls, body, path=path)


def _decode_group(value: Any, *, path: str) -> _transforms.Group:
    """`{inputs: [...], outputs: [...]}` -> transforms.Group."""
    if isinstance(value, list):  # bare list == inputs only
        value = {"inputs": value}
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: expected a transform group mapping, got {type(value).__name__}")
    unknown = set(value) - {"inputs", "outputs"}
    if unknown:
        raise ConfigError(f"{path}: unknown group keys {sorted(unknown)} (expected 'inputs'/'outputs')")
    # Only pass the keys that are present, so an absent one keeps Group's own default.
    kwargs = {
        key: [
            _decode_tagged(v, TRANSFORM_TYPES, path=f"{path}.{key}[{i}]")
            for i, v in enumerate(value[key] or [])
        ]
        for key in ("inputs", "outputs")
        if key in value
    }
    return _transforms.Group(**kwargs)


def _decode_group_factory(value: Any, *, path: str) -> Any:
    """A GroupFactory field: a registered factory, or a literal group (a lambda's stand-in)."""
    factories = _group_factory_classes()
    if isinstance(value, dict) and value.get("type") in factories:
        return _decode_tagged(value, factories, path=path)
    return YamlGroupFactory(group=_decode_group(value, path=path))


def _build_dataclass(cls: type, body: Mapping[str, Any], *, path: str) -> Any:
    """Instantiate `cls` from a YAML mapping, decoding each field by its declared type."""
    if not dataclasses.is_dataclass(cls):
        raise ConfigError(f"{path}: {cls.__name__} is not a dataclass")
    hints = _field_hints(cls)
    known = {f.name for f in dataclasses.fields(cls) if f.init}
    unknown = set(body) - known
    if unknown:
        raise ConfigError(
            f"{path}: unknown field(s) {sorted(unknown)} for {cls.__name__} (valid: {sorted(known)})"
        )
    kwargs = {
        name: _decode_value(value, hints.get(name, Any), path=f"{path}.{name}")
        for name, value in body.items()
    }
    try:
        return cls(**kwargs)
    except ConfigError:
        raise
    except Exception as e:  # noqa: BLE001 - surface the field path, not a bare TypeError.
        raise ConfigError(f"{path}: could not construct {cls.__name__}: {e}") from e


# --------------------------------------------------------------------------------------
# Building a TrainConfig
# --------------------------------------------------------------------------------------

# Handled explicitly rather than by the generic field walk.
_SPECIAL_KEYS = {"extends", "freeze_filter"}
# Not TrainConfig fields; free-form annotation for humans and the config builder.
_METADATA_KEYS = {"description", "tags"}


def _decode_freeze_filter(value: Any, model: Any, *, path: str) -> Any:
    """`from_model` (use the config's own model) or `{from_model: {<model spec>}}`."""
    if value in (None, False):
        return nnx.Nothing()
    if value == "from_model" or value is True:
        if model is None:
            raise ConfigError(f"{path}: 'from_model' needs a `model:` in this config (or its base)")
        return model.get_freeze_filter()
    if isinstance(value, dict) and set(value) == {"from_model"}:
        spec = _decode_tagged(value["from_model"], MODEL_TYPES, path=f"{path}.from_model")
        return spec.get_freeze_filter()
    raise ConfigError(
        f"{path}: expected 'from_model', or {{from_model: {{type: pi0, ...}}}}, got {value!r}"
    )


def build_config(
    body: Mapping[str, Any],
    *,
    name: str,
    path: pathlib.Path | None = None,
    overlay: Mapping[str, Any] | None = None,
) -> Any:
    """Build a TrainConfig from one parsed YAML document.

    `overlay` lets `extends:` see configs that have been built but not registered globally
    (the migration test builds a whole tree in isolation).
    """
    import openpi.training.config as _config

    if not isinstance(body, Mapping):
        raise ConfigError(f"expected a mapping at the top level, got {type(body).__name__}", path)

    body = {k: v for k, v in body.items() if k not in _METADATA_KEYS}
    hints = _field_hints(_config.TrainConfig)
    fields = {f.name for f in dataclasses.fields(_config.TrainConfig) if f.init}

    unknown = set(body) - fields - _SPECIAL_KEYS
    if unknown:
        raise ConfigError(
            f"unknown field(s) {sorted(unknown)} (valid: {sorted(fields | _SPECIAL_KEYS | _METADATA_KEYS)})",
            path,
        )

    base = None
    if (extends := body.get("extends")) is not None:
        base = _resolve_base(extends, path=path, overlay=overlay)

    kwargs: dict[str, Any] = {}
    for key, value in body.items():
        if key in _SPECIAL_KEYS:
            continue
        kwargs[key] = _decode_value(value, hints.get(key, Any), path=f"{name}.{key}")

    # freeze_filter is derived from the *effective* model, so it is resolved after the rest.
    model = kwargs.get("model") or (base.model if base is not None else None)
    if "freeze_filter" in body:
        kwargs["freeze_filter"] = _decode_freeze_filter(
            body["freeze_filter"], model, path=f"{name}.freeze_filter"
        )
    elif base is not None and "model" in kwargs:
        # The child changed the model but said nothing about freeze_filter. The base's filter is an
        # already-built nnx tree derived from the BASE's model, so inheriting it verbatim would
        # freeze parameters that no longer exist (extend a LoRA config with a full model and the
        # whole LLM stays frozen -- a "full finetune" that trains almost nothing, with no error, and
        # only bad loss curves on the TPU to show for it). If the base's filter was derived from the
        # base's own model, re-derive it from the new one; that is what `from_model` means.
        if _same(base.freeze_filter, base.model.get_freeze_filter()):
            kwargs["freeze_filter"] = model.get_freeze_filter()

    kwargs["name"] = name

    try:
        if base is not None:
            # `extends` means "this config, with these fields replaced" -- a shallow replace on
            # TrainConfig. Overriding `data:` or `model:` replaces the whole sub-object rather
            # than merging into it, which keeps the semantics obvious: what you see in the file
            # is what changed, and a partial `data:` can't silently inherit a stale repo_id.
            return dataclasses.replace(base, **kwargs)
        return _config.TrainConfig(**kwargs)
    except ConfigError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ConfigError(f"could not build TrainConfig: {e}", path) from e


def _resolve_base(name: str, *, path: pathlib.Path | None, overlay: Mapping[str, Any] | None = None) -> Any:
    import openpi.training.config as _config

    if not isinstance(name, str):
        raise ConfigError(f"`extends` must be a config name, got {type(name).__name__}", path)
    if overlay and name in overlay:
        return overlay[name]
    try:
        return _config.get_config(name)
    except ValueError as e:
        raise ConfigError(f"`extends: {name}` -- {e}", path) from e


def name_for(path: pathlib.Path) -> str:
    """A config's name is its file stem; the directory it sits in only groups."""
    return path.stem


# --------------------------------------------------------------------------------------
# Encoding: Python object -> YAML-ready plain data
#
# The inverse of the decoder above, used by scripts/export_configs_to_yaml.py to migrate the
# Python registry mechanically instead of by hand. Only fields that differ from their
# dataclass default are emitted, so the files stay readable.
# --------------------------------------------------------------------------------------


class NotEncodable(ConfigError):
    """An object has no YAML representation (e.g. a lambda). The exporter reports these."""


def _default_for(field: dataclasses.Field) -> Any:
    """A field's default value, or MISSING if it has none / cannot be built.

    `SimpleDataConfig.data_transforms` defaults to `GroupFactory`, a Protocol, which cannot be
    instantiated. There is nothing to compare against, so treat it as having no default and
    always emit the field.
    """
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        try:
            return field.default_factory()  # type: ignore[misc]
        except Exception:  # noqa: BLE001
            return dataclasses.MISSING
    return dataclasses.MISSING


def _same(a: Any, b: Any) -> bool:
    """Structural equality, tolerant of the two ways openpi spells the same value.

    `Sequence` fields are written as a tuple in some configs and a list in others -- e.g.
    `Group.inputs` defaults to `()` but is always passed a `[...]` -- and `[x] != (x,)`. Both
    are only ever iterated, so the container type carries no meaning; comparing it would pin
    an accident of how the Python literal happened to be typed. Compare elementwise instead.

    Some values (nnx filter trees) have no usable `__eq__`; fall back to their repr.
    """
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if dataclasses.is_dataclass(a) and dataclasses.is_dataclass(b) and type(a) is type(b):
        return all(
            _same(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a) if f.init
        )
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001
        return repr(a) == repr(b)


def configs_equal(a: Any, b: Any) -> bool:
    """Whether two TrainConfigs mean the same thing. The migration test's oracle.

    Everything compares structurally except the transform factories, which are compared by
    *behaviour*: a Python `lambda model: Group(...)` can never be `==` a `YamlGroupFactory`,
    but the two are equivalent exactly when they build the same Group for the same model. They
    are therefore applied to the config's own model and to probe models that differ in one
    field each, so a factory that merely happens to agree on this one model is not mistaken
    for a faithful port.
    """
    models = [a.model, *_probe_models(a.model)] if a.model is not None else []
    return _same_behavioural(a, b, models)


def _is_factory(value: Any) -> bool:
    """A callable object standing in for a transform group (a lambda, or a GroupFactory)."""
    return callable(value) and not isinstance(value, type) and not isinstance(value, _transforms.Group)


def _same_behavioural(a: Any, b: Any, models: Sequence[Any]) -> bool:
    if _is_factory(a) and _is_factory(b) and type(a) is not type(b):
        for model in models:
            try:
                want, got = a(model), b(model)
            except Exception:  # noqa: BLE001 - both must fail the same way to be equivalent.
                try:
                    b(model)
                except Exception:  # noqa: BLE001
                    continue
                return False
            if not _same(want, got):
                return False
        return True
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(
            _same_behavioural(x, y, models) for x, y in zip(a, b, strict=True)
        )
    if dataclasses.is_dataclass(a) and dataclasses.is_dataclass(b) and type(a) is type(b):
        return all(
            _same_behavioural(getattr(a, f.name), getattr(b, f.name), models)
            for f in dataclasses.fields(a)
            if f.init
        )
    return _same(a, b)


def _encode_value(value: Any, *, path: str, model: Any = None) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if value is tyro.MISSING:
        return None
    if isinstance(value, _ModelRef):
        return f"${{model.{value.attr}}}"
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, type):
        for tag, cls in CLASS_REFS.items():
            if value is cls:
                return tag
        raise NotEncodable(f"{path}: class {value.__qualname__} is not in CLASS_REFS")
    if isinstance(value, (list, tuple)):
        return [_encode_value(v, path=f"{path}[{i}]", model=model) for i, v in enumerate(value)]
    if isinstance(value, Mapping):
        return {str(k): _encode_value(v, path=f"{path}.{k}", model=model) for k, v in value.items()}
    if isinstance(value, _transforms.Group):
        return _encode_group(value, path=path, model=model)
    if isinstance(value, YamlGroupFactory):
        return _encode_group(value.group, path=path, model=model)
    if dataclasses.is_dataclass(value):
        registry = _registry_for(type(value).__mro__[1]) if len(type(value).__mro__) > 1 else None
        if registry and type(value) in registry.values():
            return _encode_tagged(value, registry, path=path, model=model)
        return _encode_fields(value, path=path, model=model)
    if callable(value):
        return _encode_group_factory(value, path=path, model=model)
    raise NotEncodable(f"{path}: cannot encode {type(value).__name__}")


def _encode_group(group: _transforms.Group, *, path: str, model: Any = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("inputs", "outputs"):
        items = getattr(group, key)
        if items:
            out[key] = [
                _encode_tagged(v, TRANSFORM_TYPES, path=f"{path}.{key}[{i}]", model=model)
                for i, v in enumerate(items)
            ]
    return out


# Model fields a transform group could plausibly be built from. Perturbing each one tells us
# whether a `lambda model: Group(...)` actually reads it.
_PROBE_INT_FIELDS = ("action_dim", "action_horizon", "max_token_len")


def _probe_models(model: Any) -> list[Any]:
    """Variants of `model` that differ in one field each, used to detect what a factory reads."""
    probes = []
    for field in _PROBE_INT_FIELDS:
        current = getattr(model, field, None)
        if isinstance(current, int) and not isinstance(current, bool):
            try:
                probes.append(dataclasses.replace(model, **{field: current + 7}))
            except Exception:  # noqa: BLE001 - a field that cannot be replaced simply isn't probed.
                pass
    # `model_type` is derived, not settable; flip the flag that determines it instead.
    if isinstance(getattr(model, "pi05", None), bool):
        try:
            probes.append(dataclasses.replace(model, pi05=not model.pi05))
        except Exception:  # noqa: BLE001
            pass
    return probes


def _substitute_refs(actual: Any, probe: Any, model: Any, probe_model: Any) -> Any:
    """Rewrite leaves that changed with the model into `${model.<attr>}`.

    `actual`/`probe` are the *encoded* groups a factory produced for two models that differ in
    exactly one field. A leaf that differs between them, and whose value equals that field's
    value on each respective model, is a read of that field -- which is what the lambda's
    closure over `model` was doing.
    """
    if isinstance(actual, dict) and isinstance(probe, dict):
        return {k: _substitute_refs(v, probe.get(k), model, probe_model) for k, v in actual.items()}
    if isinstance(actual, list) and isinstance(probe, list) and len(actual) == len(probe):
        return [_substitute_refs(a, p, model, probe_model) for a, p in zip(actual, probe, strict=True)]
    if _same(actual, probe):
        return actual
    for field in dataclasses.fields(type(model)):
        if not field.init:
            continue
        mine = getattr(model, field.name, None)
        theirs = getattr(probe_model, field.name, None)
        if _same(actual, _encode_scalar(mine)) and _same(probe, _encode_scalar(theirs)):
            return f"${{model.{field.name}}}"
    return actual


def _encode_scalar(value: Any) -> Any:
    return value.name if isinstance(value, enum.Enum) else value


def _encode_group_factory(factory: Any, *, path: str, model: Any) -> Any:
    """Encode a `GroupFactory` -- including the bare `lambda model: Group(...)` closures.

    A lambda is opaque, so it is treated as a black box: call it with the config's model and
    with probe models that differ in one field each. Leaves that move with a field become
    `${model.<field>}`; leaves that don't are constants. The result is then checked to
    reproduce the factory's output on *every* probed model, so a lambda whose dependence on
    the model this cannot express is reported rather than silently frozen at one value.
    """
    factories = _group_factory_classes()
    if type(factory) in factories.values():
        return _encode_tagged(factory, factories, path=path, model=model)

    if model is None:
        raise NotEncodable(f"{path}: cannot encode a transform factory without a model config")

    try:
        actual = factory(model)
    except Exception as e:  # noqa: BLE001
        raise NotEncodable(f"{path}: transform factory raised when called: {e}") from e
    if not isinstance(actual, _transforms.Group):
        raise NotEncodable(f"{path}: transform factory returned {type(actual).__name__}, not a Group")

    encoded = _encode_group(actual, path=path)
    for probe_model in _probe_models(model):
        try:
            probe = _encode_group(factory(probe_model), path=path)
        except Exception:  # noqa: BLE001 - a probe the factory rejects tells us nothing; skip it.
            continue
        encoded = _substitute_refs(encoded, probe, model, probe_model)

    # Behavioural check: the encoding must reproduce the factory on the real model *and* on
    # every probe. Freezing a model-dependent value would otherwise pass silently.
    rebuilt = _decode_group_factory(encoded, path=path)
    for candidate in (model, *_probe_models(model)):
        try:
            expected = factory(candidate)
        except Exception:  # noqa: BLE001
            continue
        if not _same(rebuilt(candidate), expected):
            raise NotEncodable(
                f"{path}: this transform factory depends on the model config in a way YAML cannot "
                f"express (differs for {type(candidate).__name__}"
                f"(action_dim={getattr(candidate, 'action_dim', '?')}, "
                f"action_horizon={getattr(candidate, 'action_horizon', '?')})). "
                f"Write it as a literal group with ${{model.*}} interpolation."
            )
    return encoded


def _encode_tagged(value: Any, registry: Mapping[str, type], *, path: str, model: Any = None) -> dict[str, Any]:
    return {"type": _tag_of(value, registry), **_encode_fields(value, path=path, model=model)}


def _encode_fields(value: Any, *, path: str, model: Any = None) -> dict[str, Any]:
    """A dataclass's non-default fields."""
    out: dict[str, Any] = {}
    for field in dataclasses.fields(value):
        if not field.init:
            continue
        current = getattr(value, field.name)
        default = _default_for(field)
        if default is not dataclasses.MISSING and _same(current, default):
            continue
        if current is tyro.MISSING:
            continue
        out[field.name] = _encode_value(current, path=f"{path}.{field.name}", model=model)
    return out


def to_yaml_dict(config: Any) -> dict[str, Any]:
    """A TrainConfig as plain data, ready for `yaml.safe_dump`.

    `name` is omitted: the file's stem is the config's name (see `name_for`), so writing it
    twice would let the two drift apart.
    """
    import openpi.training.config as _config

    body: dict[str, Any] = {}
    for field in dataclasses.fields(_config.TrainConfig):
        if not field.init or field.name == "name":
            continue
        current = getattr(config, field.name)
        default = _default_for(field)

        if field.name == "freeze_filter":
            if _same(current, nnx.Nothing()):
                continue
            model = config.model
            if model is not None and _same(current, model.get_freeze_filter()):
                body["freeze_filter"] = "from_model"
                continue
            raise NotEncodable(
                f"{config.name}.freeze_filter: not derivable from this config's model; "
                f"write it as {{from_model: {{type: ..., ...}}}} by hand."
            )

        if default is not dataclasses.MISSING and _same(current, default):
            continue
        if current is tyro.MISSING:
            continue
        # `model` is passed down so a transform factory can be probed against it.
        body[field.name] = _encode_value(current, path=f"{config.name}.{field.name}", model=config.model)
    return body


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------


def iter_config_files(root: pathlib.Path | None = None) -> list[pathlib.Path]:
    """Every config file under `root`. A leading `_` marks a file as not-a-config."""
    root = root or config_dir()
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.y*ml") if p.is_file() and not p.name.startswith("_"))


def read_file(path: pathlib.Path) -> dict[str, Any]:
    """The raw parsed YAML document (no TrainConfig built). Used by the launcher's API."""
    try:
        text = path.read_text()
    except OSError as e:
        raise ConfigError(f"could not read: {e}", path) from e
    try:
        body = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML: {e}", path) from e
    if body is None:
        raise ConfigError("file is empty", path)
    if not isinstance(body, Mapping):
        raise ConfigError(f"expected a mapping at the top level, got {type(body).__name__}", path)
    return dict(body)


def load_file(path: pathlib.Path, *, overlay: Mapping[str, Any] | None = None) -> Any:
    body = read_file(path)
    name = str(body.get("name") or name_for(path))
    return build_config(body, name=name, path=path, overlay=overlay)


# Configs that failed to load, name -> the error. `get_config` re-raises it for that name.
BROKEN: dict[str, ConfigError] = {}


def load_configs(
    root: pathlib.Path | None = None, *, register: bool = False, strict: bool = False
) -> list[Any]:
    """Build every YAML config under `root`, bases before the configs that extend them.

    `extends:` resolves against configs already built, so load order matters. Rather than
    impose one on the user, defer the files whose base is not available yet and retry until
    a pass makes no progress -- a small topological sort that surfaces a real cycle (or a
    genuinely missing base) as the underlying error instead of as a hang.

    With `register=True` the configs are added to the global registry as they are built, so
    a YAML config may extend any other config by name. With `register=False` they are built
    in isolation against a local overlay, which is what the migration test needs (it loads a
    whole config tree without touching the live registry).

    A file that fails to build does NOT abort the load: this runs at `import openpi.training.config`
    time, so raising would mean one typo in one config file bricks the entire package -- training a
    *different* config, serving a policy, even collecting the test suite. Broken files are recorded
    in `BROKEN` and logged; `get_config(name)` then re-raises that config's own error, so you learn
    exactly what is wrong precisely when you ask for it. Pass `strict=True` (the tests do) to fail
    the load instead.
    """
    import openpi.training.config as _config

    pending = iter_config_files(root)
    built: list[Any] = []
    overlay: dict[str, Any] = {}
    broken: dict[str, ConfigError] = {}

    def fail(path: pathlib.Path, err: ConfigError) -> None:
        if strict:
            raise err
        broken[name_for(path)] = err

    while pending:
        deferred: list[pathlib.Path] = []
        errors: dict[pathlib.Path, ConfigError] = {}
        progressed = False
        for path in pending:
            try:
                config = load_file(path, overlay=None if register else overlay)
            except ConfigError as e:
                # An unresolved `extends` may just be an ordering problem -- retry it next pass.
                # Anything else is a genuine error in this file, and only this file.
                if "`extends:" in str(e):
                    deferred.append(path)
                    errors[path] = e
                    continue
                fail(path, e)
                progressed = True
                continue
            try:
                if register:
                    _config.register_config(config)
                elif config.name in overlay:
                    raise ConfigError(f"duplicate config name '{config.name}'", path)
                else:
                    overlay[config.name] = config
            except ValueError as e:  # a duplicate name from register_config
                fail(path, ConfigError(str(e), path))
                progressed = True
                continue
            built.append(config)
            progressed = True
        if deferred and not progressed:
            # No pass can make progress, so every remaining file has an unresolvable base. That is
            # either a genuinely missing config or a cycle among the stragglers -- and the raw error
            # for a cycle would be a nonsense "'a' not found. Did you mean...?" naming a config that
            # plainly exists. Name the cycle instead.
            stuck = {name_for(p) for p in deferred}
            for path in deferred:
                base = read_file(path).get("extends")
                if base in stuck:
                    fail(
                        path,
                        ConfigError(
                            f"`extends: {base}` is part of a cycle among {sorted(stuck)} -- "
                            f"no config in it can be built.",
                            path,
                        ),
                    )
                else:
                    fail(path, errors[path])
            break
        pending = deferred

    if register:
        BROKEN.clear()
        BROKEN.update(broken)
    if broken:
        logging.warning(
            "%d config file(s) failed to load and are unavailable: %s. "
            "Selecting one raises its error; the rest are unaffected.",
            len(broken),
            ", ".join(sorted(broken)),
        )
    return built
