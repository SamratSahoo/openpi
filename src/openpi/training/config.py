"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Mapping, Sequence
import dataclasses
import difflib
import logging
import os
import pathlib
from typing import Any, Literal, Protocol, TypeAlias
import urllib.parse

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


def is_remote_path(path: str) -> bool:
    """True for an object-storage URL (`gs://...`, `s3://...`), false for a local filesystem path."""
    return urllib.parse.urlparse(str(path)).scheme != ""


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created. May be a single repo id or a list of
    # repo ids. A list is treated as a single concatenated dataset: samples are drawn with
    # probability proportional to each repo's size ("sample as one dataset"), not equally per repo.
    repo_id: str | Sequence[str] | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()

    # If true, stream the LeRobot dataset(s) directly from the HuggingFace Hub during training
    # instead of downloading them to local disk. This avoids needing storage for the datasets and
    # is the recommended mode for multi-repo mixtures. Only supported for LeRobot datasets that
    # store camera frames inline in the parquet files (feature dtype "image"), which is the common
    # case for converted DROID/LeRobot datasets. Hub rate limits (HTTP 429) are handled by waiting
    # and retrying rather than crashing.
    streaming: bool = False
    # Size of the reservoir shuffle buffer used when streaming. Larger gives better shuffling but
    # uses more host memory (each buffered sample holds its decoded camera frames).
    streaming_shuffle_buffer_size: int = 1000

    # Maps a streamed repo_id to a JSON file of non-idle keep-ranges (episode_index -> [[start, end]])
    # produced by examples/droid/compute_droid_nonidle_ranges_streaming.py. ONLY repos listed here are
    # filtered (idle frames dropped); every other repo in the mixture is sampled in full. Used to
    # filter idle frames out of DROID (lerobot/droid_1.0.1) during training.
    nonidle_filter_paths: Mapping[str, str] = dataclasses.field(default_factory=dict)

    # If true, the streaming v3.0 loader builds `actions` from action.joint_position[7] + gripper[1]
    # (absolute commanded joint targets) instead of the default action.joint_velocity[7] + gripper[1].
    # Pair with the DeltaActions/AbsoluteActions transform in LeRobotDROIDDataConfig.create.
    joint_position_actions: bool = False

    # Explicit per-repo mixing weights (repo_id -> relative weight) for a streamed multi-repo mixture.
    # Empty (the default) => sample proportional to each repo's frame count ("sample as one dataset").
    # When set it must have a positive entry for EVERY repo in the mixture; the weights are relative
    # (normalized at sampling time), so e.g. {a: 0.5, b: 0.5} draws each repo equally regardless of size
    # (oversampling the smaller one). Only used by the streaming loader.
    sampling_weights: Mapping[str, float] = dataclasses.field(default_factory=dict)

    # Per-repo cap on how many episodes to use, keyed by repo_id (repo_id -> episode count). Only the
    # FIRST N episodes (by episode index) of a listed repo are used; repos absent from the mapping (the
    # default, an empty dict) use ALL of their episodes. A cap larger than a repo's episode count is
    # clamped to "use all". Example: {"user/toys300_sim": 20} trains on only the first 20 of 300
    # episodes. Honored by both the map-style (download) and streaming loaders.
    max_episodes: Mapping[str, int] = dataclasses.field(default_factory=dict)

    # Root of a MIRROR of the datasets, e.g. "gs://bucket/user/datasets". When set, the streaming
    # loader reads each repo from <mirror_root>/<repo_id>/ (gcsfs) instead of from the HuggingFace Hub;
    # the layout, the reader and the v3.0 support are identical, only the bytes' origin changes.
    #
    # This exists because the Hub's CDN is not usable from GCP: it routes GCP-hosted clients to an edge
    # that rejects its own signed URLs, killing TPU runs. Reading from a bucket in the training region
    # also drops the cross-cloud hop. Defaults to $OPENPI_DATA_MIRROR_ROOT, so a launcher can turn a
    # config into a mirrored run by setting one env var, with no config edit.
    #
    # The mirror must be COMPLETE before training starts (every file the repo has on the Hub, same
    # relative paths) -- a partial mirror surfaces as missing files mid-epoch, not as a clean error.
    mirror_root: str | None = dataclasses.field(default_factory=lambda: os.environ.get("OPENPI_DATA_MIRROR_ROOT"))


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


def repo_id_as_list(repo_id: str | Sequence[str] | None) -> list[str] | None:
    """Normalize a repo id (single or list) into a list, or None."""
    if repo_id is None or repo_id is tyro.MISSING:
        return None
    if isinstance(repo_id, str):
        return [repo_id]
    return list(repo_id)


def combined_asset_id(repo_id: str | Sequence[str] | None) -> str | None:
    """Derive a single, filesystem-safe asset id from a (possibly multi-repo) repo id.

    A single repo keeps its own id (so existing norm-stat assets keep working); a mixture is
    joined into one deterministic id so its norm stats live in a single assets directory.
    """
    repos = repo_id_as_list(repo_id)
    if not repos:
        return None
    if len(repos) == 1:
        return repos[0]
    return "+".join(r.replace("/", "_") for r in repos)


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id. May be a single repo id or a list of repo ids (mixture).
    repo_id: str | Sequence[str] = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None
    # If true, stream the dataset(s) from the Hub instead of downloading to disk. See DataConfig.
    streaming: bool = False
    # Size of the reservoir shuffle buffer used when streaming. See DataConfig.
    streaming_shuffle_buffer_size: int = 1000
    # Maps a streamed repo_id to a non-idle keep-ranges JSON. Only listed repos are filtered. See
    # DataConfig.nonidle_filter_paths.
    nonidle_filter_paths: Mapping[str, str] = dataclasses.field(default_factory=dict)
    # If true, stream action.joint_position (absolute joint targets) as the action. See
    # DataConfig.joint_position_actions. LeRobotDROIDDataConfig.create pairs this with a delta transform.
    joint_position_actions: bool = False
    # Explicit per-repo mixing weights for a streamed multi-repo mixture. Empty => size-proportional.
    # See DataConfig.sampling_weights.
    sampling_weights: Mapping[str, float] = dataclasses.field(default_factory=dict)
    # Per-repo cap on how many (first-N) episodes to use, keyed by repo_id. Empty => use all episodes of
    # every repo. See DataConfig.max_episodes.
    max_episodes: Mapping[str, int] = dataclasses.field(default_factory=dict)
    # Read the repos from a mirror (e.g. "gs://bucket/user/datasets") instead of the Hub. Defaults to
    # $OPENPI_DATA_MIRROR_ROOT. See DataConfig.mirror_root.
    mirror_root: str | None = dataclasses.field(default_factory=lambda: os.environ.get("OPENPI_DATA_MIRROR_ROOT"))

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        # Validate the episode cap once here -- the single chokepoint for both the map-style (download) and
        # streaming loaders -- so a typo'd repo id or non-positive count fails fast and identically for both
        # instead of the download path silently ignoring the entry and training on all episodes.
        if self.max_episodes:
            repos = set(repo_id_as_list(repo_id) or [])
            extra = sorted(k for k in self.max_episodes if k not in repos)
            if extra:
                raise ValueError(f"max_episodes references repos not in repo_id {sorted(repos)}: {extra}.")
            non_positive = {k: v for k, v in self.max_episodes.items() if int(v) <= 0}
            if non_positive:
                raise ValueError(f"max_episodes values must be positive integers; got {non_positive}.")
        asset_id = self.assets.asset_id or combined_asset_id(repo_id)
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
            streaming=self.streaming,
            streaming_shuffle_buffer_size=self.streaming_shuffle_buffer_size,
            nonidle_filter_paths=self.nonidle_filter_paths,
            joint_position_actions=self.joint_position_actions,
            sampling_weights=self.sampling_weights,
            max_episodes=self.max_episodes,
            mirror_root=self.mirror_root,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # By default we assume joint *velocity* actions, so we do *not* apply a delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        # Joint-position actions are ABSOLUTE joint targets -> convert to delta (relative to the current
        # joint state) for training; the gripper (last dim) stays absolute. Mirrors the RLDS
        # JOINT_POSITION path (RLDSDroidDataConfig.create). Norm stats must be in this same delta space.
        if self.joint_position_actions:
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints. May be a cloud URL (e.g. `gs://bucket/prefix`), in which case
    # orbax streams checkpoints straight to object storage and never stages them on local disk --
    # which is what makes a large model trainable on a VM whose disk cannot hold two checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> epath.Path:
        """Get the checkpoint directory for this config.

        `epath`, not `pathlib`: `checkpoint_base_dir` may be a cloud URL, and pathlib collapses the
        `//` in a scheme, silently turning `gs://bucket/x` into the *relative* path `gs:/bucket/x`.
        `resolve()` has the same effect on a remote path -- it rewrites it against the cwd -- so it
        is applied only to local paths, where it is still needed to absolutize the `./checkpoints`
        default.
        """
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        directory = epath.Path(self.checkpoint_base_dir) / self.name / self.exp_name
        return directory if is_remote_path(self.checkpoint_base_dir) else directory.resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Configs still defined in Python. These are being migrated to <repo>/configs/**.yaml; see
# config_yaml.py. Use `get_config` to look a config up by name -- it sees both registries.
# Every TrainConfig now lives in <repo>/configs/**.yaml -- see config_yaml.py, and
# scripts/export_configs_to_yaml.py for the mechanical migration that produced them. This list
# stays as the escape hatch for a config that genuinely needs Python (one whose transforms can't
# be expressed as a literal group); anything appended here is registered alongside the YAML ones.
_PYTHON_CONFIGS: list[TrainConfig] = []

_CONFIGS: list[TrainConfig] = []
_CONFIGS_DICT: dict[str, TrainConfig] = {}


def register_config(config: TrainConfig) -> None:
    """Add a config to the registry. Names are the configs' identity, so they must be unique."""
    if config.name in _CONFIGS_DICT:
        raise ValueError(f"Config names must be unique; '{config.name}' is defined more than once.")
    _CONFIGS.append(config)
    _CONFIGS_DICT[config.name] = config


for _python_config in _PYTHON_CONFIGS:
    register_config(_python_config)


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        # A config whose YAML failed to load is absent from the registry, but "not found -- did you
        # mean...?" would send you hunting for a typo in the name when the real problem is a typo in
        # the file. Re-raise what actually went wrong.
        if broken := _config_yaml.BROKEN.get(config_name):
            raise ValueError(f"Config '{config_name}' failed to load: {broken}") from broken

        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]


# The YAML registry (<repo>/configs/**.yaml). Imported down here, after every schema class and
# `register_config`/`get_config` exist, because config_yaml lazily imports this module back for
# them. A config file that fails to build is quarantined rather than raised (see load_configs), so
# one bad file cannot brick `import openpi.training.config` for everything else.
from openpi.training import config_yaml as _config_yaml  # noqa: E402  isort:skip

_config_yaml.load_configs(register=True)
