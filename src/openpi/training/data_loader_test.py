import dataclasses
import pathlib

import jax
import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def _base_data_config(**factory_kwargs):
    """Resolve a DataConfig via the factory chokepoint (create_base_config) without touching the Hub.

    create_base_config is where max_episodes is validated for BOTH the map-style and streaming loaders;
    norm-stat loading from the (non-existent) local assets dir falls back to None.
    """
    model = pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=16)
    factory = _config.LeRobotDROIDDataConfig(repo_id="user/toys300", **factory_kwargs)
    return factory.create_base_config(pathlib.Path("./assets/does-not-exist"), model)


def test_max_episodes_valid_threads_through():
    data_config = _base_data_config(max_episodes={"user/toys300": 20})
    assert data_config.max_episodes == {"user/toys300": 20}


def test_max_episodes_default_is_empty():
    assert _base_data_config().max_episodes == {}


def test_max_episodes_extra_repo_raises():
    # A typo'd cap key that matches no repo must fail fast for BOTH loaders (was silently ignored by the
    # map-style path). Streaming raised already; this covers the shared config-creation chokepoint.
    with pytest.raises(ValueError, match="not in repo_id"):
        _base_data_config(max_episodes={"user/toys_300": 20})


def test_max_episodes_non_positive_raises():
    with pytest.raises(ValueError, match="positive"):
        _base_data_config(max_episodes={"user/toys300": 0})


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
