"""Export the Python `TrainConfig` registry to YAML files.

This is how the Python configs were migrated to `configs/**.yaml`: mechanically, by
introspecting the live objects, rather than by retyping ~2k lines of dataclass literals by
hand. It stays in the tree because it is also the way to re-export after touching the schema,
and because `tests/config_yaml_test.py` runs it to prove the YAML tree still rebuilds configs
equal to their Python originals.

    # See what would be written, and which configs cannot be expressed yet.
    uv run scripts/export_configs_to_yaml.py --dry-run

    # Write them (default: <repo>/configs, grouped into subdirectories by name prefix).
    uv run scripts/export_configs_to_yaml.py --out configs

Each file is named after its config, so the file stem *is* the config name. A config the
encoder cannot express (e.g. a `data_transforms` lambda) is reported and skipped, never
written half-right -- those few are hand-written in YAML and the equivalence test holds them
to the same standard as the exported ones.
"""

import argparse
import pathlib
import sys
from typing import Any

import yaml

import openpi.training.config as _config
import openpi.training.config_yaml as _config_yaml

# Configs are grouped into subdirectories purely so the tree is navigable. The prefix match is
# ordered: the first matching entry wins.
_GROUPS: list[tuple[str, str]] = [
    ("pi05base-", "pi05base"),
    ("pi05droid-", "pi05droid"),
    ("pi05polaris-", "pi05polaris"),
    ("debug", "debug"),
]


def group_for(name: str) -> str:
    for prefix, group in _GROUPS:
        if name.startswith(prefix):
            return group
    if "polaris" in name:
        return "pi05polaris"
    if "aloha" in name:
        return "aloha"
    if "libero" in name:
        return "libero"
    if "droid" in name:
        return "droid"
    return "misc"


class _Dumper(yaml.SafeDumper):
    """Block-style YAML with stable key order and readable long strings."""

    def ignore_aliases(self, data: Any) -> bool:  # never emit &anchors/*refs
        return True


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    # Quote anything that could be read back as a non-string (a bare `${model.x}` is fine
    # unquoted in YAML, but quoting makes the interpolation visible as a value).
    style = '"' if data.startswith("${") else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_Dumper.add_representer(str, _represent_str)


def dump(body: dict[str, Any], *, name: str, description: str | None = None) -> str:
    header = f"# {name}\n"
    if description:
        header += f"# {description}\n"
    text = yaml.dump(
        body,
        Dumper=_Dumper,
        sort_keys=False,
        default_flow_style=False,
        width=100,
        indent=2,
        allow_unicode=True,
    )
    return header + text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=pathlib.Path, default=_config_yaml.config_dir(), help="output directory")
    parser.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    parser.add_argument("--only", nargs="*", help="export just these config names")
    args = parser.parse_args()

    configs = [c for c in _config._PYTHON_CONFIGS if not args.only or c.name in args.only]  # noqa: SLF001
    if not configs:
        print("No Python configs to export (already migrated?).", file=sys.stderr)
        return 0

    written: list[pathlib.Path] = []
    skipped: list[tuple[str, str]] = []

    for config in configs:
        try:
            body = _config_yaml.to_yaml_dict(config)
        except _config_yaml.NotEncodable as e:
            skipped.append((config.name, str(e)))
            continue

        path = args.out / group_for(config.name) / f"{config.name}.yaml"
        text = dump(body, name=config.name)

        # The exported file must rebuild the very config it came from. Check now, per file,
        # so a bad round-trip is caught here rather than as a mystery at train time.
        rebuilt = _config_yaml.build_config(yaml.safe_load(text), name=config.name, path=path)
        if not _config_yaml.configs_equal(rebuilt, config):
            skipped.append((config.name, "round-trip mismatch: the exported YAML does not rebuild this config"))
            continue

        if not args.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        written.append(path)

    root = args.out
    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} {len(written)} config(s) under {root}")
    for path in written:
        print(f"  {path.relative_to(root.parent) if root.parent in path.parents else path}")

    if skipped:
        print(f"\n{len(skipped)} config(s) NOT exported -- hand-write these in YAML:", file=sys.stderr)
        for name, reason in skipped:
            print(f"  {name}: {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
