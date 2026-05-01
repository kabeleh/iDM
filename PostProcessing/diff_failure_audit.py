#!/usr/bin/env python3
"""Diff a failed audit .ini against the reconstructed YAML-derived CLASS config.

This helper reads one JSON record produced by posterior_background_heatmaps.py
failure auditing, rebuilds the CLASS parameter dictionary from the original
Cobaya best-fit + YAML inputs and the sampled chain row values, applies the
same background-only overrides as the heatmap pipeline, and compares that
expected configuration against the preserved failed .ini.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from BestFitPlot import extract_class_parameters, load_bestfit_file, load_input_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diff a preserved failure-audit .ini against the reconstructed "
            "YAML-derived CLASS configuration."
        )
    )
    parser.add_argument(
        "audit_json",
        type=str,
        help="Path to one failure audit JSON record.",
    )
    parser.add_argument(
        "--show-matching",
        action="store_true",
        help="Also print keys whose values match exactly.",
    )
    return parser.parse_args()


def _normalize_sbbn_path(class_params: dict[str, Any]) -> None:
    key = "sBBN file"
    if key not in class_params:
        return
    val = str(class_params[key]).strip()
    if not val:
        return
    if "/" not in val and "\\" not in val:
        class_params[key] = f"/external/bbn/{val}"
    elif val.startswith("bbn/"):
        class_params[key] = f"/external/{val}"


def _parse_ini_file(ini_path: Path) -> dict[str, str]:
    params: dict[str, str] = {}
    with open(ini_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            params[key.strip()] = value.strip()
    return params


def _chain_stem(chain_file: Path) -> Path:
    name = re.sub(r"\.\d+\.txt$", "", chain_file.name)
    name = re.sub(r"\.txt$", "", name)
    return chain_file.with_name(name)


def _build_expected_run_params(
    audit_payload: dict[str, Any],
) -> tuple[dict[str, str], Path, Path]:
    chain_file = Path(str(audit_payload["chain_file"])).resolve()
    stem = _chain_stem(chain_file)
    bestfit_path = stem.with_suffix(".bestfit")
    yaml_path = stem.with_suffix(".input.yaml")

    bestfit_values = load_bestfit_file(str(bestfit_path))
    yaml_config = load_input_yaml(str(yaml_path))

    sampled_parameter_values = {
        str(key): value
        for key, value in dict(audit_payload["sampled_parameter_values"]).items()
    }

    sample_bestfit = dict(bestfit_values)
    sample_bestfit.update(sampled_parameter_values)

    warnings: list[str] = []
    class_params = extract_class_parameters(
        sample_bestfit, yaml_config, warnings=warnings
    )

    for key, value in list(class_params.items()):
        if isinstance(value, bool):
            class_params[key] = "yes" if value else "no"

    _normalize_sbbn_path(class_params)

    actual_ini = _parse_ini_file(Path(str(audit_payload["generated_ini_file"])))

    run_params = dict(class_params)
    run_params["write_background"] = "yes"
    run_params["lensing"] = "no"
    run_params["overwrite_root"] = "yes"
    if "root" in actual_ini:
        run_params["root"] = actual_ini["root"]

    run_params.pop("output", None)
    run_params.pop("non linear", None)
    run_params.pop("non_linear", None)

    expected = {key: str(value) for key, value in sorted(run_params.items())}
    return expected, bestfit_path, yaml_path


def _diff_configs(
    expected: dict[str, str], actual: dict[str, str]
) -> tuple[list[str], list[str], list[tuple[str, str, str]], list[tuple[str, str]]]:
    missing = sorted(key for key in expected if key not in actual)
    extra = sorted(key for key in actual if key not in expected)
    changed: list[tuple[str, str, str]] = []
    matching: list[tuple[str, str]] = []

    for key in sorted(set(expected).intersection(actual)):
        if expected[key] == actual[key]:
            matching.append((key, expected[key]))
        else:
            changed.append((key, expected[key], actual[key]))

    return missing, extra, changed, matching


def main() -> None:
    args = parse_args()

    audit_json = Path(args.audit_json).resolve()
    with open(audit_json, "r", encoding="utf-8") as f:
        audit_payload = json.load(f)

    ini_path = Path(str(audit_payload["generated_ini_file"])).resolve()
    actual = _parse_ini_file(ini_path)
    expected, bestfit_path, yaml_path = _build_expected_run_params(audit_payload)

    missing, extra, changed, matching = _diff_configs(expected, actual)

    print("=" * 72)
    print("Failure Audit Diff")
    print("=" * 72)
    print(f"Audit JSON:     {audit_json}")
    print(f"Preserved INI:  {ini_path}")
    print(f"Best-fit file:  {bestfit_path}")
    print(f"Input YAML:     {yaml_path}")
    print(f"Chain file:     {audit_payload['chain_file']}")
    print(f"Post-burn row:  {audit_payload['row_index_postburn']}")
    print(f"Trajectory idx: {audit_payload['trajectory_index']}")
    print()
    print(
        f"Summary: {len(changed)} changed, {len(missing)} missing in ini, "
        f"{len(extra)} extra in ini, {len(matching)} matching"
    )

    if changed:
        print()
        print("Changed values")
        print("-" * 72)
        for key, expected_value, actual_value in changed:
            print(f"{key}")
            print(f"  expected: {expected_value}")
            print(f"  actual:   {actual_value}")

    if missing:
        print()
        print("Missing In INI")
        print("-" * 72)
        for key in missing:
            print(f"{key} = {expected[key]}")

    if extra:
        print()
        print("Extra In INI")
        print("-" * 72)
        for key in extra:
            print(f"{key} = {actual[key]}")

    if args.show_matching and matching:
        print()
        print("Matching Values")
        print("-" * 72)
        for key, value in matching:
            print(f"{key} = {value}")


if __name__ == "__main__":
    main()
