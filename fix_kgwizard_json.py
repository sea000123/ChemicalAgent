#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 0523 for gpt5.2
"""
fix_kgwizard_json.py

Fix DataRaider JSON files before running kgwizard.

This script fixes the common error:

    TypeError: string indices must be integers
    optimization_runs = list(react_dict["Optimization Runs"].keys())

Typical causes:
  1. The .json file was saved as a JSON string instead of a JSON object.
     Example: "{\"Optimization Runs Dictionary\": {...}}"
  2. DataRaider used the key "Optimization Runs Dictionary", while kgwizard
     expects "Optimization Runs".

Usage from MERMaid project root:

  # Safer: write fixed files to a new folder
  python fix_kgwizard_json.py ./Results/JSON --output_dir ./Results/JSON_fixed

  # Then run kgwizard on the fixed folder
  kgwizard transform ./Results/JSON_fixed --output_dir ./Results/KGIntermediate --schema photo --graph_name g --address ws://localhost --port 8182 --output_file "C:/Users/user/Documents/GitHub/MERMaid/Results/Graphs/g.graphml"

  # Or fix files in place, with automatic backup
  python fix_kgwizard_json.py ./Results/JSON --in-place
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


def strip_code_fence(text: str) -> str:
    """Remove ```json ... ``` wrappers if present."""
    text = text.strip()

    fence = re.fullmatch(r"```(?:json|JSON)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fence:
        return fence.group(1).strip()

    return text


def extract_first_json_object(text: str) -> str:
    """
    Extract the first balanced JSON object from text.
    Useful when a raw LLM response contains prose around JSON.
    """
    text = strip_code_fence(text)

    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object start '{' found.")

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError("Could not find a complete balanced JSON object.")


def json_loads_flexible(text: str) -> Any:
    """
    Parse JSON flexibly:
      - plain JSON object
      - JSON string containing JSON object
      - markdown fenced JSON
      - raw OpenAI-style response containing choices[0].message.content
    """
    text = strip_code_fence(text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = json.loads(extract_first_json_object(text))

    # Some APIs save the full chat completion instead of only message.content.
    if isinstance(data, dict) and "choices" in data:
        try:
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, str):
                return json_loads_flexible(content)
        except Exception:
            pass

    # Some gateway APIs save {"success": true, "result": "..."}.
    if isinstance(data, dict) and "result" in data and len(data) <= 3:
        result = data.get("result")
        if isinstance(result, str):
            return json_loads_flexible(result)
        if isinstance(result, dict):
            return result

    # Main fix: JSON file is actually a JSON string containing JSON.
    # Repeat a few times to handle double-encoding.
    for _ in range(5):
        if isinstance(data, str):
            s = strip_code_fence(data)
            try:
                data = json.loads(s)
                continue
            except json.JSONDecodeError:
                data = json.loads(extract_first_json_object(s))
                continue
        break

    return data


def normalize_for_kgwizard(data: Any) -> dict[str, Any]:
    """
    Normalize DataRaider output into the shape kgwizard expects.

    kgwizard traceback shows it expects:
      react_dict["Optimization Runs"].keys()
    """
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        data = data[0]

    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object/dict after parsing, got {type(data).__name__}")

    # Add canonical keys expected by kgwizard while preserving original keys.
    if "Optimization Runs" not in data:
        if "Optimization Runs Dictionary" in data:
            data["Optimization Runs"] = data["Optimization Runs Dictionary"]
        elif "optimization_runs" in data:
            data["Optimization Runs"] = data["optimization_runs"]
        elif "OptimizationRuns" in data:
            data["Optimization Runs"] = data["OptimizationRuns"]

    if "Footnotes" not in data:
        if "Footnotes Dictionary" in data:
            data["Footnotes"] = data["Footnotes Dictionary"]
        elif "footnotes" in data:
            data["Footnotes"] = data["footnotes"]

    if "Optimization Runs" not in data:
        raise KeyError(
            'Missing required key "Optimization Runs". '
            f"Available keys: {list(data.keys())}"
        )

    if not isinstance(data["Optimization Runs"], dict):
        raise TypeError(
            '"Optimization Runs" must be a JSON object/dict, '
            f'got {type(data["Optimization Runs"]).__name__}'
        )

    # Ensure run IDs are strings and entries are dictionaries.
    fixed_runs: dict[str, Any] = {}
    for run_id, run_value in data["Optimization Runs"].items():
        fixed_id = str(run_id)
        if isinstance(run_value, str):
            try:
                run_value = json_loads_flexible(run_value)
            except Exception:
                # Keep as string if it is not JSON; kgwizard may fail later,
                # but the main top-level error will be fixed.
                pass

        if not isinstance(run_value, dict):
            raise TypeError(
                f'Optimization Runs["{fixed_id}"] must be a dict, '
                f"got {type(run_value).__name__}"
            )

        fixed_runs[fixed_id] = run_value

    data["Optimization Runs"] = fixed_runs

    # Also keep the old DataRaider key aligned, in case other code uses it.
    data["Optimization Runs Dictionary"] = fixed_runs

    if "Footnotes" in data and isinstance(data["Footnotes"], dict):
        data["Footnotes Dictionary"] = data["Footnotes"]

    return data


def fix_one_file(input_path: Path, output_path: Path) -> tuple[bool, str]:
    text = input_path.read_text(encoding="utf-8", errors="replace")
    parsed = json_loads_flexible(text)
    fixed = normalize_for_kgwizard(parsed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(fixed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    runs = len(fixed.get("Optimization Runs", {}))
    return True, f"OK: {input_path.name} -> {output_path} ({runs} optimization runs)"


def iter_json_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    files = sorted(p for p in input_path.glob("*.json") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No .json files found in {input_path}")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fix DataRaider JSON files so kgwizard can read them."
    )
    parser.add_argument(
        "input",
        help="Input JSON file or folder, e.g. ./Results/JSON",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Folder for fixed JSON files. Default: <input_folder>_fixed",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite original files. A backup folder is created automatically.",
    )
    parser.add_argument(
        "--backup_dir",
        default=None,
        help="Backup folder for --in-place. Default: <input_folder>/_backup_before_kgwizard_fix",
    )

    args = parser.parse_args()
    input_path = Path(args.input).resolve()

    if not input_path.exists():
        print(f"[ERROR] Input path does not exist: {input_path}")
        return 2

    files = iter_json_files(input_path)

    if args.in_place:
        base_folder = input_path.parent if input_path.is_file() else input_path
        backup_dir = Path(args.backup_dir).resolve() if args.backup_dir else base_folder / "_backup_before_kgwizard_fix"
        backup_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] In-place mode enabled.")
        print(f"[INFO] Backup folder: {backup_dir}")

        output_pairs = []
        for f in files:
            backup_path = backup_dir / f.name
            shutil.copy2(f, backup_path)
            output_pairs.append((f, f))
    else:
        if args.output_dir:
            output_dir = Path(args.output_dir).resolve()
        else:
            if input_path.is_file():
                output_dir = input_path.parent / (input_path.stem + "_fixed")
            else:
                output_dir = input_path.parent / (input_path.name + "_fixed")

        print(f"[INFO] Output folder: {output_dir}")
        output_pairs = [(f, output_dir / f.name) for f in files]

    ok_count = 0
    fail_count = 0

    for input_file, output_file in output_pairs:
        try:
            _, message = fix_one_file(input_file, output_file)
            print("[OK]", message)
            ok_count += 1
        except Exception as e:
            print(f"[FAILED] {input_file}")
            print(f"         {type(e).__name__}: {e}")
            fail_count += 1

    print()
    print("========== Summary ==========")
    print(f"Fixed : {ok_count}")
    print(f"Failed: {fail_count}")

    if fail_count == 0:
        print()
        if args.in_place:
            print("Now retry kgwizard with the original folder:")
            print('  kgwizard transform ./Results/JSON --output_dir ./Results/KGIntermediate --schema photo --graph_name g --address ws://localhost --port 8182 --output_file "C:/Users/user/Documents/GitHub/MERMaid/Results/Graphs/g.graphml"')
        else:
            print("Now retry kgwizard with the fixed folder, for example:")
            print('  kgwizard transform ./Results/JSON_fixed --output_dir ./Results/KGIntermediate --schema photo --graph_name g --address ws://localhost --port 8182 --output_file "C:/Users/user/Documents/GitHub/MERMaid/Results/Graphs/g.graphml"')

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
