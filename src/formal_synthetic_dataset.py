"""Generate, load, and verify the frozen formal synthetic FJSP dataset.

The dataset is deliberately independent of the Brandimarte MK files.  It
contains dynamic job arrivals and a heterogeneous AGV fleet represented by
type-specific job/vehicle compatibility.  Every instance is serialized as
JSON so all learning methods can consume exactly the same immutable cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "formal-synthetic-dfjsp-hagv-v1"
PROTOCOL_ID = "SYN-DYN-HAGV-MK01-10-SCALE-V1-20260825"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "formal_synthetic_dynamic_hetero_v1"

# These are only dimensional envelopes.  No route, processing-time, due-date,
# or arrival datum is read from a public benchmark while generating the data.
SCALE_REGIMES = (
    {"id": "J10_M06", "jobs": 10, "initial": 7, "inserted": 3, "machines": 6, "agvs": 6},
    {"id": "J10_M15", "jobs": 10, "initial": 7, "inserted": 3, "machines": 15, "agvs": 12},
    {"id": "J15_M04", "jobs": 15, "initial": 10, "inserted": 5, "machines": 4, "agvs": 6},
    {"id": "J15_M08", "jobs": 15, "initial": 10, "inserted": 5, "machines": 8, "agvs": 9},
    {"id": "J20_M05", "jobs": 20, "initial": 14, "inserted": 6, "machines": 5, "agvs": 6},
    {"id": "J20_M10", "jobs": 20, "initial": 14, "inserted": 6, "machines": 10, "agvs": 9},
    {"id": "J20_M15", "jobs": 20, "initial": 14, "inserted": 6, "machines": 15, "agvs": 12},
)

SPLIT_PROTOCOL = {
    "train": {
        "instances_per_regime": 20,
        "structure_seed_base": 310_000,
        "arrival_seed_base": 410_000,
        "arrival_rates": (0.05, 0.10, 0.15, 0.20),
        "due_profiles": (
            ("tight", 1.10, 1.45),
            ("standard", 1.30, 1.70),
            ("relaxed", 1.55, 2.00),
        ),
    },
    "validation": {
        "instances_per_regime": 6,
        "structure_seed_base": 910_000,
        "arrival_seed_base": 1_010_000,
        "arrival_rates": (0.075, 0.125, 0.175),
        "due_profiles": (
            ("tight", 1.15, 1.50),
            ("standard", 1.35, 1.75),
            ("relaxed", 1.60, 1.95),
        ),
    },
}


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _agv_fleet(count: int, type_count: int = 3) -> list[dict[str, int]]:
    return [{"agv_id": idx, "agv_type": idx % type_count} for idx in range(count)]


def _balanced_job_types(job_count: int, rng: random.Random, type_count: int = 3) -> list[int]:
    job_types = [idx % type_count for idx in range(job_count)]
    rng.shuffle(job_types)
    return job_types


def _generate_instance(
    split: str,
    regime: dict[str, int | str],
    local_index: int,
    structure_seed: int,
    arrival_seed: int,
    arrival_rate: float,
    due_profile: tuple[str, float, float],
) -> dict[str, Any]:
    structure_rng = random.Random(structure_seed)
    arrival_rng = random.Random(arrival_seed)
    job_count = int(regime["jobs"])
    initial_count = int(regime["initial"])
    inserted_count = int(regime["inserted"])
    machine_count = int(regime["machines"])
    agv_count = int(regime["agvs"])
    due_name, due_low, due_high = due_profile
    job_types = _balanced_job_types(job_count, structure_rng)

    jobs: list[dict[str, Any]] = []
    average_workloads: list[float] = []
    min_ops = max(2, machine_count // 2)
    for job_id in range(job_count):
        operation_count = structure_rng.randint(min_ops, machine_count)
        operations = []
        workload = 0.0
        for operation_id in range(operation_count):
            max_options = min(machine_count, max(2, int(math.ceil(machine_count * 0.60))))
            option_count = structure_rng.randint(1, max_options)
            machine_ids = sorted(structure_rng.sample(range(machine_count), option_count))
            machine_options = {
                str(machine_id): structure_rng.randint(10, 100)
                for machine_id in machine_ids
            }
            workload += sum(machine_options.values()) / len(machine_options)
            operations.append(
                {
                    "operation_id": operation_id,
                    "machine_options": machine_options,
                }
            )
        average_workloads.append(workload)
        jobs.append(
            {
                "job_id": job_id,
                "job_type": job_types[job_id],
                "arrival_time": 0.0,
                "due_date": 0.0,
                "priority": round(structure_rng.uniform(0.5, 3.0), 1),
                "operations": operations,
            }
        )

    nominal_initial_load = sum(average_workloads[:initial_count]) / max(1, machine_count)
    insertion_start = max(20.0, round(0.25 * nominal_initial_load, 3))
    current_arrival = insertion_start
    for job_id, job in enumerate(jobs):
        if job_id >= initial_count:
            current_arrival += arrival_rng.expovariate(arrival_rate)
            job["arrival_time"] = round(current_arrival, 3)
        urgency = structure_rng.uniform(due_low, due_high)
        job["due_date"] = round(job["arrival_time"] + average_workloads[job_id] * urgency, 3)

    instance_id = f"{split}_{regime['id']}_{local_index:03d}"
    instance = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "instance_id": instance_id,
        "split": split,
        "regime_id": regime["id"],
        "seeds": {
            "structure_seed": structure_seed,
            "arrival_seed": arrival_seed,
        },
        "environment": {
            "n_initial": initial_count,
            "n_insert": inserted_count,
            "m": machine_count,
            "n_agv": agv_count,
            "agv_types": 3,
            "agv_heterogeneity_model": "exclusive_job_type_compatibility",
            "agv_fleet": _agv_fleet(agv_count),
            "transport_distance_model": "abs(origin_machine-destination_machine)+1",
        },
        "dynamic_profile": {
            "arrival_process": "shifted_poisson_process",
            "arrival_rate": arrival_rate,
            "insertion_start": insertion_start,
            "due_profile": due_name,
            "due_factor_range": [due_low, due_high],
        },
        "jobs": jobs,
    }
    instance["content_fingerprint"] = _instance_content_fingerprint(instance)
    return instance


def _instance_content_fingerprint(instance: dict[str, Any]) -> str:
    payload = {
        "environment": instance["environment"],
        "dynamic_profile": instance["dynamic_profile"],
        "jobs": instance["jobs"],
    }
    return _sha256_bytes(_json_bytes(payload))


def jobs_from_instance(instance: dict[str, Any]) -> list[tuple[Any, ...]]:
    """Return the tuple representation accepted by ``FJSP_DES_Env.reset``."""
    jobs = []
    for job in instance["jobs"]:
        operations = []
        for operation in job["operations"]:
            operations.append(
                {int(machine): int(duration) for machine, duration in operation["machine_options"].items()}
            )
        jobs.append(
            (
                int(job["job_id"]),
                int(job["job_type"]),
                float(job["arrival_time"]),
                float(job["due_date"]),
                operations,
                float(job["priority"]),
            )
        )
    return jobs


def environment_config_from_instance(instance: dict[str, Any]) -> dict[str, Any]:
    env = instance["environment"]
    return {
        "n_initial": int(env["n_initial"]),
        "n_insert": int(env["n_insert"]),
        "m": int(env["m"]),
        "n_agv": int(env["n_agv"]),
        "agv_types": int(env["agv_types"]),
        "Tr": float(instance["dynamic_profile"]["insertion_start"]),
        "arrival_rate": float(instance["dynamic_profile"]["arrival_rate"]),
    }


def load_instance(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_frozen_split(dataset_root: str | Path, split: str) -> list[dict[str, Any]]:
    root = Path(dataset_root)
    manifest = json.loads((root / "freeze_manifest.json").read_text(encoding="utf-8"))
    prefix = f"{split}/"
    paths = sorted(item["path"] for item in manifest["files"] if item["path"].startswith(prefix))
    return [load_instance(root / path) for path in paths]


def _protocol_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN",
        "created_on": "2026-08-25",
        "purpose": {
            "train": "policy learning and search on synthetic instances only",
            "validation": "model selection, rule screening, and safety selection only",
            "public_test": "reserved for a later one-time MK01-MK10 evaluation",
        },
        "data_isolation": {
            "public_benchmark_files_read_during_generation": False,
            "mk01_mk10_routes_or_processing_times_used": False,
            "mk11_mk15_used": False,
            "validation_used_for_gradient_updates": False,
            "split_seed_sets_must_be_disjoint": True,
            "split_content_fingerprints_must_be_disjoint": True,
        },
        "problem_definition": {
            "objectives": ["makespan", "total_tardiness", "total_flow_time"],
            "dynamic_orders": "every instance has jobs released after time zero",
            "heterogeneous_agvs": "three vehicle types with exclusive job-type compatibility",
            "processing_time_range": [10, 100],
            "priority_range": [0.5, 3.0],
        },
        "scale_regimes": list(SCALE_REGIMES),
        "split_protocol": {
            split: {
                "instances_per_regime": int(spec["instances_per_regime"]),
                "structure_seed_base": int(spec["structure_seed_base"]),
                "arrival_seed_base": int(spec["arrival_seed_base"]),
                "arrival_rates": list(spec["arrival_rates"]),
                "due_profiles": [list(item) for item in spec["due_profiles"]],
            }
            for split, spec in SPLIT_PROTOCOL.items()
        },
    }


def _readme_text() -> str:
    return """# Frozen synthetic dynamic FJSP + heterogeneous AGV dataset

This directory is the immutable development dataset for the formal protocol.

- `train/`: 140 synthetic instances for learning/search only.
- `validation/`: 42 independent synthetic instances for model selection, rule
  screening, and safety selection only.
- Every instance contains dynamic inserted jobs and three AGV compatibility
  types.
- No MK01--MK15 process route, processing time, due date, or arrival datum was
  read or copied during generation.  The seven dimensional regimes only span
  the 10--20 job and 4--15 machine range needed before public testing.
- `freeze_manifest.json` and `FROZEN.sha256` make any later content change
  detectable.  Run `formal_synthetic_dataset.py --verify` before every formal
  training batch.

The public MK benchmark remains external to this directory and is not a source
of gradients, checkpoint selection, rule generation, or repair feedback.
"""


def generate_dataset(output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty frozen dataset directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "protocol.json", _protocol_payload())
    (output / "README.md").write_text(_readme_text(), encoding="utf-8", newline="\n")

    instance_rows = []
    for split, split_spec in SPLIT_PROTOCOL.items():
        count = int(split_spec["instances_per_regime"])
        for regime_index, regime in enumerate(SCALE_REGIMES):
            for local_index in range(1, count + 1):
                offset = regime_index * 10_000 + local_index
                structure_seed = int(split_spec["structure_seed_base"]) + offset
                arrival_seed = int(split_spec["arrival_seed_base"]) + offset
                arrival_rates = split_spec["arrival_rates"]
                due_profiles = split_spec["due_profiles"]
                arrival_rate = float(arrival_rates[(local_index - 1) % len(arrival_rates)])
                due_profile = due_profiles[(local_index + regime_index - 1) % len(due_profiles)]
                instance = _generate_instance(
                    split,
                    regime,
                    local_index,
                    structure_seed,
                    arrival_seed,
                    arrival_rate,
                    due_profile,
                )
                relative = Path(split) / f"{instance['instance_id']}.json"
                _write_json(output / relative, instance)
                instance_rows.append(
                    {
                        "path": relative.as_posix(),
                        "instance_id": instance["instance_id"],
                        "split": split,
                        "regime_id": regime["id"],
                        "structure_seed": structure_seed,
                        "arrival_seed": arrival_seed,
                        "content_fingerprint": instance["content_fingerprint"],
                    }
                )

    index_payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "instances": instance_rows,
    }
    _write_json(output / "instance_index.json", index_payload)

    listed_paths = [output / "README.md", output / "protocol.json", output / "instance_index.json"]
    listed_paths.extend(output / row["path"] for row in instance_rows)
    files = [
        {
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(listed_paths)
    ]
    dataset_digest = _sha256_bytes(
        "\n".join(f"{item['path']}:{item['sha256']}" for item in files).encode("utf-8")
    )
    split_counts = Counter(row["split"] for row in instance_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN",
        "created_on": "2026-08-25",
        "generator": {
            "path": Path(__file__).name,
            "sha256": _sha256_file(Path(__file__)),
        },
        "instance_counts": dict(sorted(split_counts.items())),
        "file_count": len(files),
        "dataset_sha256": dataset_digest,
        "files": files,
    }
    _write_json(output / "freeze_manifest.json", manifest)
    manifest_hash = _sha256_file(output / "freeze_manifest.json")
    (output / "FROZEN.sha256").write_text(
        f"{manifest_hash}  freeze_manifest.json\n", encoding="utf-8", newline="\n"
    )
    return manifest


def _assert_structure(instance: dict[str, Any], expected_split: str) -> None:
    assert instance["schema_version"] == SCHEMA_VERSION
    assert instance["protocol_id"] == PROTOCOL_ID
    assert instance["split"] == expected_split
    env = instance["environment"]
    jobs = instance["jobs"]
    assert len(jobs) == int(env["n_initial"]) + int(env["n_insert"])
    assert int(env["n_insert"]) > 0
    assert int(env["agv_types"]) >= 2
    fleet_types = {int(item["agv_type"]) for item in env["agv_fleet"]}
    assert fleet_types == set(range(int(env["agv_types"])))
    assert len(env["agv_fleet"]) == int(env["n_agv"])
    arrivals = [float(job["arrival_time"]) for job in jobs]
    assert all(value == 0.0 for value in arrivals[: int(env["n_initial"])])
    assert all(value > 0.0 for value in arrivals[int(env["n_initial"]) :])
    assert arrivals[int(env["n_initial"]) :] == sorted(arrivals[int(env["n_initial"]) :])
    job_types = {int(job["job_type"]) for job in jobs}
    assert job_types == set(range(int(env["agv_types"])))
    for job in jobs:
        assert float(job["due_date"]) > float(job["arrival_time"])
        assert 0.5 <= float(job["priority"]) <= 3.0
        assert job["operations"]
        for expected_operation_id, operation in enumerate(job["operations"]):
            assert int(operation["operation_id"]) == expected_operation_id
            assert operation["machine_options"]
            for machine_id, duration in operation["machine_options"].items():
                assert 0 <= int(machine_id) < int(env["m"])
                assert 10 <= int(duration) <= 100
    assert instance["content_fingerprint"] == _instance_content_fingerprint(instance)


def _dispatch_rule(jobs: Iterable[Any], machines: Iterable[Any], agvs: Iterable[Any]):
    ready = [
        job for job in jobs
        if not job.finished and job.ops[job.current_op_idx].status != 2
    ]
    if not ready:
        return None, None, None
    job = min(ready, key=lambda item: (item.due_date, item.remaining_ops, item.id))
    compatible_agvs = [agv for agv in agvs if agv.type == job.type]
    agv = min(compatible_agvs, key=lambda item: (item.free_time, item.id))
    operation = job.ops[job.current_op_idx]
    eligible_machines = [machine for machine in machines if machine.id in operation.machine_options]
    machine = min(
        eligible_machines,
        key=lambda item: (max(item.free_time, agv.free_time) + operation.machine_options[item.id], item.id),
    )
    return job, machine, agv


def _execution_smoke(instance: dict[str, Any]) -> tuple[float, float, float]:
    from des_engine import FJSP_DES_Env

    env = FJSP_DES_Env(environment_config_from_instance(instance))
    env.reset(jobs_from_instance(instance))
    total_operations = sum(len(job["operations"]) for job in instance["jobs"])
    max_steps = max(500, 6 * total_operations)
    done = False
    steps = 0
    while not done and steps < max_steps:
        _, done, _ = env.step(_dispatch_rule, 0)
        steps += 1
    assert done, f"Execution smoke did not finish {instance['instance_id']} in {max_steps} steps"
    metrics = tuple(float(value) for value in env.calculate_metrics())
    assert all(math.isfinite(value) and value >= 0.0 for value in metrics)
    return metrics


def verify_dataset(output: Path, execute_all: bool = True) -> dict[str, Any]:
    manifest_path = output / "freeze_manifest.json"
    frozen_line = (output / "FROZEN.sha256").read_text(encoding="utf-8").strip().split()
    assert len(frozen_line) == 2 and frozen_line[1] == "freeze_manifest.json"
    assert frozen_line[0] == _sha256_file(manifest_path), "Freeze manifest hash mismatch"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "FROZEN"
    for item in manifest["files"]:
        path = output / item["path"]
        assert path.stat().st_size == int(item["bytes"]), f"Size mismatch: {item['path']}"
        assert _sha256_file(path) == item["sha256"], f"Hash mismatch: {item['path']}"
    rebuilt_digest = _sha256_bytes(
        "\n".join(f"{item['path']}:{item['sha256']}" for item in manifest["files"]).encode("utf-8")
    )
    assert rebuilt_digest == manifest["dataset_sha256"]

    index = json.loads((output / "instance_index.json").read_text(encoding="utf-8"))
    structure_seeds: dict[str, set[int]] = {"train": set(), "validation": set()}
    arrival_seeds: dict[str, set[int]] = {"train": set(), "validation": set()}
    fingerprints: dict[str, set[str]] = {"train": set(), "validation": set()}
    smoke_metrics = []
    for row in index["instances"]:
        instance = load_instance(output / row["path"])
        split = row["split"]
        _assert_structure(instance, split)
        assert row["content_fingerprint"] == instance["content_fingerprint"]
        assert int(row["structure_seed"]) not in structure_seeds[split]
        assert int(row["arrival_seed"]) not in arrival_seeds[split]
        assert row["content_fingerprint"] not in fingerprints[split]
        structure_seeds[split].add(int(row["structure_seed"]))
        arrival_seeds[split].add(int(row["arrival_seed"]))
        fingerprints[split].add(row["content_fingerprint"])
        if execute_all:
            smoke_metrics.append(_execution_smoke(instance))
    assert structure_seeds["train"].isdisjoint(structure_seeds["validation"])
    assert arrival_seeds["train"].isdisjoint(arrival_seeds["validation"])
    assert fingerprints["train"].isdisjoint(fingerprints["validation"])
    assert len(fingerprints["train"]) == int(manifest["instance_counts"]["train"])
    assert len(fingerprints["validation"]) == int(manifest["instance_counts"]["validation"])
    return {
        "status": "PASS",
        "protocol_id": manifest["protocol_id"],
        "train_instances": len(fingerprints["train"]),
        "validation_instances": len(fingerprints["validation"]),
        "executed_instances": len(smoke_metrics),
        "dataset_sha256": manifest["dataset_sha256"],
        "manifest_sha256": _sha256_file(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--structural-only", action="store_true")
    args = parser.parse_args()
    if not args.generate and not args.verify:
        parser.error("Choose --generate or --verify")
    if args.generate:
        generate_dataset(args.output)
    result = verify_dataset(args.output, execute_all=not args.structural_only)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
