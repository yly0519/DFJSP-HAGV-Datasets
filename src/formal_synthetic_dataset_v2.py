"""Freeze and verify the capacity-speed AGV development protocol (V2).

V2 deliberately reuses V1's independently generated processing structures,
arrival processes and split seeds.  It changes only the transport model:
jobs carry demands 1--3, vehicles carry nested capacities 1--3 and inverse
speed levels 1.2--0.8.  V1 is imported read-only and is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import formal_synthetic_dataset as v1


SCHEMA_VERSION = "formal-synthetic-dfjsp-hagv-v2"
PROTOCOL_ID = "SYN-DYN-HAGV-CAPSPEED-MK01-10-SCALE-V2-20260825"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "formal_synthetic_dynamic_hetero_v2"
CAPACITY_BY_TYPE = {0: 1.0, 1: 2.0, 2: 3.0}
SPEED_BY_TYPE = {0: 1.2, 1: 1.0, 2: 0.8}


def _agv_fleet(count: int) -> list[dict[str, Any]]:
    return [
        {
            "agv_id": idx,
            "agv_type": idx % 3,
            "capacity": CAPACITY_BY_TYPE[idx % 3],
            "speed": SPEED_BY_TYPE[idx % 3],
        }
        for idx in range(count)
    ]


def _upgrade_instance(instance: dict[str, Any]) -> dict[str, Any]:
    instance = json.loads(json.dumps(instance))
    instance["schema_version"] = SCHEMA_VERSION
    instance["protocol_id"] = PROTOCOL_ID
    env = instance["environment"]
    env["agv_heterogeneity_model"] = "nested_capacity_compatibility_with_speed"
    env["agv_fleet"] = _agv_fleet(int(env["n_agv"]))
    env["transport_time_model"] = "transport_distance/agv_speed"
    env["eligibility_rule"] = "agv_capacity >= job_transport_demand"
    for job in instance["jobs"]:
        job["transport_demand"] = float(int(job["job_type"]) + 1)
    instance["content_fingerprint"] = v1._instance_content_fingerprint(instance)
    return instance


def jobs_from_instance(instance: dict[str, Any]) -> list[tuple[Any, ...]]:
    jobs = []
    for job in instance["jobs"]:
        operations = [
            {int(machine): int(duration) for machine, duration in op["machine_options"].items()}
            for op in job["operations"]
        ]
        jobs.append((
            int(job["job_id"]),
            int(job["job_type"]),
            float(job["arrival_time"]),
            float(job["due_date"]),
            operations,
            float(job["priority"]),
            float(job["transport_demand"]),
        ))
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
        "agv_capacity_speed_heterogeneity_enabled": True,
        "agv_fleet": [dict(item) for item in env["agv_fleet"]],
    }


def load_instance(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _protocol_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "created_on": "2026-08-25",
        "purpose": "frozen dynamic FJSP development with bounded AGV capacity-speed heterogeneity",
        "public_benchmark_dependency": False,
        "parent_structure_protocol": v1.PROTOCOL_ID,
        "split_protocol": v1.SPLIT_PROTOCOL,
        "scale_regimes": list(v1.SCALE_REGIMES),
        "transport_model": {
            "job_demand_levels": [1.0, 2.0, 3.0],
            "agv_capacity_levels": [1.0, 2.0, 3.0],
            "agv_speed_levels": [1.2, 1.0, 0.8],
            "eligibility": "capacity >= demand",
            "time": "distance / speed",
            "batching": False,
            "battery": False,
            "failures": False,
        },
    }


def _readme_text() -> str:
    return """# Frozen synthetic dynamic FJSP + heterogeneous AGV dataset V2

V2 preserves V1's independently generated job-shop structures, dynamic
arrivals, due dates and train/validation split seeds.  Its bounded transport
heterogeneity adds job demand, nested vehicle capacity and vehicle speed.

- job demand: 1, 2 or 3;
- vehicle capacity: 1, 2 or 3;
- compatibility: capacity >= demand;
- speed by vehicle class: 1.2, 1.0 or 0.8;
- travel time: deterministic distance / speed.

No battery, charging, batching, breakdown or stochastic travel mechanism is
introduced.  Public MK instances remain excluded from development feedback.
"""


def generate_dataset(output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty frozen dataset directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    v1._write_json(output / "protocol.json", _protocol_payload())
    (output / "README.md").write_text(_readme_text(), encoding="utf-8", newline="\n")
    rows = []
    for split, split_spec in v1.SPLIT_PROTOCOL.items():
        count = int(split_spec["instances_per_regime"])
        for regime_index, regime in enumerate(v1.SCALE_REGIMES):
            for local_index in range(1, count + 1):
                offset = regime_index * 10_000 + local_index
                structure_seed = int(split_spec["structure_seed_base"]) + offset
                arrival_seed = int(split_spec["arrival_seed_base"]) + offset
                rates = split_spec["arrival_rates"]
                profiles = split_spec["due_profiles"]
                instance = v1._generate_instance(
                    split,
                    regime,
                    local_index,
                    structure_seed,
                    arrival_seed,
                    float(rates[(local_index - 1) % len(rates)]),
                    profiles[(local_index + regime_index - 1) % len(profiles)],
                )
                instance = _upgrade_instance(instance)
                relative = Path(split) / f"{instance['instance_id']}.json"
                v1._write_json(output / relative, instance)
                rows.append({
                    "path": relative.as_posix(),
                    "instance_id": instance["instance_id"],
                    "split": split,
                    "regime_id": regime["id"],
                    "structure_seed": structure_seed,
                    "arrival_seed": arrival_seed,
                    "content_fingerprint": instance["content_fingerprint"],
                })
    v1._write_json(output / "instance_index.json", {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "instances": rows,
    })
    listed = [output / "README.md", output / "protocol.json", output / "instance_index.json"]
    listed.extend(output / row["path"] for row in rows)
    files = [{
        "path": path.relative_to(output).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": v1._sha256_file(path),
    } for path in sorted(listed)]
    digest = v1._sha256_bytes(
        "\n".join(f"{item['path']}:{item['sha256']}" for item in files).encode("utf-8")
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN",
        "created_on": "2026-08-25",
        "generator": {"path": Path(__file__).name, "sha256": v1._sha256_file(Path(__file__))},
        "parent_generator": {"path": Path(v1.__file__).name, "sha256": v1._sha256_file(Path(v1.__file__))},
        "instance_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "file_count": len(files),
        "dataset_sha256": digest,
        "files": files,
    }
    v1._write_json(output / "freeze_manifest.json", manifest)
    manifest_hash = v1._sha256_file(output / "freeze_manifest.json")
    (output / "FROZEN.sha256").write_text(
        f"{manifest_hash}  freeze_manifest.json\n", encoding="utf-8", newline="\n"
    )
    return manifest


def _assert_structure(instance: dict[str, Any], split: str) -> None:
    assert instance["schema_version"] == SCHEMA_VERSION
    assert instance["protocol_id"] == PROTOCOL_ID
    assert instance["split"] == split
    env, jobs = instance["environment"], instance["jobs"]
    assert len(jobs) == int(env["n_initial"]) + int(env["n_insert"])
    assert len(env["agv_fleet"]) == int(env["n_agv"])
    capacities = {float(item["capacity"]) for item in env["agv_fleet"]}
    speeds = {float(item["speed"]) for item in env["agv_fleet"]}
    assert capacities == {1.0, 2.0, 3.0}
    assert speeds == {0.8, 1.0, 1.2}
    demands = {float(job["transport_demand"]) for job in jobs}
    assert demands == {1.0, 2.0, 3.0}
    assert all(any(float(a["capacity"]) >= float(j["transport_demand"]) for a in env["agv_fleet"]) for j in jobs)
    arrivals = [float(job["arrival_time"]) for job in jobs]
    n_initial = int(env["n_initial"])
    assert all(value == 0.0 for value in arrivals[:n_initial])
    assert arrivals[n_initial:] == sorted(arrivals[n_initial:])
    for job in jobs:
        assert float(job["due_date"]) > float(job["arrival_time"])
        assert float(job["transport_demand"]) == float(int(job["job_type"]) + 1)
        for expected_id, operation in enumerate(job["operations"]):
            assert int(operation["operation_id"]) == expected_id
            for machine, duration in operation["machine_options"].items():
                assert 0 <= int(machine) < int(env["m"])
                assert 10 <= int(duration) <= 100
    assert instance["content_fingerprint"] == v1._instance_content_fingerprint(instance)


def _dispatch_rule(jobs: Iterable[Any], machines: Iterable[Any], agvs: Iterable[Any]):
    from des_engine import agv_can_serve, agv_transport_time
    ready = [job for job in jobs if not job.finished and job.ops[job.current_op_idx].status != 2]
    if not ready:
        return None, None, None
    job = min(ready, key=lambda item: (item.due_date, item.remaining_ops, item.id))
    compatible = [agv for agv in agvs if agv_can_serve(job, agv)]
    operation = job.ops[job.current_op_idx]
    choices = []
    for agv in compatible:
        for machine in machines:
            if machine.id in operation.machine_options:
                travel = agv_transport_time(agv, machine)
                finish = max(machine.free_time, agv.free_time + travel) + operation.machine_options[machine.id]
                choices.append((finish, travel, agv.id, machine.id, machine, agv))
    _, _, _, _, machine, agv = min(choices)
    return job, machine, agv


def _execution_smoke(instance: dict[str, Any]) -> tuple[float, float, float]:
    from des_engine import FJSP_DES_Env
    env = FJSP_DES_Env(environment_config_from_instance(instance))
    env.reset(jobs_from_instance(instance))
    max_steps = max(500, 6 * sum(len(job["operations"]) for job in instance["jobs"]))
    done = False
    for _ in range(max_steps):
        _, done, _ = env.step(_dispatch_rule, 0)
        if done:
            break
    assert done, f"Execution smoke failed: {instance['instance_id']}"
    metrics = tuple(float(value) for value in env.calculate_metrics())
    assert all(math.isfinite(value) and value >= 0.0 for value in metrics)
    return metrics


def verify_dataset(output: Path, execute_all: bool = True) -> dict[str, Any]:
    manifest_path = output / "freeze_manifest.json"
    frozen = (output / "FROZEN.sha256").read_text(encoding="utf-8").strip().split()
    assert len(frozen) == 2 and frozen[1] == "freeze_manifest.json"
    assert frozen[0] == v1._sha256_file(manifest_path)
    manifest = load_instance(manifest_path)
    assert manifest["status"] == "FROZEN"
    for item in manifest["files"]:
        path = output / item["path"]
        assert path.stat().st_size == int(item["bytes"])
        assert v1._sha256_file(path) == item["sha256"]
    rebuilt = v1._sha256_bytes(
        "\n".join(f"{item['path']}:{item['sha256']}" for item in manifest["files"]).encode("utf-8")
    )
    assert rebuilt == manifest["dataset_sha256"]
    index = load_instance(output / "instance_index.json")
    seen = {"train": set(), "validation": set()}
    smoke = []
    for row in index["instances"]:
        instance = load_instance(output / row["path"])
        _assert_structure(instance, row["split"])
        assert instance["content_fingerprint"] == row["content_fingerprint"]
        assert row["content_fingerprint"] not in seen[row["split"]]
        seen[row["split"]].add(row["content_fingerprint"])
        if execute_all:
            smoke.append(_execution_smoke(instance))
    assert seen["train"].isdisjoint(seen["validation"])
    assert len(seen["train"]) == int(manifest["instance_counts"]["train"])
    assert len(seen["validation"]) == int(manifest["instance_counts"]["validation"])
    return {
        "status": "PASS",
        "protocol_id": manifest["protocol_id"],
        "train_instances": len(seen["train"]),
        "validation_instances": len(seen["validation"]),
        "dataset_sha256": manifest["dataset_sha256"],
        "executed_instances": len(smoke),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--execute-all", action="store_true")
    args = parser.parse_args()
    result = verify_dataset(args.output, execute_all=args.execute_all) if args.verify else generate_dataset(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
