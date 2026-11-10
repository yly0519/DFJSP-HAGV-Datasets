# DFJSP-HAGV Datasets

Research datasets for dynamic multi-objective flexible job shop scheduling with heterogeneous automated guided vehicles (AGVs), accompanying the LLM-HGT-PPO study. The objectives are makespan, total tardiness, and total flow time.

## Released instances

| Collection | Instances | Purpose |
|---|---:|---|
| `data/development/train` | 140 | Synthetic training instances across seven job–machine sizes |
| `data/development/validation` | 42 | Frozen synthetic validation instances |
| `data/formal_test` | 30 | MK01–MK10 with three fixed dynamic scenarios per benchmark |
| `data/generalization` | 70 | Ten paired replicates under each of seven conditions |

The seven development sizes are 10×6, 10×15, 15×4, 15×8, 20×5, 20×10, and 20×15. These synthetic instances were generated independently; their routes and processing times were not copied from MK01–MK10. Generalization conditions vary arrival intensity, AGV fleet composition, and problem size. See `protocols/generalization_conditions.json` for the exact settings.

The formal test instances extend the Brandimarte benchmarks. Their original operation sequences, eligible machines, and processing times are preserved. Dynamic arrivals, due dates, job transportation demands, and AGV information are added. These 30 JSON files materialize the inputs used by the formal evaluation; no new test scenarios have been substituted.

## Repository structure

```text
data/
  development/       # Frozen training/validation files and original checksums
  formal_test/       # Thirty materialized dynamic MK instances
  generalization/    # Seventy original frozen JSON instances
  reference/brandimarte/  # The ten unmodified source files used in the experiments
protocols/           # Fixed seeds, environment settings, and condition manifests
src/                 # Original dataset generation and benchmark adapter scripts
generate_datasets.py # Regeneration and checksum verification entry point
DATA_FORMAT.md
THIRD_PARTY_NOTICES.md
checksums.json
```

## Install and verify

Python 3.10 or later and NumPy are sufficient for the dataset commands:

```bash
pip install -r requirements.txt
python generate_datasets.py --verify
```

To regenerate all 282 instances and compare them with the released files:

```bash
python generate_datasets.py --output regenerated_data
```

Use an empty output directory. The generator refuses to overwrite existing contents. Regeneration checks compare the complete JSON instance contents, including seeds, arrivals, processing options, due dates, and AGV settings. This repository provides scheduling inputs and generation scripts; it does not contain the policy-training implementation or trained checkpoints.

The files in `src/` are preserved copies of the original generators. Their optional simulator-based execution checks require the original scheduling environment; they are not needed by the commands above.

## Formal test extension

- Initial jobs: the first `floor(0.7 × number_of_jobs)` jobs are released at time zero.
- Remaining jobs: the first arrives after time 250; each interarrival increment is an integer sampled from 20 through 79.
- Due date: release time plus `int(1.5 × sum of the mean eligible processing times for each operation))`.
- Job class: sampled by the fixed arrival seed using NumPy's legacy seeded random generator; transportation demand is class + 1.
- AGV classes: capacities 1, 2, and 3; speeds 1.2, 1.0, and 0.8.
- Eligibility: vehicle capacity must be at least the job's transportation demand.
- Travel distance: `abs(previous_location - destination_machine) + 1`; from the initial loading location it is `destination_machine + 1`. Travel time is distance divided by vehicle speed. Machine IDs are zero-based.

The original formal evaluator used the physical environment settings stored in `protocols/formal_environment.json` (15 machine objects and 12 AGVs). The number of machines declared in each source benchmark is separately recorded as `base_benchmark.machine_count_in_source`. Eligible machine sets remain those of the source benchmark; the release preserves the evaluated configuration without changing it. Synthetic and generalization instances carry their own environment settings.

## Load an instance

```python
import json
from pathlib import Path

case = json.loads(Path("data/formal_test/MK01_dyn_r1.json").read_text(encoding="utf-8"))
jobs = case["jobs"]
fleet = case["environment"]["agv_fleet"]
```

See `DATA_FORMAT.md` for field definitions. No physical unit conversion has been applied; processing and transportation times use the common simulation time scale.

## Data provenance and citation

The ten source MK files match the hashes recorded before the formal experiments. Their public distribution source is [SchedulingLab/fjsp-instances, Brandimarte collection](https://github.com/SchedulingLab/fjsp-instances/tree/main/brandimarte). The source files use zero-based machine IDs. The original benchmark publication is:

Brandimarte, P. (1993). Routing and scheduling in a flexible job shop by tabu search. *Annals of Operations Research, 41*(3), 157–183. https://doi.org/10.1007/BF02023073

Please cite this original source when using the benchmark-derived instances, and cite this repository with the version or commit used. The synthetic development and generalization instances are generated data, not original Brandimarte benchmarks. Third-party attribution is retained in `THIRD_PARTY_NOTICES.md`.
