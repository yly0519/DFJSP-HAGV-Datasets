# Instance format

Each instance is a UTF-8 JSON file. All IDs are zero-based.

| Field | Meaning |
|---|---|
| `instance_id` | Stable case identifier |
| `split` | Training, validation, formal test, or generalization role |
| `seeds` | Fixed generation seeds; formal tests specify the arrival seed |
| `environment.m` | Number of machine objects in the evaluated environment |
| `environment.n_agv` | Number of vehicles |
| `environment.n_initial`, `n_insert` | Initially available and subsequently arriving jobs |
| `environment.agv_fleet` | Vehicle IDs, classes, capacities, and speeds |
| `dynamic_profile` | Arrival and due-date generation settings |
| `jobs` | Full list of jobs |
| `content_fingerprint` | SHA-256 of canonical environment, dynamic-profile, and job content |

Each job contains `job_id`, `job_type`, `transport_demand`, `arrival_time`, `due_date`, `priority`, and `operations`. Each operation contains its sequence index (`operation_id`) and `machine_options`, a mapping from machine ID to processing duration. JSON object keys are strings and should be converted to integers when used as machine indices.

Vehicle capacity must be at least the job demand. Jobs and vehicles do not form processing batches. Battery, charging, failures, and stochastic travel are not part of these datasets.

Formal test cases also include `base_benchmark.file`, `base_benchmark.sha256`, and `base_benchmark.machine_count_in_source`. The benchmark's eligible-machine sets are unchanged. The physical configuration is recorded explicitly rather than inferred from the largest eligible-machine ID.

Development and generalization cases use shifted exponential interarrival times, rounded to three decimal places. Formal MK cases use integer interarrival increments from 20 to 79. Do not replace one arrival process with the other.

For each formal job, the due date is its arrival time plus the integer part of 1.5 times the sum of mean processing durations over the eligible machines of each operation. The integer conversion is part of the original implementation.

The generalization dataset contains seven conditions, with replicate numbers 001–010 paired across conditions. It is a separate synthetic evaluation set. Training seeds for policy models are not dataset-generation seeds and are not additional test instances.
