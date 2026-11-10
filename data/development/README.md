# Frozen synthetic dynamic FJSP + heterogeneous AGV dataset V2

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
