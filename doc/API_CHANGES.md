# API Changes

This repository is prototype-first. API changes are expected.

## 2026-03-09

### `a08d8a8` - Remove legacy motion-planning services
- Removed legacy services:
  - `~/plan_block_motion`
  - `~/execute_planned_motion`
- Removed legacy interfaces:
  - `srv/PlanBlockMotion.srv`
  - `srv/ExecutePlannedMotion.srv`
- Kept clean service API:
  - `~/plan_geometric_path`
  - `~/compute_trajectory`
  - `~/execute_trajectory`
  - `~/execute_named_configuration`
  - `~/get_next_assembly_task`

### `1b5646a` - Add integration coverage for clean API
- Added ROS integration tests for service availability and key request/response paths.

### `a1c8cf1` - Add wall-plan and launch defaults support
- Added deterministic wall-plan model and loader APIs.
- Added default data/config files for named configurations and wall plans.
- Added wall-plan regression tests.

## Migration Guidance

- Do not depend on removed legacy services.
- Consume only service names/types documented in `doc/SERVICES.md`.
- For behavior tree/service clients, prefer defensive handling of missing IDs and runtime-unavailable responses.
