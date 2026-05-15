# Outputs Layout

- local-only runtime outputs and checkpoints live here
- date-named folders can be mounted as external knowledge batches from `libra-ingest`
- `libra_state*`: LangGraph checkpoints, run logs, follow-up queues
- `manual_runs/` and `debug/`: ad hoc local experiment artifacts

Do not treat this directory as source-controlled sample data. Team-shared fixtures belong in `examples/`.
