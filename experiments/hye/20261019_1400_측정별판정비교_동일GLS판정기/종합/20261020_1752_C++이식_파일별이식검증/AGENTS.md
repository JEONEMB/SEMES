# C++ migration workspace

- Read `../ALGORITHM_RULES.md` before modifying tracking, models, evaluation, or explanatory figures. It remains the single algorithm-rule source until the final verified archive transition.
- Latest user direction (2026-10-27): finish the complete replacement, port Python implementations to native C++, and copy data and runtime assets into a self-contained final structure. Do the implementation and verification without asking again.
- Preserve originals while migrating. Once replacement and isolated reruns are verified, move the originals to legacy with a hash-verified move manifest. Previous instructions prohibiting copies/legacy additions and deferring recent experiments are superseded.
- Data copies and saved reference outputs are inputs/provenance, never evidence that their producing Python algorithms have been ported. Keep implementation, replay and independent numerical parity separate.
- Place new C++ source, experiment descriptions, provenance, and new results inside `final`.
- Latest Git direction: organize `final` files only for now. Defer commits and pushes.
- Keep each historical experiment's inputs, splits, origin assumptions, and results distinct. Record source creation/modified times separately from observed write events and new port execution times. Do not infer authorship from time alone.
- Report partial ports and numerical differences explicitly. Do not treat reading a saved result as reimplementing the experiment.
- Maintain `전체_파일_목록.md` with one checkbox per original source, C++ destination, relationships, source hash/timestamps, write-event evidence and validation. Check only implemented, built and verified files.
- New algorithm folders use `analysis_vX_YYYYMMDD_유지한것_바뀐것`. Inventory/provenance snapshots and results stay in separate numbered folders.
- Update `config/port_progress.json` and reviewed relationships, then refresh the master through native `file_list`. Preserve previous numbered lists and results.
