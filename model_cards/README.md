# Generated model cards

Checkpoint cards in this directory are generated from
`src/fastplms/models.toml`:

```bash
PYTHONPATH=src python -m tools.artifacts.generate_docs
```

Use `--check` to reject stale cards without writing files. Edit the typed
manifest or the renderer, not an individual generated card. Keep model cards
in this directory rather than beside runtime modules under
`src/fastplms/models/`.

Each card combines:

- installation and platform requirements before the Hub quick start;
- direct Hub and offline artifact loading;
- family-appropriate preparation, inference, embedding, generation, or folding
  examples;
- declared AutoClasses, backends, precision, and generation behavior;
- immutable checkpoint and upstream source records;
- validation boundaries, limitations, and checkpoint terms.
- explicit AutoClass weight status and weight-publication policy;
- ESMC backend diagnostic tables whose missing frozen-head measurements remain
  labeled pending rather than inferred.

Examples must follow the current public API. Do not restore legacy backend
names, pickle embedding output, unsupported TTT paths, or capabilities from
another family. Release tests compile the examples, validate local links and
Hub license metadata, and check representative family-specific sections. They
do not execute every model example or establish a scientific result.
