# Official reference implementations

The repositories in `vendor/upstream/` are pinned parity oracles. They define
the official configuration, tokenization, parameter, and inference behavior
against which FastPLMs is tested. Production code under `src/fastplms/` must not
import them, copy their source, or require them at runtime.

Initialize the complete reference tree with:

```bash
git submodule update --init --recursive
```

The committed Git link selects the revision. Do not follow an upstream branch
or use an unpinned source archive in a compliance run.

## Pinned sources

| Directory | Revision | Reference environment | License source |
|---|---|---|---|
| `ankh` | `02b4e25ce5389b9e771c9df6e546c62af1216f8e` | `reference-ankh` | `LICENSE.md` |
| `biohub-esm` | `82ee35553d39169d678f784c8d3f8712ffd7d2c4` | `reference-biohub-esm`, `reference-esmfold2` | `LICENSE.md`, `THIRD_PARTY_NOTICE.md` |
| `biohub-transformers` | `3a8956fb4d4ea16b0ec8e71deef2c2909b6a5cbf` | `reference-biohub-esm`, `reference-esmfold2` | `LICENSE` |
| `boltz` | `b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc` | `reference-boltz2` | `LICENSE` |
| `dplm` | `8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d` | `reference-dplm` | `LICENSE` |
| `e1` | `bfd2620a602248499f3d2583d85a7ecddf0b6e02` | `reference-e1` | `LICENSE`, `ATTRIBUTION`, `NOTICE` |
| `fair-esm` | `2b369911bb5b4b0dda914521b9475cad1656b2ac` | `reference-esm2`, `reference-esmfold` | `LICENSE` |
| `openfold` | `4b41059694619831a7db195b7e0988fc4ff3a307` | `reference-esmfold` | `LICENSE` |
| `protein-ttt` | `fde2817cd84b936167cc76ccabf31e5c0fe49962` | `reference-protein-ttt` | `LICENSE` |

`src/fastplms/models.toml` is the machine-readable source for these revisions,
their model-family ownership, checkpoint snapshots, and license expressions.

## Revision updates

Update one source at a time:

1. Review upstream code, dependency, model, tokenizer, and license changes.
2. Check out a specific commit in the corresponding submodule.
3. Update the matching revision and legal-file digests in `models.toml`, then
   refresh the verbatim copies and notices under `LICENSES/`.
4. Rebuild only the owning reference image.
5. Run exact configuration, tokenizer, state, and live inference compliance for
   every affected checkpoint.
6. Record any intentional semantic difference before the Git link is changed.

An upstream revision is not accepted because its tests pass in isolation. It is
accepted only after the FastPLMs compliance contract passes against that exact
source.

## Container and distribution boundary

Reference images receive only the source directories assigned to their target.
Runtime images and local Hub artifacts contain FastPLMs code, model assets, and
required notices, but never an official source checkout. Checkpoint weights are
immutable Hub snapshots identified in `models.toml`; they are not Git
submodules.

## Reference adapter boundary

Reference adapters may call an upstream repository's public loading and
inference APIs, then normalize the returned configuration, tokenizer, state, or
output into the shared parity protocol. They must remain independent oracles:

- They must not import `fastplms` or any production module under
  `src/fastplms/`.
- They must not patch upstream model classes or replace upstream layers.
- They must not reuse FastPLMs tokenizers, checkpoint loaders, converters, or
  attention implementations.
- They must not reconstruct an upstream forward pass from copied equations.

An adapter that cannot obtain a required value through the pinned
implementation's public API must report that limitation explicitly. It must not
manufacture an equivalent result with FastPLMs code.
