# Profluent-E1 modified-file notice

FastPLMs implements Profluent-E1 behavior against the pinned official source at
revision `bfd2620a602248499f3d2583d85a7ecddf0b6e02`. The FastPLMs files listed
below are modified or independently reorganized implementations of the
corresponding E1 interfaces. They are not byte-for-byte copies of the upstream
files.

| FastPLMs file | Modification notice |
|---|---|
| `src/fastplms/models/e1/modeling_e1.py` | Reorganized for Transformers AutoClasses, shared attention selection, sequence and RAG preparation, task heads, and checkpoint-compatible loading. |
| `src/fastplms/models/e1/get_weights.py` | Added deterministic mapping from the official checkpoint layout to the FastPLMs checkpoint layout. |
| `src/fastplms/models/e1/__init__.py` | Added FastPLMs package exports. |
| `src/fastplms/models/e1/README.md` | Added FastPLMs loading, embedding, fine-tuning, and behavior notes. |

These changes were present in the FastPLMs 1.0 repository as reviewed on
2026-07-14. The conversion identifier is `e1_to_fastplms_v1`. Recipients must
retain the Profluent-E1 agreement, `ATTRIBUTION`, `NOTICE`, this modified-file
notice, and the applicable Apache-2.0 and BSD-3-Clause texts.

The BSD-3-Clause component is the padding utility identified by the official E1
repository as adapted from Dao-AILab FlashAttention. FastPLMs does not copy that
official utility into its production package, but preserves the notice because
the official source is the parity oracle for the E1 behavior contract.
