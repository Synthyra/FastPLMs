# Maintained diagnostics

This directory contains small, reusable repository checks that do not belong in the
runtime package. Diagnostic scripts must be deterministic, accept their inputs at
runtime, and remain free of credentials, Hub mutations, cache patching, and
machine-specific paths.

`check_notation.py` enforces the documentation and comment notation contract. It is
also exercised by the release test suite.

One-off parity investigations belong in an untracked work directory. Promote an
investigation into this directory only when it becomes a supported diagnostic with
tests and documented inputs.
