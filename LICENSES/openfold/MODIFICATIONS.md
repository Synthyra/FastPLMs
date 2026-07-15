# OpenFold reference-build modification

FastPLMs pins `aqlaboratory/openfold` revision
`4b41059694619831a7db195b7e0988fc4ff3a307` as an ESMFold parity oracle.
The pinned checkout under `vendor/upstream/openfold/` is not modified.

The `reference-esmfold` image applies
`docker/constraints/openfold-sm90.patch` to the copied `setup.py` before
installing OpenFold. The patch replaces OpenFold's build-time fallback CUDA
architecture list with `sm90` and changes the extension build flag from C++14
to C++17. Docker BuildKit cannot observe the workstation GPU, so the original
setup otherwise requests legacy architectures that CUDA 12.1 no longer
accepts. PyTorch 2.2 requires C++17, and the H100 requires an `sm90` extension.

This is a build-only packaging change. It does not alter OpenFold model classes,
extension source, checkpoint data, or the public API used by the native oracle.
The resulting reference image is specific to the declared H100 validation
environment.

OpenFold imports PyTorch Lightning and NVIDIA DLLogger from its package
initializers even though ESMFold inference does not use their training or
logging features. The native image therefore pins PyTorch Lightning `1.9.5`,
TorchMetrics `0.11.4`, Lightning Utilities `0.15.2`, and NVIDIA DLLogger commit
`0478734ff7be75adde8d160e04872664d1c62e5f`. These Apache-2.0 dependencies are
native-reference-only and are not FastPLMs runtime dependencies.
