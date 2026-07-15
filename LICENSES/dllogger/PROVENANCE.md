# NVIDIA DLLogger provenance

- Upstream: `https://github.com/NVIDIA/dllogger`
- Revision: `0478734ff7be75adde8d160e04872664d1c62e5f`
- Installed version: `1.1.0`
- License: Apache-2.0
- Scope: isolated native ESMFold reference image only

The pinned OpenFold package imports DLLogger eagerly from its package
initializer. FastPLMs production code does not import or depend on DLLogger.
The historical OpenFold environment selected DLLogger from an unpinned Git
URL, so the reference image fixes the revision above for reproducibility.

DLLogger's wheel contains its verbatim license at
`dllogger-1.1.0.dist-info/licenses/LICENSE`. The reference image preserves that
file and fails its build if the license is absent or is not the Apache License.
