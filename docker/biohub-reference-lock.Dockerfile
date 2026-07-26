# syntax=docker/dockerfile:1.19@sha256:b6afd42430b15f2d2a4c5a02b919e98a525b785b1aaff16747d2f623364e39b6

# This workflow is intentionally native-only. It builds the BioTraj wheel used
# by the Biohub reference on the GH200 Linux ARM64 target and does not claim an
# x86_64 or emulated build contract.
ARG PYTHON312_ARM64_IMAGE=python:3.12.11-slim-bookworm@sha256:9bb659dc6d5218917236f3711e866a5634bb4c2f208de9d4533aa4863f57c1d3
ARG CUDA13_ARM64_IMAGE=nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04@sha256:6fd95f7235f228fc7cff6f45d7b2d16ed93bddff4e80a94288731c6c7cea81d7

FROM ${PYTHON312_ARM64_IMAGE} AS cpython312-arm64

FROM ${CUDA13_ARM64_IMAGE} AS biotraj-wheel-builder

COPY --from=cpython312-arm64 /usr/local /usr/local

ENV PATH=/opt/biotraj-build/bin:/usr/local/bin:/usr/local/cuda/bin:$PATH \
    SOURCE_DATE_EPOCH=1730547054 \
    PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/opt/contract-root \
    TZ=UTC \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    CFLAGS="-O3 -g0 -fno-record-gcc-switches -ffile-prefix-map=/tmp/biotraj-one=/usr/src/biotraj -ffile-prefix-map=/tmp/biotraj-two=/usr/src/biotraj" \
    LDFLAGS="-Wl,--build-id=none" \
    ARFLAGS=crD \
    ZERO_AR_DATE=1

COPY tools/remote/biohub_reference_lock.py /opt/contract-root/tools/remote/biohub_reference_lock.py
COPY docker/biohub-reference-lock.Dockerfile /opt/contract-root/docker/biohub-reference-lock.Dockerfile
COPY docker/constraints/biohub-reference.in /opt/contract-root/docker/constraints/biohub-reference.in
COPY docker/constraints/biohub-reference.lock.txt /opt/contract-root/docker/constraints/biohub-reference.lock.txt
COPY docker/constraints/biohub-biotraj-build.in /opt/contract-root/docker/constraints/biohub-biotraj-build.in
COPY docker/constraints/biohub-biotraj-build.lock.txt /opt/contract-root/docker/constraints/biohub-biotraj-build.lock.txt
COPY docker/constraints/biohub-reference-lock.json /opt/contract-root/docker/constraints/biohub-reference-lock.json

RUN test "$(uname -m)" = aarch64 && \
    test "$(python --version)" = "Python 3.12.11" && \
    /usr/local/bin/python -m venv /opt/biotraj-build && \
    /opt/biotraj-build/bin/python -m pip install \
        --require-hashes \
        --only-binary=:all: \
        --no-deps \
        --requirement /opt/contract-root/docker/constraints/biohub-biotraj-build.lock.txt && \
    /opt/biotraj-build/bin/python -m pip check && \
    /opt/biotraj-build/bin/python -m tools.remote.biohub_reference_lock verify-contract \
        --root /opt/contract-root \
        --contract /opt/contract-root/docker/constraints/biohub-reference-lock.json

ADD --checksum=sha256:4bcba92101ed50f369cc1487fb5dfcfe1d8402ad47adaa9232b080553271663a \
    https://files.pythonhosted.org/packages/07/21/2287edfd0d2569639eea706e25c39e63b46a384cf1712db8ea05768317b0/biotraj-1.2.2.tar.gz \
    /opt/sources/biotraj-1.2.2.tar.gz

RUN mkdir -p /tmp/biotraj-one /tmp/biotraj-two \
        /opt/wheels/one /opt/wheels/two /opt/artifact && \
    tar -xzf /opt/sources/biotraj-1.2.2.tar.gz \
        --strip-components=1 -C /tmp/biotraj-one && \
    tar -xzf /opt/sources/biotraj-1.2.2.tar.gz \
        --strip-components=1 -C /tmp/biotraj-two && \
    sed -i \
        -e 's|src/biotraj/xtc.pyx|src/biotraj/xtc.c|' \
        -e 's|src/biotraj/trr.pyx|src/biotraj/trr.c|' \
        -e 's|src/biotraj/dcd.pyx|src/biotraj/dcd.c|' \
        /tmp/biotraj-one/setup.py /tmp/biotraj-two/setup.py && \
    test "$(grep -h -E 'src/biotraj/(xtc|trr|dcd)\.c' \
        /tmp/biotraj-one/setup.py /tmp/biotraj-two/setup.py | wc -l)" = 6 && \
    ! grep -q 'src/biotraj/.*\.pyx' \
        /tmp/biotraj-one/setup.py /tmp/biotraj-two/setup.py && \
    umask 022 && \
    /opt/biotraj-build/bin/python -m pip wheel --no-deps --no-build-isolation \
        --wheel-dir /opt/wheels/one /tmp/biotraj-one && \
    /opt/biotraj-build/bin/python -m pip wheel --no-deps --no-build-isolation \
        --wheel-dir /opt/wheels/two /tmp/biotraj-two && \
    test "$(find /opt/wheels/one -maxdepth 1 -type f -name '*.whl' | wc -l)" = 1 && \
    test "$(find /opt/wheels/two -maxdepth 1 -type f -name '*.whl' | wc -l)" = 1 && \
    cmp /opt/wheels/one/*.whl /opt/wheels/two/*.whl && \
    cp /opt/wheels/one/*.whl /opt/artifact/ && \
    sha256sum /opt/artifact/*.whl > /opt/artifact/biotraj-wheel.sha256 && \
    python --version > /opt/artifact/python-version.txt && \
    gcc --version | head -n 1 > /opt/artifact/gcc-version.txt && \
    ld --version | head -n 1 > /opt/artifact/ld-version.txt && \
    uname -m > /opt/artifact/machine.txt

FROM scratch AS biotraj-wheel-artifact

COPY --from=biotraj-wheel-builder /opt/artifact /biohub-reference-lock

FROM biotraj-wheel-builder AS biotraj-wheel-evidence

ENTRYPOINT ["/opt/biotraj-build/bin/python", "-m", "tools.remote.biohub_reference_lock", \
    "write-build-evidence", "--root", "/opt/contract-root", \
    "--contract", "/opt/contract-root/docker/constraints/biohub-reference-lock.json", \
    "--wheel", "/opt/artifact/biotraj-1.2.2-cp312-cp312-linux_aarch64.whl", \
    "--output", "/output/biohub-reference-build-evidence.json"]
