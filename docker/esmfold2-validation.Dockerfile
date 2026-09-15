# Local BF16 parity uses the same dependency environment in isolated processes.
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea AS dependencies
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HOME=/cache/huggingface
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements /requirements
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130 \
    && pip install --no-cache-dir -r /requirements/core.in -r /requirements/features/structure.in \
        -c /requirements/constraints/validation.txt pytest attrs pandas cloudpathlib httpx tenacity \
        zstd scikit-learn boto3 pygtrie dna_features_viewer pydssp ipython
WORKDIR /workspace

FROM dependencies AS candidate
ENV PYTHONPATH=/workspace/src:/workspace
RUN pip install --no-cache-dir -r /requirements/features/train.in -r /requirements/features/dev.in \
    -c /requirements/constraints/validation.txt

FROM dependencies AS reference
COPY --from=upstream_biohub_transformers . /opt/reference/transformers
COPY --from=upstream_biohub_esm . /opt/reference/esm
RUN pip install --no-cache-dir --no-deps /opt/reference/transformers /opt/reference/esm \
    && pip install --no-cache-dir huggingface-hub==0.36.0
ENV PYTHONPATH=/workspace
