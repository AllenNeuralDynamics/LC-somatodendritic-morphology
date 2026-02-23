# hash:sha256:76129fa2abb739ccae574dab8f8385c3d32f86b1d8c9b1806958bf246d5c70a2
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir --upgrade-strategy=only-if-needed \
    allensdk==2.16.2 \
    brainrender==2.1.16 \
    k3d==2.17.0 \
    matplotlib==3.10.8 \
    neuron-morphology==1.2.2 \
    pandas==1.5.3 \
    s3fs==2026.2.0 \
    scikit-image==0.24.0 \
    scipy==1.10.1 \
    zarr==2.18.2

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
