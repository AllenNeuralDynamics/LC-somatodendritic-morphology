# hash:sha256:a856dbf8b204b5d0e11f6effb82e0314528dc55ef61ec683b096d9d7d1bdc8dd
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1=1.4.0-1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir --upgrade-strategy=only-if-needed \
    allensdk==2.16.2 \
    brainrender==2.1.16 \
    colorcet==3.1.0 \
    k3d==2.17.0 \
    matplotlib==3.10.8 \
    neuron-morphology==1.2.2 \
    pandas==1.5.3 \
    s3fs==2026.2.0 \
    scikit-image==0.24.0 \
    scipy==1.10.1 \
    svgutils==0.3.4 \
    zarr==2.18.2

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
