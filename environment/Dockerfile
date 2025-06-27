# hash:sha256:f60ca979177f0410b346df229e8ffaeb666b615ad63c43e2cbc0c86d5a2ce800
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libboost-all-dev=1.71.0.0ubuntu2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    allensdk==2.15.1 \
    k3d==2.16.1 \
    pybind11==2.13.6 \
    trimesh==4.6.9 \
    umap-learn==0.5.7

COPY postInstall /
RUN /postInstall
