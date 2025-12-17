# hash:sha256:da844bb57e48c980993fab3f2504dddcfac39d2676e9b2ab5ad4a18e5070b36e
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    pip==25.3
RUN pip install -U --no-cache-dir \
    allensdk==2.15.1 \
    k3d==2.16.1 \
    matplotlib==3.4.2 \
    pandas==1.5.3 \
    s3fs==2021.11.0 \
    scikit-image==0.24.0 \
    scipy==1.13.1 \
    zarr==2.18.2

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
