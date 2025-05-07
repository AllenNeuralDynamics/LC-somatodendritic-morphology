# hash:sha256:953d7c4cdd65dc87c89382e21701b0f8909a0e7f0aaa323c7664be99639fcfb7
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
    pybind11==2.13.6

COPY postInstall /
RUN /postInstall
