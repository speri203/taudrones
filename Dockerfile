ARG REGISTRY
ARG TAG
FROM supervisely/base-tensorflow:latest

RUN pip install keras==2.2.4

############### copy code ###############
ARG MODULE_PATH=.
COPY . /workdir
COPY supervisely_lib /workdir/supervisely_lib

ENV PYTHONPATH /workdir:/workdir/src:/workdir/supervisely_lib/worker_proto:$PYTHONPATH
WORKDIR /workdir/src
