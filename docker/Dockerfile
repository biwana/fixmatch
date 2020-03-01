FROM tensorflow/tensorflow:1.10.0-gpu-py3 
LABEL maintainer="Brian Kenji Iwana"

RUN apt-get update \
    && apt-get install -y imagemagick \
    && apt-get install -y vim \
    && DEBIAN_FRONTEND=noniteractive apt-get install -y python3-tk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install scikit-image==0.15.0
RUN pip install tqdm
RUN pip install cython
RUN pip install easydict
RUN pip install absl-py
RUN pip install LibML
