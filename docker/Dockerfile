FROM nvcr.io/nvidia/pytorch:22.10-py3

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y git vim sudo gpustat ffmpeg libsm6 libxext6

COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt