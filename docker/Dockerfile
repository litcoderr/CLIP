FROM nvcr.io/nvidia/pytorch:22.09-py3

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt-get -y update
RUN apt-get install -y tzdata
RUN apt-get install -y ffmpeg
RUN apt install -y tmux

COPY requirements.txt ./
RUN pip install -r requirements.txt && rm ./requirements.txt

WORKDIR /CLIP