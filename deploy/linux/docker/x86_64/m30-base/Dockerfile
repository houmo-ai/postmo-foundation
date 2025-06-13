FROM ubuntu:24.04

RUN apt-get update && apt-get install -y build-essential

WORKDIR /opt/houmo/bin

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo 'Asia/Shanghai' >/etc/timezone

COPY ./tmp /opt/houmo/

ENV HDPL_PLATFORM=ASIC \
    DEVICE_CHOICE=0
