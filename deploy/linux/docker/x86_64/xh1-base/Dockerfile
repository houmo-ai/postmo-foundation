FROM ubuntu:24.04

RUN apt-get update && apt-get install -y build-essential

WORKDIR /opt/HLIECpp/

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo 'Asia/Shanghai' >/etc/timezone

COPY ./tmp /opt/HLIECpp/

ENV HDPL_PLATFORM=ASIC \
    TCIM_LOG_TAGS=all \
    LD_LIBRARY_PATH=/opt/HLIECpp/lib:$LD_LIBRARY_PATH
