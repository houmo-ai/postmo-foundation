ARG DOCKER_REGISTRY=registry.example.invalid
FROM ${DOCKER_REGISTRY}/application_software/hlie.cpp-base-xh1:v2.4.2-x86_64

COPY ./tmp/bin /opt/HLIECpp/bin
COPY ./tmp/lib/* /opt/HLIECpp/lib/
COPY ./tmp/config /opt/HLIECpp/config

CMD ["./bin/HLIECpp"]
