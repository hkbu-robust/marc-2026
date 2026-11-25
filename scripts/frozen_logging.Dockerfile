# A frozen agent image with this checkout's Python sources, and nothing else.
# BASE defaults to the frozen final system (final-system-v1); pass
# --build-arg BASE=marc-agent:qualifier-1450aa9 for the qualifier system.
# Weights, dependencies and the base layers are the frozen image's own; only
# the sources under marc_agent/ and demo/ are replaced. Built from the #85 commit
# it is marc-agent:frozen-logging; built from a final-system branch it is that
# branch's smoke-run image (scripts/README.md).
#   docker build -f scripts/frozen_logging.Dockerfile -t marc-agent:frozen-logging .
ARG BASE=robust-agent:finals
FROM ${BASE}
COPY marc_agent /agent/marc_agent
COPY demo /agent/demo
