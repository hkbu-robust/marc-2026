# The frozen qualifier agent plus the #85 logging diff, and nothing else.
# Weights, dependencies and the base layers are the frozen image's own; only
# the Python sources under marc_agent/ and demo/ are replaced.
#   docker build -f scripts/frozen_logging.Dockerfile -t marc-agent:frozen-logging .
FROM marc-agent:qualifier-1450aa9
COPY marc_agent /agent/marc_agent
COPY demo /agent/demo
