# Jetson Local Deploy
FROM ubuntu:22.04

# Basic runtime dependencies (fetching web pages, communicating externally)
RUN apt-get update && apt-get install -y ca-certificates curl && rm -rf /var/lib/apt/lists/*

# Copy the locally compiled artifacts (skipping the painful multi-stage Docker build)
COPY target/release-fast/zeroclaw /usr/local/bin/zeroclaw
COPY web/dist /zeroclaw-data/web/dist

# Environment variables matching the default deployment
ENV LANG=C.UTF-8
ENV ZEROCLAW_WORKSPACE=/zeroclaw-data/workspace
ENV HOME=/zeroclaw-data

# Run as UID 1000 so the host-mounted files strictly match the Jetson 'lhzn' host permissions
RUN adduser --disabled-password --gecos "" --uid 1000 lhzn
USER 1000:1000

WORKDIR /zeroclaw-data
EXPOSE 42617

HEALTHCHECK --interval=60s --timeout=10s --retries=3 --start-period=10s \
    CMD ["zeroclaw", "status", "--format=exit-code"]

ENTRYPOINT ["zeroclaw"]
CMD ["daemon"]