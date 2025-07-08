# Final stage: Create minimal runtime image
FROM mambaorg/micromamba:debian-slim as build-base
USER root
RUN apt-get update \
    && apt-get install -y build-essential cmake cppcheck valgrind clang lldb llvm gdb \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN micromamba install -y -n base -c conda-forge -c bioconda -c defaults \
    openmm \
    pdbfixer

ENV PATH="/opt/conda/bin:$PATH"
ENV CONDA_DEFAULT_ENV=base
ENV CONDA_PREFIX=/opt/conda
ENV UV_PYTHON="/opt/conda/bin/python"

FROM build-base as devcontainer
RUN groupadd devcontainer --gid 1000 \
    && useradd --uid 1000 --gid 1000 -m -s /bin/bash devcontainer \
    && pip install nox uv \
    && chown -R devcontainer:devcontainer /opt/conda

# The actual build - via a script called "your-build-script-goes-here.sh" that copies into a folder called "/app"
FROM build-base as build 
COPY . /app
RUN cd /app && pip install nox uv && uv pip install . && cp /app/.venv/bin/{ABodyBuilder2,TCRBuilder2,NanoBodyBuilder2} /usr/local/bin/

FROM build as production
# Create a wrapper script to handle different commands
COPY <<EOF /usr/local/bin/entrypoint.sh
#!/bin/bash
set -e

# If first argument is one of our known commands, use it
if [[ "$1" == "ABodyBuilder2" || "$1" == "TCRBuilder2" || "$1" == "NanoBodyBuilder2" ]]; then
    exec "$@"
else
    # Default to ABodyBuilder2 with all arguments
    exec ABodyBuilder2 "$@"
fi
EOF

RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD []