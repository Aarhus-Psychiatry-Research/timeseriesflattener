FROM python:3.9
COPY --from=ghcr.io/astral-sh/uv:0.4.0 /uv /bin/uv

# Set the working directory to /app
WORKDIR /app
VOLUME psycop-common

COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-extras