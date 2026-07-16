# Install uv
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Change the working directory to the `app` directory
WORKDIR /app


# Copy the project into the image
ADD . /app

# Build the environment from the destination's mounted uv cache at startup
# rather than baking .venv into the image. Baking it produces a ~7GB image
# with no caching opportunity: it gets rebuilt and shipped dozens of times a
# day, and every push transfers the whole thing. Instead the deploy target
# keeps a bind-mounted uv cache, so `uv run` here only materializes the venv
# from already-cached wheels -- new/changed deps download, everything else is
# already local. UV_LINK_MODE=copy avoids hardlink warnings across the
# cache/venv filesystem boundary.
ENV UV_LINK_MODE=copy
CMD ["uv", "run", "--locked", "sleeper"]
