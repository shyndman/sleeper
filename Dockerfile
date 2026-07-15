# Install uv
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Change the working directory to the `app` directory
WORKDIR /app


# Copy the project into the image
ADD . /app

# Build the environment from the destination's mounted uv cache at startup,
# keeping dependency layers out of the image transferred between machines.
ENV UV_LINK_MODE=copy
CMD ["uv", "run", "--locked", "sleeper"]
