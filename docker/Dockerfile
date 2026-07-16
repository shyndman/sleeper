# Install uv. The full python:3.12 image (buildpack-deps) already ships a C
# compiler and build toolchain, so we avoid an apt-get run entirely -- Debian's
# apt mirrors are painfully slow, and the only reason we touched apt was to get
# build-essential for native extension builds. The full image is a larger pull
# than -slim, but it's pulled once and every subsequent build skips apt.
FROM python:3.12
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install ffmpeg from BtbN's static build rather than Debian's apt (whose mirrors
# are painfully slow). This is a statically-linked GPL binary for x86_64 glibc,
# so it needs no runtime libs -- we just extract ffmpeg/ffprobe onto the PATH.
ADD https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz /tmp/ffmpeg.tar.xz
RUN tar --extract --xz --file /tmp/ffmpeg.tar.xz --directory /tmp \
    && mv /tmp/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg /tmp/ffmpeg-master-latest-linux64-gpl/bin/ffprobe /usr/local/bin/ \
    && rm -rf /tmp/ffmpeg.tar.xz /tmp/ffmpeg-master-latest-linux64-gpl

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
