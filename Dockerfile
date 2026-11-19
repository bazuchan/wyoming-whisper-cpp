FROM python:3.11-slim AS builder
ARG WHISPER_CPP_VULKAN=ON
ARG CMAKE_BUILD_PARALLEL_LEVEL=2
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git libvulkan-dev glslc glslang-tools spirv-headers \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build
COPY pyproject.toml README.md LICENSE.md NOTICE.md ./
COPY wyoming_whisper_cpp ./wyoming_whisper_cpp
RUN if [ "$WHISPER_CPP_VULKAN" = ON ]; then \
      CMAKE_BUILD_PARALLEL_LEVEL="$CMAKE_BUILD_PARALLEL_LEVEL" \
      GGML_VULKAN=1 CMAKE_ARGS="-DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON" \
      pip install --no-cache-dir --no-binary=pywhispercpp pywhispercpp==1.5.1; \
    else pip install --no-cache-dir pywhispercpp==1.5.1; fi \
    && pip install --no-cache-dir '.[hass,vad]'

FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libvulkan1 mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 wyoming \
    && mkdir /data && chown wyoming:wyoming /data
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    WYO_WHISPER_URI=tcp://0.0.0.0:10300 \
    WYO_WHISPER_DATA_DIR=/data \
    WYO_WHISPER_DEVICE=gpu
USER wyoming
EXPOSE 10300
VOLUME /data
HEALTHCHECK --start-period=120s --interval=30s --timeout=10s \
    CMD python -m wyoming_whisper_cpp.health_check
ENTRYPOINT ["python", "-m", "wyoming_whisper_cpp"]
