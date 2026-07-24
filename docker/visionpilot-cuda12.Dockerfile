# VisionPilot GPU image for CUDA 12 hosts.
#
# The verified VisionPilot release uses a custom CUDA 13 ONNX Runtime archive.
# This integration image instead uses the official ONNX Runtime 1.22.0 CUDA 12
# archive supplied as docker/ort.cuda12.tgz by setup_visionpilot_a100.sh.

# syntax=docker/dockerfile:1
ARG CUDA_TAG=12.8.1-devel-ubuntu24.04
ARG ENABLE_ROS2=OFF

FROM nvcr.io/nvidia/cuda:${CUDA_TAG} AS builder
ARG ENABLE_ROS2
ARG BUILD_JOBS=8
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y universe \
    && apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget ca-certificates gnupg \
        python3 python3-pip \
        libopencv-dev \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer-plugins-bad1.0-dev \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-nice \
        libnice-dev \
        libsrtp2-dev \
        libboost-system-dev \
        nlohmann-json3-dev \
        coinor-libipopt-dev \
        libcppad-dev \
        liblapack-dev \
        libblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN hdr_dir="$(dirname "$(find /usr/include -name IpIpoptApplication.hpp | head -1)")" \
    && [ -n "$hdr_dir" ] \
    && ln -sf "$hdr_dir" /usr/include/coin-or

RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple \
        --trusted-host pypi.tuna.tsinghua.edu.cn \
        --timeout 120 --retries 10 \
        --no-cache-dir --break-system-packages \
        opencv-python numpy

RUN if [ "$ENABLE_ROS2" = "ON" ]; then \
        apt-get update && apt-get install -y --no-install-recommends curl \
        && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
            -o /usr/share/keyrings/ros-archive-keyring.gpg \
        && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
            > /etc/apt/sources.list.d/ros2.list \
        && apt-get update && apt-get install -y --no-install-recommends \
            ros-jazzy-ros-base \
            ros-jazzy-cv-bridge \
            ros-jazzy-image-transport \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# Official ONNX Runtime 1.22.0 GPU binaries require CUDA 12 and cuDNN 9.
# TensorRT packages are intentionally omitted because the verified configs use
# engine.provider = cuda, not engine.provider = tensorrt.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
COPY docker/ort.cuda12.tgz /opt/ort.tgz
RUN mkdir -p /usr/share/visionpilot /opt/ort_extract \
    && test -s /opt/ort.tgz \
    && tar -xzf /opt/ort.tgz -C /opt/ort_extract \
    && mv /opt/ort_extract/*/ /usr/share/visionpilot/onnxruntime \
    && rm -rf /opt/ort.tgz /opt/ort_extract

WORKDIR /opt/visionpilot
COPY . .

RUN mkdir build && cd build \
    && if [ "$ENABLE_ROS2" = "ON" ]; then . /opt/ros/jazzy/setup.bash; fi \
    && cmake -DONNXRUNTIME_ROOT=/usr/share/visionpilot/onnxruntime -DGPU=ON \
             -DENABLE_ROS2_INTERFACE=${ENABLE_ROS2} \
             -DCMAKE_CXX_STANDARD_LIBRARIES="-lcppad_lib" .. \
    && make -j"$BUILD_JOBS" VisionPilot

FROM nvcr.io/nvidia/cuda:${CUDA_TAG%-devel*}-runtime-ubuntu24.04 AS runtime
ARG ENABLE_ROS2
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y universe \
    && apt-get update && apt-get install -y --no-install-recommends \
        libopencv-dev \
        libgstreamer1.0-0 \
        libgstreamer-plugins-base1.0-0 \
        libgstreamer-plugins-bad1.0-0 \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-nice \
        libnice10 \
        libsrtp2-1 \
        libboost-system1.83.0 \
        libcudnn9-cuda-12 \
        coinor-libipopt-dev \
        liblapack3 \
        libblas3 \
        libcppad-lib1456.0t64 \
    && rm -rf /var/lib/apt/lists/*

RUN if [ "$ENABLE_ROS2" = "ON" ]; then \
        apt-get update && apt-get install -y --no-install-recommends curl \
        && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
            -o /usr/share/keyrings/ros-archive-keyring.gpg \
        && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
            > /etc/apt/sources.list.d/ros2.list \
        && apt-get update && apt-get install -y --no-install-recommends \
            ros-jazzy-ros-base \
            ros-jazzy-cv-bridge \
            ros-jazzy-image-transport \
        && rm -rf /var/lib/apt/lists/*; \
    fi

ENV LD_LIBRARY_PATH="/opt/ros/jazzy/lib:${LD_LIBRARY_PATH}"

RUN mkdir -p /usr/share/visionpilot

COPY --from=builder /usr/share/visionpilot/onnxruntime /usr/share/visionpilot/onnxruntime
COPY --from=builder /opt/visionpilot/build/VisionPilot /usr/bin/VisionPilot
COPY --from=builder /opt/visionpilot/build/config /usr/share/visionpilot/config
COPY --from=builder /opt/visionpilot/assets/icons /usr/share/visionpilot/assets/icons
COPY --from=builder /opt/visionpilot/modules/models/weights /usr/share/visionpilot/modules/models/weights

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics,display

LABEL org.opencontainers.image.title="VisionPilot CUDA 12 A100 integration"
LABEL org.opencontainers.image.source="https://github.com/130070/Alpamayo1.5-VLA"

WORKDIR /usr/share/visionpilot
ENTRYPOINT ["/usr/bin/VisionPilot"]
