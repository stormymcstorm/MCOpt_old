FROM ubuntu

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
  && apt-get install -y \
    wget curl unzip zip nano sudo less man-db manpages manpages-dev git \
  && wget -nv \
    https://github.com/topology-tool-kit/ttk/releases/download/1.1.0/ttk-1.1.0-ubuntu-22.04.deb \
    -O /tmp/ttk.deb \
  && wget -nv \
    https://github.com/topology-tool-kit/ttk-paraview/releases/download/v5.10.1/ttk-paraview-v5.10.1-ubuntu-22.04.deb \
    -O /tmp/ttk-paraview.deb \
  && apt-get install -y \
    /tmp/ttk.deb /tmp/ttk-paraview.deb \
    python3-pip \
    python3-venv \
  && rm /tmp/ttk.deb /tmp/ttk-paraview.deb \
  && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid ${USER_GID} ${USERNAME} \
  && useradd -s /bin/bash --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/${USERNAME}

RUN python3 -m venv /home/${USERNAME}/.venv --system-site-packages \
  && echo "source ~/.venv/bin/activate" >> /home/${USERNAME}/.bashrc