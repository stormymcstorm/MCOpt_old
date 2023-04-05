FROM ubuntu

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
  && apt-get install -y \
    wget curl unzip zip nano sudo less man-db manpages manpages-dev git \
    cmake mesa-common-dev mesa-utils freeglut3-dev libgl1-mesa-dev libgl1-mesa-glx \
  && wget -nv \
    https://github.com/topology-tool-kit/ttk/releases/download/1.1.0/ttk-1.1.0-ubuntu-22.04.deb \
    -O /tmp/ttk.deb \
  && wget -nv \
    https://github.com/topology-tool-kit/ttk-paraview/releases/download/v5.10.1/ttk-paraview-v5.10.1-ubuntu-22.04.deb \
    -O /tmp/ttk-paraview.deb \
  && apt-get install -y \
    /tmp/ttk.deb /tmp/ttk-paraview.deb \
    python3-venv \
    python3-pip \
  && rm /tmp/ttk.deb /tmp/ttk-paraview.deb \
  && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN apt purge -y python3-setuptools 

# Workaround to get ttk to use it's own vtk installation while still allowing 
# for the pypi version of vtk to be installed
# RUN cd /usr/lib/python3/dist-packages/topologytoolkit \
#   && cp __init__.py __init__.py.back \
#   && sed -i '/from __future__ import absolute_import/a import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))' __init__.py \
#   && echo "sys.path.pop(0)" >> __init__.py

ENV PYTHONPATH /usr/lib/python3/dist-packages

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid ${USER_GID} ${USERNAME} \
  && useradd -s /bin/bash --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/${USERNAME}

RUN python3 -m venv /home/${USERNAME}/.venv --system-site-packages \
  && chown -R ${USERNAME} /home/${USERNAME}/.venv \
  && echo "source ~/.venv/bin/activate" >> /home/${USERNAME}/.bashrc

COPY requirements.txt /tmp/requirements.txt

USER ${USERNAME}

RUN pip install -r /tmp/requirements.txt