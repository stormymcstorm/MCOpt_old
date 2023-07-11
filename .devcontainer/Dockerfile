FROM ghcr.io/stormymcstorm/ttk-python:py3.11-ttk1.1.0

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
  && apt-get install -y \
    sudo git curl unzip zip wget \
    gcc g++ \
  && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip
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