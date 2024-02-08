FROM python:3.10

RUN apt-get update && apt-get install -y curl

# NVM and NPM are required for Graphite
# Install nvm
# Explicitly set HOME environment variable 
ENV NVM_DIR=$HOME/.nvm
RUN mkdir -p $NVM_DIR
ENV NODE_VERSION=18.2.0

# Install nvm with node and npm
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash \
    && . $NVM_DIR/nvm.sh \
    && nvm install $NODE_VERSION \
    && nvm alias default $NODE_VERSION \
    && nvm use default

ENV NODE_PATH=$NVM_DIR/v$NODE_VERSION/lib/node_modules
ENV PATH=$NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH


# Install graphite (PR stacking), experimental, can be deleted without notice
RUN npm install -g @withgraphite/graphite-cli@stable

# Install lefthook (git hooks, e.g. pre-commit)
RUN curl -1sLf 'https://dl.cloudsmith.io/public/evilmartians/lefthook/setup.deb.sh' | bash
RUN apt install lefthook

# Set the working directory to /app
WORKDIR /app
VOLUME psycop-common

COPY . /app
RUN --mount=type=cache,target=/root/.cache/pip pip install .