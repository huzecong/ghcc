FROM gcc:latest

# Install packages for compilation.
RUN apt-get update && apt-get install -y --no-install-recommends \
    nasm \
    ca-certificates \
    curl \
    python3-dev
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py

# Credit: https://denibertovic.com/posts/handling-permissions-with-docker-volumes/
# Install `gosu` to avoid running as root.
RUN gpg --keyserver ha.pool.sks-keyservers.net --recv-keys B42F6819007F00F88E364FD4036A9C25BF357DD4
RUN curl -o /usr/local/bin/gosu -SL "https://github.com/tianon/gosu/releases/download/1.4/gosu-$(dpkg --print-architecture)" \
    && curl -o /usr/local/bin/gosu.asc -SL "https://github.com/tianon/gosu/releases/download/1.4/gosu-$(dpkg --print-architecture).asc" \
    && gpg --verify /usr/local/bin/gosu.asc \
    && rm /usr/local/bin/gosu.asc \
    && chmod +x /usr/local/bin/gosu

# Install Python libraries.
COPY requirements.txt /usr/src/
RUN pip install -r /usr/src/requirements.txt && \
    rm /usr/src/requirements.txt

# Create entrypoint that sets appropriate group and user IDs.
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Copy `ghcc` files into image, and set PYTHONPATH and PATH.
ENV CUSTOM_PATH="/usr/custom"
COPY ghcc/ $CUSTOM_PATH/ghcc/
COPY scripts/ $CUSTOM_PATH/scripts/
ENV PATH="$CUSTOM_PATH/scripts/:$PATH"
ENV PYTHONPATH="$CUSTOM_PATH/:$PYTHONPATH"
