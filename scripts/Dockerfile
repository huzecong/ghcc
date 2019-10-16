FROM gcc:latest

# Install some packages for compilation.
RUN apt-get update && apt-get install -y --no-install-recommends \
    nasm \
    ca-certificates \
    curl

# Credit: https://denibertovic.com/posts/handling-permissions-with-docker-volumes/
# Install `gosu` to avoid running as root.
RUN gpg --keyserver ha.pool.sks-keyservers.net --recv-keys B42F6819007F00F88E364FD4036A9C25BF357DD4
RUN curl -o /usr/local/bin/gosu -SL "https://github.com/tianon/gosu/releases/download/1.4/gosu-$(dpkg --print-architecture)" \
    && curl -o /usr/local/bin/gosu.asc -SL "https://github.com/tianon/gosu/releases/download/1.4/gosu-$(dpkg --print-architecture).asc" \
    && gpg --verify /usr/local/bin/gosu.asc \
    && rm /usr/local/bin/gosu.asc \
    && chmod +x /usr/local/bin/gosu

# Create entrypoint that sets appropriate group and user IDs.
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Add mock `gcc` and `sudo` to the PATH.
ENV CUSTOM_PATH="/usr/custom/bin/"
RUN mkdir -p $CUSTOM_PATH
COPY mock_path/ $CUSTOM_PATH
ENV PATH="$CUSTOM_PATH:$PATH"
