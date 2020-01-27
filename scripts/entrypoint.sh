#!/bin/bash

function __setup_bashrc {
  {
    echo "source $CUSTOM_PATH/.commacd.sh"
    echo "export COMMACD_SEQSTART=1"
    echo "alias ll='ls -alFG'"
  } >> "$HOME/.bashrc"
}

if [[ -z "${LOCAL_USER_ID}" ]]; then
  # LOCAL_USER_ID not set, proceed as root.
  __setup_bashrc
  exec bash -c "$@"
else
  # Add local user with specified UID and GID.
  USER_ID=${LOCAL_USER_ID}
  GROUP_ID=${LOCAL_GROUP_ID}

  echo "Starting with UID: $USER_ID, GID: $GROUP_ID"
  GROUP_NAME=$(getent group "$GROUP_ID" | cut -d: -f1)
  if [[ -z "${GROUP_NAME}" ]]; then
    # The group doesn't exist; create a new one.
    groupadd -g "$GROUP_ID" host
    GROUP_NAME="host"
  fi
  useradd --shell /bin/bash -u "$USER_ID" -g "$GROUP_NAME" -o -c "" -m user
  export HOME=/home/user
  chown -R user /usr/src/
  __setup_bashrc

  # Set a limit on the number of processes one can run (ulimit), before running user scripts.
  #
  # Note: Using $@ instead of $* when embedded in another word would result in unexpected behavior:
  # the first field is merged with the string, and the rest left out as separate fields.
  # If the command is `chmod +x ./configure`, it would become:
  # >  "ulimit -u 256; chmod" "+x" "./configure"
  exec /usr/local/bin/gosu user bash -c "ulimit -u 256; $*"
fi
