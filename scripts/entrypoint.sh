#!/bin/bash

if [[ -z "${LOCAL_USER_ID}" ]]; then
  # LOCAL_USER_ID not set, proceed as root.
  exec bash -c "$@"
else
  # Add local user with specified UID and GID.
  USER_ID=${LOCAL_USER_ID}
  GROUP_ID=${LOCAL_GROUP_ID}

  echo "Starting with UID: $USER_ID, GID: $GROUP_ID"
  groupadd -g "$GROUP_ID" host
  useradd --shell /bin/bash -u "$USER_ID" -g host -o -c "" -m user
  export HOME=/home/user
  chown -R user /usr/src/

  # Set a limit on the number of processes one can run (ulimit), before running user scripts.
  #
  # Note: Using $@ instead of $* when embedded in another word would result in unexpected behavior:
  # the first field is merged with the string, and the rest left out as separate fields.
  # If the command is `chmod +x ./configure`, it would become:
  # >  "ulimit -u 1024; chmod" "+x" "./configure"
  exec /usr/local/bin/gosu user bash -c "ulimit -u 1024; $*"
fi
