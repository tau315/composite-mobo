#!/bin/sh
# Run a command on the Unicorn cluster over SSH key authentication.
#
#   scripts/unicorn.sh 'sinfo'
#   git bundle create - HEAD | scripts/unicorn.sh 'cat > ~/repo.bundle'
#
# Wraps the remote command in a LOGIN shell (bash -lc) because a non-interactive
# SSH session does not have Slurm (/usr/local/slurm/current/bin) or conda on
# PATH. Single-quotes in the command are escaped so arbitrary input survives
# intact, and stdin is left free so the wrapper can be piped into.
#
# Password authentication is refused rather than silently fallen back to, so a
# missing or revoked key fails loudly instead of hanging on a prompt. Only the
# private key lives outside the repository, which is where a credential belongs.
#
# Override any of these from the environment:
UNICORN_HOST="${UNICORN_HOST:-unicorn-login-01.coecis.cornell.edu}"
UNICORN_USER="${UNICORN_USER:-sg2636}"
UNICORN_KEY="${UNICORN_KEY:-$HOME/.ssh/unicorn_key}"

if [ "$#" -eq 0 ]; then
    echo "usage: $0 '<remote command>'" >&2
    exit 2
fi
if [ ! -f "$UNICORN_KEY" ]; then
    echo "No SSH key at $UNICORN_KEY." >&2
    echo "Generate one and install it with:" >&2
    echo "  ssh-keygen -t ed25519 -f $UNICORN_KEY -C unicorn-$UNICORN_USER" >&2
    echo "  ssh-copy-id -i $UNICORN_KEY.pub $UNICORN_USER@$UNICORN_HOST" >&2
    exit 3
fi

escaped=$(printf "%s" "$*" | sed "s/'/'\\\\''/g")
exec ssh -i "$UNICORN_KEY" -o IdentitiesOnly=yes \
    -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no \
    -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o LogLevel=ERROR \
    "$UNICORN_USER@$UNICORN_HOST" "bash -lc '$escaped'"
