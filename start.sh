#!/bin/bash
# Container entrypoint: set up SSH access, then stay alive.
set -euo pipefail

# RunPod passes the SSH public keys from your account settings in $PUBLIC_KEY.
# Locally, pass it yourself: docker run -e PUBLIC_KEY="$(cat ~/.ssh/id_ed25519.pub)" ...
mkdir -p /root/.ssh && chmod 700 /root/.ssh
if [ -n "${PUBLIC_KEY:-}" ]; then
    echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
else
    echo "WARNING: PUBLIC_KEY is empty - you won't be able to log in over SSH."
fi

# Template env vars (HF_TOKEN, API keys, GITHUB_TOKEN, ...) are only visible to this
# process, not to SSH sessions. Pass through the ones worth having, quoted safely.
: > /etc/profile.d/20-pod-env.sh
for var in $(compgen -e); do
    case "$var" in
        RUNPOD_*|HF_*|HUGGING_FACE_*|OPENAI_*|ANTHROPIC_*|GITHUB_*|GIT_*|WANDB_*|MY_REPO|TOGETHER_*)
            printf 'export %s=%q\n' "$var" "${!var}" >> /etc/profile.d/20-pod-env.sh ;;
    esac
done
chmod 600 /etc/profile.d/20-pod-env.sh

# Host keys are generated per container, never baked into the image.
ssh-keygen -A >/dev/null
/usr/sbin/sshd -o PasswordAuthentication=no -o PermitRootLogin=prohibit-password

echo "Ready. SSH in, then run: arena-session"
sleep infinity
