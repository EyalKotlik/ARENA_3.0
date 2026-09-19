#!/bin/bash
# Run once per pod, after SSH-ing in: clones repos and configures git + VS Code.
# Safe to re-run. Reads these template env vars (all optional):
#   MY_REPO=owner/repo      your private work repo on GitHub
#   GITHUB_TOKEN            fine-grained token with access to MY_REPO only
#   GIT_NAME, GIT_EMAIL     commit identity
set -euo pipefail
source /etc/profile.d/20-pod-env.sh 2>/dev/null || true
cd /root

ARENA_BASE_COMMIT=eb82e92   # commit whose requirements.txt this image was built from

# --- git identity + token auth (token stays in the env var, never written to disk) ---
[ -n "${GIT_NAME:-}" ]  && git config --global user.name  "$GIT_NAME"
[ -n "${GIT_EMAIL:-}" ] && git config --global user.email "$GIT_EMAIL"
if [ -n "${GITHUB_TOKEN:-}" ]; then
    git config --global credential.https://github.com.helper \
        '!f() { test "$1" = get && echo username=x-access-token && echo "password=$GITHUB_TOKEN"; }; f'
fi

# --- ARENA ---
if [ ! -d ARENA_materials ]; then
    git clone --filter=blob:none https://github.com/ARENA-education/ARENA_materials.git
else
    git -C ARENA_materials pull --ff-only
fi
if ! git -C ARENA_materials diff --quiet "$ARENA_BASE_COMMIT" HEAD -- requirements.txt 2>/dev/null; then
    echo "NOTE: ARENA's requirements.txt changed since this image was built."
    echo "      Probably fine - rebuild the image if you hit import errors."
fi

# --- your repo ---
if [ -n "${MY_REPO:-}" ]; then
    name=$(basename "$MY_REPO" .git)
    if [ ! -d "$name" ]; then
        git clone "https://github.com/${MY_REPO}.git"
    else
        git -C "$name" pull --ff-only
    fi
    # Your own symlink step, if you keep one in the repo.
    [ -x "$name/setup-links.sh" ] && (cd "$name" && ./setup-links.sh)
fi

# --- VS Code: interpreter + import paths for the exercise folders ---
mkdir -p /root/.vscode
cat > /root/.vscode/settings.json <<JSON
{
    "python.defaultInterpreterPath": "/opt/venv/bin/python",
    "python.analysis.extraPaths": [
        "/root/ARENA_materials/chapter0_fundamentals/exercises",
        "/root/ARENA_materials/chapter1_transformer_interp/exercises",
        "/root/ARENA_materials/chapter2_rl/exercises",
        "/root/ARENA_materials/chapter3_llm_evals/exercises",
        "/root/ARENA_materials/chapter4_alignment_science/exercises"
    ]
}
JSON

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "Done. Open /root in VS Code (Remote-SSH)."
