#!/usr/bin/env bash
# Build and optionally push the LiteLLM proxy image with streaming retry patch.
#
# Usage:
#   ./docker/build.sh [OPTIONS]
#
# Options:
#   -t, --tag TAG        Image tag (default: litellm-proxy:local)
#   -p, --push           Push image after build
#   -r, --registry REG   Registry prefix, e.g. registry.example.com/myorg
#   --no-cache           Disable Docker build cache
#   -h, --help           Show this help

set -euo pipefail

# ---------- defaults ----------
IMAGE_NAME="litellm-proxy"
IMAGE_TAG="local"
REGISTRY=""
PUSH=false
NO_CACHE=""
DOCKERFILE="docker/Dockerfile.dev"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--tag)       IMAGE_TAG="$2";      shift 2 ;;
    -r|--registry)  REGISTRY="$2";       shift 2 ;;
    -p|--push)      PUSH=true;           shift   ;;
    --no-cache)     NO_CACHE="--no-cache"; shift  ;;
    -h|--help)
      sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ---------- resolve full image name ----------
if [[ -n "$REGISTRY" ]]; then
  FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
else
  FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
fi

# ---------- sync frontend ----------
UI_OUT="$REPO_ROOT/ui/litellm-dashboard/out"
PROXY_OUT="$REPO_ROOT/litellm/proxy/_experimental/out"

if [ -d "$UI_OUT" ] && [ "$(ls -A "$UI_OUT" 2>/dev/null)" ]; then
  echo "==> Syncing frontend build to proxy directory..."
  rm -rf "$PROXY_OUT"/*
  cp -r "$UI_OUT"/* "$PROXY_OUT"/
  echo "    Frontend synced: $(ls "$PROXY_OUT" | wc -l) items"
else
  echo "==> WARNING: No frontend build found at $UI_OUT"
  echo "    Run 'cd ui/litellm-dashboard && npm run build' first"
fi

# ---------- build ----------
echo "==> Building image: $FULL_IMAGE"
echo "    Dockerfile : $DOCKERFILE"
echo "    Context    : $REPO_ROOT"
echo "    Git branch : $(git -C "$REPO_ROOT" branch --show-current 2>/dev/null || echo 'unknown')"
echo "    Git commit : $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo ""

docker build \
  $NO_CACHE \
  --file "$REPO_ROOT/$DOCKERFILE" \
  --tag "$FULL_IMAGE" \
  --label "git.branch=$(git -C "$REPO_ROOT" branch --show-current 2>/dev/null || echo unknown)" \
  --label "git.commit=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)" \
  --label "build.date=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$REPO_ROOT"

echo ""
echo "==> Build complete: $FULL_IMAGE"

# ---------- push ----------
if [[ "$PUSH" == true ]]; then
  echo "==> Pushing $FULL_IMAGE ..."
  docker push "$FULL_IMAGE"
  echo "==> Push complete."
fi
