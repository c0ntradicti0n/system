set export := true
set dotenv-required := true
set dotenv-load := true

mod kube

nix *cmd:
   nix-shell shell.nix --run "just {{cmd}}"

a *args:
    just nix deepseek-claude {{args}}


deepseek-claude *args:
    DEBUG=* \
    CLAUDE_CODE_USER_ID="anonymous" \
    ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic \
    ANTHROPIC_AUTH_TOKEN=$DEEPSEEK_API_KEY \
    API_TIMEOUT_MS=600000 \
    ANTHROPIC_MODEL=deepseek-v4-pro \
    ANTHROPIC_SMALL_FAST_MODEL=deepseek-v4-flash \
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
    npx claude --debug --verbose --model deepseek-v4-flash {{args}}

# wipe chromadb data + restart service (auto-indexes on empty collection)
update-chroma:
    rm -rf .caches/.cache-cr
    docker compose up -d service

# re-index linker chroma
update-chroma-linker:
    rm -rf .caches/.cache-cr-linker
    docker compose up -d linker

# wipe + re-index both
update-chroma-all: update-chroma update-chroma-linker

# translate .md files to .de.md via DeepSeek
translate p="":
    python3 -m lib.translate {{p}} --lang de

# dry-run to preview translation targets
translate-dry p="":
    python3 -m lib.translate {{p}} --lang de --dry-run

start:
    make start