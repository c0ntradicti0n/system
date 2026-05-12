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

