#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Startup Animation — shell integration
# ═══════════════════════════════════════════════════════════════════
#
# Runs the startup_anim binary when opening a Kitty terminal.
# The animation forks to the background automatically — your shell
# prompt appears immediately and you can type right away.
#
# INSTALLATION:
#
#   1. Build the binary:
#      cargo build --release --example startup_anim
#
#   2. Install it:
#      cp target/release/examples/startup_anim ~/.local/bin/
#
#   3. Source this script in your shell RC:
#      echo 'source /path/to/startup_anim.sh' >> ~/.bashrc
#      # or for zsh:
#      echo 'source /path/to/startup_anim.sh' >> ~/.zshrc
#
# ALTERNATIVE — Kitty session file:
#
#   Add to ~/.config/kitty/startup.session:
#     launch --type=background startup_anim
#
# ═══════════════════════════════════════════════════════════════════

# Only run in Kitty terminal
if [ -z "$KITTY_PID" ]; then
    return 2>/dev/null || exit 0
fi

# Only run in interactive shells (not scripts, not nested)
if [ -z "$PS1" ]; then
    return 2>/dev/null || exit 0
fi

# Find the binary
STARTUP_ANIM=""
if command -v startup_anim &>/dev/null; then
    STARTUP_ANIM="startup_anim"
elif [ -x "$HOME/.local/bin/startup_anim" ]; then
    STARTUP_ANIM="$HOME/.local/bin/startup_anim"
fi

# Run it (it forks to background internally)
if [ -n "$STARTUP_ANIM" ]; then
    "$STARTUP_ANIM"
fi
