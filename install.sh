#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"

echo "Building scry CLI (release)..."
cargo build --release -p scry-cli

mkdir -p "$INSTALL_DIR"
cp target/release/scry "$INSTALL_DIR/scry"
chmod +x "$INSTALL_DIR/scry"

echo "Installed scry to $INSTALL_DIR/scry"

# Ensure install dir is in PATH
case ":$PATH:" in
  *":$INSTALL_DIR:"*)
    echo "$INSTALL_DIR is already in PATH."
    ;;
  *)
    # Detect the user's shell rc file
    SHELL_NAME="$(basename "$SHELL")"
    case "$SHELL_NAME" in
      zsh)  RC_FILE="$HOME/.zshrc" ;;
      fish) RC_FILE="$HOME/.config/fish/config.fish" ;;
      *)    RC_FILE="$HOME/.bashrc" ;;
    esac

    LINE="export PATH=\"$INSTALL_DIR:\$PATH\""

    # Append to rc file if not already present
    if [ -f "$RC_FILE" ] && grep -qF "$INSTALL_DIR" "$RC_FILE" 2>/dev/null; then
      echo "$INSTALL_DIR already referenced in $RC_FILE."
    else
      echo "" >> "$RC_FILE"
      echo "# Added by scry installer" >> "$RC_FILE"
      echo "$LINE" >> "$RC_FILE"
      echo "Added $INSTALL_DIR to PATH in $RC_FILE"
    fi

    # Also export for the current session
    export PATH="$INSTALL_DIR:$PATH"
    echo "$INSTALL_DIR added to PATH for this session."
    ;;
esac

echo ""
echo "Run 'scry info' to verify the installation."
