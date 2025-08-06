#!/bin/bash

CONFIG="$HOME/.tmux.conf"

# Lines to add
MOUSE_LINE="set -g mouse on"
VIM_MODE_LINE="setw -g mode-keys vi"

# Add settings if not already in the config
grep -qxF "$MOUSE_LINE" "$CONFIG" 2>/dev/null || echo "$MOUSE_LINE" >> "$CONFIG"
grep -qxF "$VIM_MODE_LINE" "$CONFIG" 2>/dev/null || echo "$VIM_MODE_LINE" >> "$CONFIG"

echo "✅ Mouse mode and vim-style copy mode added to ~/.tmux.conf"

# Reload tmux config if inside a tmux session
if [ -n "$TMUX" ]; then
    tmux source-file "$CONFIG"
    echo "🔄 Reloaded tmux config"
else
    echo "💡 Start tmux and run: tmux source-file ~/.tmux.conf"
fi
tmux source-file ~/.tmux.conf