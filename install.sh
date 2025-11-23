#!/usr/bin/env bash
# install.sh - Setup script for esrgan-GUI

set -e  # exit immediately if a command fails

# Target base directory
TARGET_DIR="$HOME/.local/share/esrgan-APP"

echo "Installing esrgan-GUI resources into $TARGET_DIR ..."

# Create required directories
mkdir -p "$TARGET_DIR/Resources"
mkdir -p "$TARGET_DIR/Help"

# Copy resources
cp -r Resources/* "$TARGET_DIR/Resources/"
cp Help/help.txt "$TARGET_DIR/Help/"

echo "Installation complete."
echo "Resources copied to $TARGET_DIR/Resources"
echo "Help file copied to $TARGET_DIR/Help"
