#!/usr/bin/env python3
#  esrgan_app.py Copyright (c) 2025 Nikki Cooper
#
#  This program and the accompanying materials are made available under the
#  terms of the GNU Lesser General Public License, version 3.0 which is available at
#  https://www.gnu.org/licenses/gpl-3.0.html#license-text
#
import sys
import esrgan_cli
import esrgan_gui

VERSION="0.5.0"
def main():
    print(
        f"Unifed Esrgan application version: {VERSION}\n\n"
        f"Usage:\n"
        f"Command-Line-Interface: --cli or '--cli --help' for more info.\n"
        f"GUI-Interface: --help or -h for GUI related info.\n"
    )
    argv = sys.argv[1:]  # skip program name

    if "--cli" in argv:
        # Strip --cli and pass everything else untouched to argparse
        argv = [a for a in argv if a != "--cli"]
        # Now argparse inside esrgan_cli sees --help / -h normally
        sys.exit(esrgan_cli.main(argv))
    else:
        # GUI path: pass everything to Qt
        sys.exit(esrgan_gui.main(sys.argv))

if __name__ == "__main__":
    main()
