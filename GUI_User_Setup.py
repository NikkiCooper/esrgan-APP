#  GUI_User_Setup.py Copyright (c) 2025 Nikki Cooper
#
#  This program and the accompanying materials are made available under the
#  terms of the GNU Lesser General Public License, version 3.0 which is available at
#  https://www.gnu.org/licenses/gpl-3.0.html#license-text
#
from pathlib import Path
# Constants
# Specify the path to your Real-ESARGAN inference script
REAL_ESRGAN_SCRIPT = "/path_to_local_Real-ESRGAN-Repo/src/inference_realesrgan.py"
# Specify the default image output ROOT directory
DEFAULT_OUTPUT_DIR = Path("~/AI_IMAGES").expanduser()
# Specify the default input image Root directory
DEFAULT_ROOT_DIR = Path("~/Photos").expanduser()
