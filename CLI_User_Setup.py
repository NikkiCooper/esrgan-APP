#  CLI_User_Setup.py Copyright (c) 2025 Nikki Cooper
#
#  This program and the accompanying materials are made available under the
#  terms of the GNU Lesser General Public License, version 3.0 which is available at
#  https://www.gnu.org/licenses/gpl-3.0.html#license-text
#
from pathlib import Path
#
# ===== CONFIG =====
# User specified presets:
_paths = [
    "/home/nikki/+Graphics/+Models",                # Preset_1
    "/home/nikki/+ImagesToPrint",                   # Preset_2
    "/home/nikki/+Graphics/+Anime",                 # Preset_3
    "/home/nikki/+Graphics/+Private",               # Preset_4
    "/home/nikki/+Graphics/+Cartoons",              # Preset_5
    "/home/nikki/+Graphics/Other"                   # Preset_6
]

ROOT_PRESETS = [Path(p) for p in _paths]

# The output root directory where processed images will be saved.
OUTPUT_ROOT = Path("/mnt/raid1/AI_IMAGES")
# Specify the path to your Real-ESRGAN inference script
REAL_ESRGAN_SCRIPT = "/mnt/17E0E95D/Real-ESRGAN/src/inference_realesrgan.py"
