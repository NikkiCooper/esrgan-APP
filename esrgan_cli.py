#  esrgan_cli.py Copyright (c) 2025 Nikki Cooper
#
#  This program and the accompanying materials are made available under the
#  terms of the GNU Lesser General Public License, version 3.0 which is available at
#  https://www.gnu.org/licenses/gpl-3.0.html#license-text
#
import sys
import re
import subprocess
from pathlib import Path
import shutil
from Bcolors import Bcolors
from CLI_User_Setup import *
try:
    from CLI_User_Setup_local import *
except ImportError:
    pass

bc = Bcolors()

global opts
global ROOT_DIR

# ===== FUNCTIONS =====
def run_esrgan_on_file(img: Path, suffix: str):
    """
    Processes an image file using the Real-ESRGAN script.

    This function takes an image file path, runs the Real-ESRGAN upscaling
    script on it with the specified configurations, and saves the processed
    output in a designated directory. The function provides optional
    features like face enhancement and custom output suffix handling.

    :param img: The path to the image file to be processed.
    :type img: Path
    :param suffix: Custom suffix to append to the output file name when saving.
    :type suffix: str
    :return: None
    :rtype: None
    """
    opts.rel_path = img.relative_to(ROOT_DIR)
    output_dir = OUTPUT_ROOT / opts.rel_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", REAL_ESRGAN_SCRIPT,
        "-n", MODEL_NAME,
        "-i", str(img),
        "-o", str(output_dir),
        "--outscale", OUTSCALE,
        "--gpu-id", GPU_ID,
        "--ext", EXT,
        "--tile", TILE,
        "--tile_pad", TILE_PAD
    ]

    if opts.suffix:
        cmd.extend(["--suffix", suffix])

    if opts.face_enhance:
        cmd.append("--face_enhance")

    print(f"🖼{bc.BOLD}{bc.Light_Yellow_f} Processing{bc.Green_f} {img.name}{bc.Light_Yellow_f} → {bc.Light_Blue_f}{output_dir}{bc.RESET}")
    subprocess.run(cmd, check=True)

def run_esrgan_on_folder(input_dir: Path, suffix: str):
    """
    Runs the ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks) algorithm
    on all image files with specified extensions in a given folder. Supported file
    extensions include .jpg, .jpeg, and .png. The function creates an output directory
    relative to a pre-defined root directory and processes each valid image file, applying
    the ESRGAN algorithm with the specified suffix.

    :param input_dir: The path to the directory containing input image files.
    :type input_dir: Path
    :param suffix: A string suffix to be added to processed files.
    :type suffix: str
    :return: The relative path of the input directory to the root directory.
    :rtype: Path
    """
    exts = {".jpg", ".jpeg", ".png"}
    opts.rel_path = input_dir.relative_to(ROOT_DIR)
    output_dir = OUTPUT_ROOT / opts.rel_path
    output_dir.mkdir(parents=True, exist_ok=True)

    for img in sorted(input_dir.iterdir()):
        if img.suffix.lower() in exts:
            run_esrgan_on_file(img, suffix)

    return opts.rel_path  # return relative path for patching

def find_image_dirs(parent_dir):
    """
    Recursively finds subdirectories within a given directory that contain
    files with specific image file extensions. The function searches for
    image extensions such as '.jpg', '.jpeg', and '.png' and adds the
    subdirectories containing such images to the returned list.

    :param parent_dir: The path to the top-level directory where subdirectory
                       search should be initiated.
    :type parent_dir: str
    :return: A list of relative paths for subdirectories that contain image
             files.
    :rtype: List[Path]
    """
    parent = Path(parent_dir)
    image_exts = {".jpg", ".jpeg", ".png"}

    result = []
    for subdir in parent.iterdir():
        if subdir.is_dir():
            # Check if any file in this directory has an image extension
            if any(f.suffix.lower() in image_exts for f in subdir.iterdir() if f.is_file()):
                result.append(subdir.relative_to(parent))

    return result

def get_sets_to_process(model_dir: Path, sets_arg):
    """
    Determines the directories to process based on the provided model directory and sets argument.

    This function processes directories under the given model directory, filtering based
    on the `sets_arg` parameter. If no `sets_arg` is provided, the function will default
    to processing the `model_dir` itself. If `sets_arg` is `["*"]`, it processes all
    subdirectories of `model_dir` that contain images with extensions `.jpg`, `.jpeg`, or `.png`.
    If a specific list of sets is provided in `sets_arg`, it limits processing to those
    subdirectories identified by the set names.

    :param model_dir: The root directory containing subdirectories to possibly process.
    :type model_dir: Path
    :param sets_arg: List of sets to process; may be empty, contain `["*"]` for all image
        containing subdirectories, or a specific list of set names.
    :type sets_arg: list[str]
    :return: A list of directories to be processed.
    :rtype: list[Path]
    """
    image_exts = {".jpg", ".jpeg", ".png"}

    if not model_dir.is_dir():
        raise ValueError(f"{model_dir} is not a valid model directory")

    if not sets_arg:
        # No --sets provided: just process the model_dir itself
        return [model_dir]

    if sets_arg == ["*"]:
        # Process all subdirectories that contain images
        return [
            s for s in sorted(model_dir.iterdir())
            if s.is_dir() and any(f.suffix.lower() in image_exts for f in s.iterdir() if f.is_file())
        ]

    # Otherwise, process only the named sets
    return [
        model_dir / set_name
        for set_name in sets_arg
        if (model_dir / set_name).is_dir()
    ]

def unique_sets():
    """Remove duplicates from --sets while preserving order."""
    if opts.sets is None:
        return None
    seen = set()
    unique = []
    for s in opts.sets:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    opts.sets = unique

def sets_exist():
    sets = []
    if opts.sets is None:
        return

    if "*" in opts.sets:
        return

    absolute_input = ROOT_DIR / opts.Path

    for set_name in sorted(opts.sets):
        if (absolute_input/set_name).is_dir():
            sets.append(set_name)

    opts.sets = sets

def print_options():
    """
    Prints formatted options information based on the settings of the `opts` object.

    The function evaluates various options and formats their output for display.
    Options such as `--root`, `--root_preset`, `--Path`, `--Files`, and others
    are mutually exclusive and displayed based on their presence or specified
    values. It includes default and custom values handled by the program during
    execution.

    :raises AttributeError: If `opts` object or its attributes are missing or invalid.
    """
    print()
    # First check --root and --root_preset to see which one was used.
    # These are mutually exclusive, so only one will be set to None
    if opts.root is not None:
        print(f"{bc.Light_Blue_f}--root:{bc.Green_f} {opts.root}")
    else:
        print(f"{bc.Light_Blue_f}--root_preset:{bc.Green_f} {opts.root_preset}")

    # Check opts.Path to see if --Path was specified
    # --Path and --Files are mutually exclusive
    if opts.Path is not None:
        print(f"{bc.Light_Blue_f}--Path:{bc.Green_f} {opts.Path}")
    else:
        # Otherwise, --Files must have been specified instead
        print(f"{bc.Light_Blue_f}--Files:{bc.Green_f} {opts.Files}")

    # Check if --sets was specified
    # --sets defaults to None if not specified on the CLI
    if opts.sets is not None:
        print(f"{bc.Light_Blue_f}--sets:{bc.Green_f} {opts.sets}")

    # --suffix defaults to ""
    if len(opts.suffix) > 0:
        print(f"{bc.Light_Blue_f}--suffix:{bc.Green_f} {opts.suffix}")

    print(f"{bc.Light_Blue_f}--ext: {bc.Green_f}{opts.ext}")
    print(f"{bc.Light_Blue_f}--model:  {bc.Green_f}{opts.model}")
    print(f"{bc.Light_Blue_f}--face_enhance: {bc.White_f}{'Enabled' if opts.face_enhance else 'Disabled'}")
    print(f"{bc.Light_Blue_f}--tile:{bc.Green_f} {opts.tile}")
    print(f"{bc.Light_Blue_f}--tile_pad:{bc.Green_f} {opts.tile_pad}")
    print(f"{bc.Light_Blue_f}--outscale:{bc.Green_f} {opts.outscale}")
    print(f"{bc.Light_Blue_f}--gpu_id:{bc.Green_f} {opts.gpu_id}")
    print()

def main(argv=None):

    global opts
    global REAL_ESRGAN_SCRIPT, ROOT_PRESETS, OUTPUT_ROOT, ROOT_DIR
    global MODEL_NAME, OUTSCALE, TILE, TILE_PAD, GPU_ID, EXTROOT_DIR, MODEL_NAME, OUTSCALE, TILE, TILE_PAD, GPU_ID, EXT

    from cmdLineOpts import cmdLineOptions
    if argv is None:
        argv = sys.argv[1:]

    opts = cmdLineOptions(argv, ROOT_PRESETS)

    if opts.root_overide:
        ROOT_DIR = opts.root_preset
    else:
        ROOT_DIR = Path(opts.root)

    # Model and processing parameters
    match opts.model_val_int:
        case 0:
            MODEL_NAME = "realesr-general-x4v3"
        case 1:
            MODEL_NAME = "RealESRGAN_x4plus"
        case 2:
            MODEL_NAME = "RealESRNet_x4plus"
        case 3:
            MODEL_NAME = "RealESRGAN_x2plus"
        case 4:
            MODEL_NAME = "RealESRGAN_x4plus_anime_6B"

    OUTSCALE = str(opts.outscale)
    TILE = str(opts.tile)
    TILE_PAD = str(opts.tile_pad)
    GPU_ID = str(opts.gpu_id)
    EXT = str(opts.ext)

    # if opts.sets was set (--sets).
    # check the user supplied sets for uniqueness
    # unique_sets() will remove duplicate sets from the --sets string.
    unique_sets()
    sets_exist()
    print_options()

    # Folder mode
    if opts.Path:
        abs_input = ROOT_DIR / opts.Path
        if not abs_input.exists():
            print(f"❌ Invalid path: {abs_input}")
            sys.exit(1)

        # Handle --sets if provided
        if opts.sets:
            sets_to_process = get_sets_to_process(abs_input, opts.sets)
        else:
            sets_to_process = [abs_input]

        for setdir in sets_to_process:
            opts.rel_path = run_esrgan_on_folder(setdir, opts.suffix)
            # print(f"opts.rel_path: {opts.rel_path}, len(opts.rel_path.parts) = {len(opts.rel_path.parts)}")
            if len(opts.rel_path.parts) == 3:
                # rel_path looks like model/setnum/image. No Studio
                model, setnum, image =  opts.rel_path.parts[-3:]
                studio = None
            else:
                # Now we know rel_path looks like studio/model/setnum/image
                studio, model, setnum = opts.rel_path.parts[-4:-1]
            #patch_imagination(model, setnum, studio, opts.suffix)

    # Single file mode
    elif opts.Files:
        for image in opts.Files:
            abs_input = ROOT_DIR / image
            if abs_input.is_file():
                run_esrgan_on_file(abs_input, opts.suffix)
                print(f"ℹ️{bc.BOLD}{bc.Green_f} Single file processed —{bc.Light_Yellow_f} no slideshow patching performed.{bc.RESET}")

    sys.exit(0)

if __name__ == "__main__":

    main()
