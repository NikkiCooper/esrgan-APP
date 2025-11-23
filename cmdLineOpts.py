#  cmdLineOpts.py Copyright (c) 2025 Nikki Cooper
#
#  This program and the accompanying materials are made available under the
#  terms of the GNU Lesser General Public License, version 3.0 which is available at
#  https://www.gnu.org/licenses/gpl-3.0.html#license-text
#
import os
import sys
import re
import argparse
from typing import List, Tuple, Optional
from functools import partial
from Bcolors import Bcolors

bc = Bcolors()

bc.clear()

def cmdLineOptions(argv, ROOT_PRESETS):
    """
    Parse and process command-line arguments for an ESRGAN-based image restoration and
    upscaling application. The function utilizes argparse to define and parse a range of
    parameters related to root directories, images, models, GPU settings, scaling options,
    and other configurations. The function ensures required arguments are set and enforces
    mutual exclusivity between specific options.

    :param ROOT_PRESET_1: Root directory preset corresponding to `p1`.
    :type ROOT_PRESET_1: str
    :param ROOT_PRESET_2: Root directory preset corresponding to `p2`.
    :type ROOT_PRESET_2: str
    :param ROOT_PRESET_3: Root directory preset corresponding to `p3`.
    :type ROOT_PRESET_3: str
    :param ROOT_PRESET_4: Root directory preset corresponding to `p4`.
    :type ROOT_PRESET_4: str
    :param ROOT_PRESET_5: Root directory preset corresponding to `p5`.
    :type ROOT_PRESET_5: str
    :param ROOT_PRESET_6: Root directory preset corresponding to `p6`.
    :type ROOT_PRESET_6: str
    :return: Parsed command-line arguments incorporating all required and optional settings.
    :rtype: argparse.Namespace
    """
    MODEL_MAPPING= {
        "x4v3":0,
        "x4plus":1,
        "net_x4plus":2,
        "x2plus":3,
        "x4plus_anime_6B":4
    }

    parser = argparse.ArgumentParser(
        description=f"{bc.BOLD}{bc.Blue_f}ESRGAN image restoration + upscale{bc.RESET}",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Root Directory Options
    required_root = parser.add_argument_group(f"{bc.BOLD}{bc.Light_Blue_f}Root Directory Options.  Required.  One of: {bc.RESET}")
    root_required_group = required_root.add_mutually_exclusive_group(required=True)
    root_required_group.add_argument(
        "--root",
        type=partial(validate_user_dirs),
        #type=str,
        default=None,
        help=(
            f"{bc.Light_Yellow_f}The base root directory in which all images reside.\n{bc.RESET}"
            # f"{bc.BOLD}{bc.Magenta_f}ROOT_DIR = {bc.Green_f}{ROOT_DIR}\n{bc.RESET} "
        )
    )
    root_required_group.add_argument(
        "--root_preset",
        type=str,
        choices=["p1", "p2", "p3", "p4", "p5", "p6"],
        help=(
            f"{bc.BOLD}{bc.Light_Yellow_f}Root directory presets.{bc.Light_Blue_f}  One of:\n"
            f"{bc.Cyan_f}p1 = {ROOT_PRESETS[0]}\n"
            f"{bc.White_f}p2 = {ROOT_PRESETS[1]}\n"
            f"{bc.Magenta_f}p3 = {ROOT_PRESETS[2]}\n"
            f"{bc.Light_Blue_f}p4 = {ROOT_PRESETS[3]}\n"
            f"{bc.Green_f}p5 = {ROOT_PRESETS[4]}\n"
            f"{bc.Light_Yellow_f}p6 = {ROOT_PRESETS[5]}"
            f"\n{bc.Magenta_f}Default: {bc.Green_f}None{bc.RESET}"
        )
    )
    # File Required Options Group
    required = parser.add_argument_group(f"{bc.BOLD}{bc.Light_Blue_f}Image Path Options.  Required. One of{bc.RESET}")
    file_required_group = required.add_mutually_exclusive_group(required=True)
    file_required_group.add_argument(
        "--Path",
                                     #type=partial(validate_user_dirs, root_dir=ROOT_DIR),
                                     type=str,
                                     default=None,
                                     help=(
                                         f"{bc.BOLD}{bc.Light_Yellow_f}Image path relative to specified root directory.{bc.Green_f} (See above)\n{bc.RESET}"
                                         #f"{bc.BOLD}{bc.Magenta_f}ROOT_DIR = {bc.Green_f}{ROOT_DIR}\n{bc.RESET} "
                                     )
    )
    file_required_group.add_argument(
        "--Files",
        nargs="+",
        #type=partial(validate_user_files, root_dir=ROOT_DIR),
        type=str,
        default=None,
        help=f"{bc.Light_Yellow_f}List of images to process.{bc.RESET}"
    )

    # Optional File Options Group
    file_group = parser.add_argument_group(f"{bc.BOLD}{bc.Light_Blue_f}Optional File Options{bc.RESET}")
    file_group.add_argument(
        "--sets",
        nargs="*",
        type=str,
        default=None,
        help=(
            f"{bc.Light_Yellow_f}Specify which {bc.White_f}SETS{bc.Light_Yellow_f} to process.\n"
            f"{bc.Light_Blue_f}Format options:\n"
            f" • {bc.Cyan_f}'*'{bc.White_f} = Process {bc.UNDERLINE}{bc.Green_f}ALL{bc.White_f}{bc.RESET_UNDERLINED} sets\n"
            f" • {bc.Cyan_f}'NNN'{bc.White_f} = Single set number (e.g. '001')\n"
            f" • {bc.Cyan_f}'NNN-MMM'{bc.White_f} = Range of sets (e.g. '001-025')\n"
            f" • {bc.Cyan_f}'NNN-'{bc.White_f} = From set NNN to end\n"
            f" Multiple ranges can be specified\n{bc.RESET}"
        )
    )

    file_group.add_argument("--suffix", default="", help=f"{bc.Light_Yellow_f}Optional suffix (e.g. V1, V2).\n{bc.BOLD}{bc.Magenta_f}Default: {bc.Green_f}''\n{bc.RESET}")
    file_group.add_argument("--ext", default="png", help=f"{bc.Light_Yellow_f}Output image extension.\n{bc.BOLD}{bc.Magenta_f}Default: {bc.Green_f}png{bc.RESET}")
    # Models Group
    model_settings = parser.add_argument_group(f"{bc.BOLD}{bc.Light_Blue_f}Model Options{bc.RESET}")
    model_settings.add_argument(
                   "--model",
                                type=str,
                                #choices=["x4v3", "x4plus", "net_x4plus", "x2plus", "x4_plus_anime_6B"],
                                choices=list(MODEL_MAPPING.keys()),
                                default="x4v3",
                                help=(
                                    f"{bc.BOLD}{bc.Light_Blue_f}Model Name Options.  One of: "
                                    f"{bc.BOLD}{bc.Green_f}x4v3, "
                                    f"{bc.Cyan_f}x4plus, "
                                    f"{bc.Yellow_f}net_x4plus, "
                                    f"{bc.Magenta_f}x2plus, "
                                    f"{bc.Blue_f}x4plus_anime_6B "
                                    f"\n{bc.Magenta_f}Default: {bc.Green_f}x4v3\n{bc.RESET}"
                                )
    )
    model_settings.add_argument(
        "--model_help",
        action="store_true",
        help=f"{bc.BOLD}{bc.Light_Yellow_f}Gives more information about each available model name specified above.{bc.RESET}"
    )
    # Enhancement Group
    enhancement_settings = parser.add_argument_group(
        f"{bc.BOLD}{bc.Light_Blue_f}Enhancement Options{bc.RESET}"
    )
    enhancement_settings.add_argument(
        "--face_enhance",
        action="store_true",
        help=f"{bc.BOLD}{bc.Light_Yellow_f}Use{bc.Light_Blue_f} GFPGAN{bc.Light_Yellow_f} to enhance faces after upscaling.{bc.RESET}"
    )
    # Tile Settings
    tile_settings = parser.add_argument_group(f"{bc.BOLD}{bc.Light_Blue_f}Tile Options{bc.RESET}")
    tile_settings.add_argument(
        "--tile",
        type=int,
        default=800,
        help=f"{bc.Light_Yellow_f}Tile size for image processing.\n{bc.BOLD}{bc.Magenta_f}Default: {bc.Green_f}800\n{bc.RESET}"
    )
    tile_settings.add_argument(
        "--tile_pad",
        type=int,
        default=10,
        help=f"{bc.Light_Yellow_f}Tile padding for image processing.\n{bc.BOLD}{bc.Magenta_f}Default: {bc.Green_f}10{bc.RESET}"
    )
    # Scale Settings
    scale_settings = parser.add_argument_group(f"{bc.BOLD}{bc.Light_Blue_f}Scale Options{bc.RESET}")
    scale_settings.add_argument(
        "--outscale",
        type=float,
        default=1.0,
        help=f"{bc.Light_Yellow_f}Output scale factor float (float 0.5, 1, 2, 3.5 ...)\n{bc.BOLD}{bc.Magenta_f}Default: {bc.Green_f}1.0{bc.White_f} (do not upscale){bc.RESET}"
    )
    # GPU Settings
    gpu_settings = parser.add_argument_group(f"{bc.BOLD}{bc.Light_Blue_f}GPU Options{bc.RESET}")
    gpu_settings.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help=f"{bc.Light_Yellow_f}GPU ID to use for processing.\n{bc.BOLD}{bc.Magenta_f}Default: {bc.Green_f}0{bc.RESET}"
    )

    args = parser.parse_args(argv)

    if args.model_help:
        print_model_help()
        sys.exit(0)

    if not args.root_preset and not args.root:
        parser.error("One of --root OR --root_preset is required.")

    if not args.Path and not args.Files:
        parser.error("One of --Path OR --Files must be supplied.")

    # Add validation for --sets argument
    if args.sets:
        args.sets = validate_sets_argument(args.sets)

    args.model_val_int = int(MODEL_MAPPING.get(args.model, MODEL_MAPPING["x4v3"]))  # Default to "x4v3" if unset\
    if args.root_preset:
        root = ""
        match args.root_preset:
            case "p1":
                root = ROOT_PRESETS[0]
            case "p2":
                root = ROOT_PRESETS[1]
            case "p3":
                root = ROOT_PRESETS[2]
            case "p4":
                root = ROOT_PRESETS[3]
            case "p5":
                root = ROOT_PRESETS[4]
            case "p6":
               root = ROOT_PRESETS[5]

        args.root_preset = root
        args.root_overide = True
    else:
        args.root_overide = False
    args.rel_path = None
    return args

def parse_set_range(range_str: str) -> Optional[Tuple[int, Optional[int]]]:
    """
    Parse a set range string into start and end numbers.

    Examples:
        "001-025" -> (1, 25)
        "100-200" -> (100, 200)
        "225-"    -> (225, None)
        "042"     -> (42, 42)

    Returns None if the string is not a valid numeric range.
    """
    # Handle single numbers (e.g. "001" or "42")
    if re.match(r'^\d+$', range_str):
        num = int(range_str)
        return (num, num)

    # Handle ranges (e.g. "001-025" or "225-")
    match = re.match(r'^(\d+)-(\d+)?$', range_str)
    if match:
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else None
        return (start, end)

    return None

def validate_sets_argument(sets_arg: List[str]) -> List[str]:
    """
    Validate and process the --sets argument values.
    Returns processed list of set specifications.

    Example:
        Input: ["001-025", "100-200", "225-", "Blue Dress"]
        Output: ["001", "002", ..., "025", "100", ..., "200", "225", ..., "999"]
    """
    if not sets_arg:
        return []

    if "*" in sets_arg:
        return ["*"]

    result = set()
    for range_spec in sets_arg:
        parsed = parse_set_range(range_spec)
        if parsed:
            start, end = parsed
            if end is None:
                # For ranges like "225-", use 999 as the maximum
                end = 999
            # Generate all numbers in the range, formatted as 3-digit strings
            result.update(f"{i:03d}" for i in range(start, end + 1))

    return sorted(list(result))

def validate_user_dirs(root_dir):
    """
    Checks if the given path is a valid directory.

    The function ensures the provided path is not None and is a valid directory.
    It raises an error if the path does not meet the specified criteria.

    :param root_dir: Path to the root directory that needs validation.
    :type root_dir: str
    :return: The validated root directory path.
    :rtype: str
    :raises argparse.ArgumentTypeError: If the path is None or not a valid directory.
    """
    # Checks if a given path is a valid directory.
    if root_dir is None:
        raise argparse.ArgumentTypeError(f"Error: {bc.Light_Yellow_f}At least one valid {bc.Green_f}root directory{bc.Light_Yellow_f} must be supplied.{bc.RESET}")

    if not os.path.isdir(os.path.expanduser(root_dir)):
        raise argparse.ArgumentTypeError(f"Error: {bc.Red_f}'{root_dir}'{bc.Light_Yellow_f} is not a valid directory.{bc.RESET}")
    return root_dir

# Validate user supplied media files
def validate_user_files(file_path, root_dir):
    """
    Validates the user-provided file path to ensure it corresponds to a valid file. If the
    file path is invalid or does not exist in the given root directory, appropriate
    exceptions are raised. This validation ensures that only existing and accessible
    files are processed.

    :param file_path: The relative file path provided by the user.
    :type file_path: str
    :param root_dir: The root directory against which the file path is validated.
    :type root_dir: str
    :return: The valid file path if all checks pass.
    :rtype: str
    :raises argparse.ArgumentTypeError: If the file path is None, or if the file does not
        exist in the specified root directory.
    """
    # Checks if a given path is a valid file.
    if file_path is None:
        raise argparse.ArgumentTypeError(f"Error: {bc.Light_Yellow_f}At least one valid file must be supplied.{bc.RESET}")
    full_path = os.path.join(root_dir, file_path)
    if not os.path.isfile(os.path.expanduser(full_path)):
        raise argparse.ArgumentTypeError(
            f"Error: {bc.Red_f}'{full_path}' {bc.Light_Yellow_f} is not valid.\n"
            f"ROOT_DIR: {root_dir}\n"
            f"--Files parameter: {file_path}{bc.RESET}"
        )
    return file_path

def print_model_help():
    """
    Displays information about various models including their characteristics,
    recommendations for use, and performance notes. The function prints a detailed
    help message describing different models and their suited applications. After
    displaying the information, the program exits.

    :return: This function does not return any value.
    :rtype: None
    """
    print(
        f"""
        {bc.BOLD}{bc.Light_Blue_f}Model Help:\n
        {bc.Green_f}x4v3:
        {bc.White_f} • Small, fast, good all-rounder at fine detail and natural textures.
        {bc.White_f} • Default choice for {bc.Light_Blue_f}{bc.UNDERLINE}speed{bc.RESET_UNDERLINED}{bc.White_f} and{bc.Light_Blue_f}{bc.UNDERLINE} quality{bc.RESET_UNDERLINED}.
        
        {bc.Green_f}x4plus:
        {bc.White_f} • Larger model, better at fine detail and natural textures.
        {bc.White_f} • {bc.UNDERLINE}More{bc.RESET_UNDERLINED}{bc.Light_Blue_f} detail recovery{bc.White_f} but slower processing.
        {bc.Light_Blue_f} • HINT:{bc.Red_f}  It's slower than dog shit!
        
        {bc.Green_f}net_x4plus:
        {bc.White_f} • Less aggressive sharpening, more faithful to original.
        {bc.White_f} • Good model if current output looks {bc.Light_Blue_f}{bc.UNDERLINE}over-processed{bc.RESET_UNDERLINED}{bc.White_f} or {bc.Light_Blue_f}{bc.UNDERLINE}plastic{bc.RESET_UNDERLINED}.
        
        {bc.Green_f}x2plus:
        {bc.White_f} • 2x upscale model.
        {bc.White_f} • If you want {bc.Light_Blue_f}{bc.UNDERLINE}gentler{bc.RESET_UNDERLINED}{bc.White_f} enhancemant and {bc.Light_Blue_f}{bc.UNDERLINE}less{bc.RESET_UNDERLINED}{bc.White_f} risk of artifacts.
        
        {bc.Green_f}x4plus_anime_6B:
        {bc.White_f} • Optimized for {bc.Light_Blue_f}{bc.UNDERLINE}line art{bc.RESET_UNDERLINED}{bc.White_f} and {bc.Light_Blue_f}{bc.UNDERLINE}flat colors{bc.RESET_UNDERLINED}.
        {bc.White_f} • Use if {bc.Light_Blue_f} {bc.UNDERLINE}problem{bc.RESET_UNDERLINED}{bc.White_f} images have strong edges or stylized elements.{bc.RESET_BOLD}{bc.RESET}
        """
    )

    sys.exit(1)
