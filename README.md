# esrgan-APP

## Overview
esrgan-APP is a GUI and CLI wrapper for [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) image restoration and upscaling. The GUI component is written in PyQT5.  The program requires a specific directory structure; if it is not followed, the program will not run.

## Disclaimer

This project and its workflow reflect the author’s personal setup and usage.  
It is not intended to be a universal solution, nor does it make any assertions about accommodating other workflows.  

Users are welcome to adapt the scripts and structure to their own needs, but the documentation here is written from the perspective of the author’s environment and practices.

NOTE: This project *requires* an already working installation of Real-ESRGAN and any required dependencies.
---

## Required directory structure

- **Root directory**: Base path containing all studios and models (e.g., */home/nikki/+Graphics/+Models*).
- **Studio**: Subdirectory under the root representing a collection of models (e.g., *Nikki Studios*, *Baby Studios*).
- **Model**: Subdirectory under a studio representing a single model (e.g., *Nikki*, *Chrissy*, *Baby1*).
- **Set**: Subdirectory under a model numbered 001–999. Sets may skip numbers.
- **Images**: Files inside each set following the filename pattern below.

### Filename pattern

    Model-SetNumber-ImageNumber.ext

- Model → model name (e.g., Nikki)
- SetNumber → three-digit set ID (e.g., 001)
- ImageNumber → three-digit sequence (001–999)
- ext → png or jpg
- Optional suffix → appended before extension as _Suffix (e.g., _X4V3)

### Example layout

    /home/nikki/+Graphics/+Models/
    └── Nikki Studios/
        └── Nikki/
            ├── 001/
            │   ├── Nikki-001-001.jpg
            │   ├── Nikki-001-002.jpg
            ├── 002/
            │   └── Nikki-002-001.jpg
            ├── 004/
            │   └── Nikki-004-001.jpg
            ├── 005/
            │   └── Nikki-005-001.jpg
            └── 006/
                ├── Nikki-006-001.jpg
                ├── Nikki-006-002.jpg
                └── Nikki-006-003.jpg

---

## Output directory structure

The output root mirrors the input hierarchy:

    OUTPUT_ROOT/Studio/Model/Set/Model-SetNumber-ImageNumber_{suffix}.ext

- Suffix: Optional via --suffix; automatically prepended with _ when present (e.g., Nikki-001-001_X4V3.png).
- Extension: Defaults to png; override with --ext jpg.

### Input to output examples

Input:

    INPUT_ROOT/Nikki Studios/Nikki/001/Nikki-001-001.jpg

Output (default):

    OUTPUT_ROOT/Nikki Studios/Nikki/001/Nikki-001-001.png

Output (with suffix set to X4V3 and extension set to jpg):

    OUTPUT_ROOT/Nikki Studios/Nikki/001/Nikki-001-001_X4V3.jpg

---

## CLI usage

Invoke the CLI via the esrgan_app.sh script. It is recommended to create a symlink to esrgan_app.sh in a directory on your $PATH.

### Root selection (mutually exclusive)

- --root ROOT → full path to the root directory
- --root_preset {p1,p2,p3,p4,p5,p6} → shorthand preset defined in user configuration

### Examples

Using --root:

    esrgan_app.sh \
      --root /home/nikki/+Graphics/+Models \
      --Path "Nikki Studios/Nikki" \
      --sets 001-006 \
      --ext jpg

Using --root_preset (p1 == /home/nikki/+Graphics/+Models):

    esrgan_app.sh \
      --root_preset p1 \
      --Path "Nikki Studios/Nikki" \
      --sets 001-006 \
      --ext jpg

---

## CLI Options

- **Root and Paths: (mutually exclusive)** 
  - --root ROOT
  - --root_preset {p1,p2,p3,p4,p5,p6}
  - --Path Studio/Model

- **Set Selection:**
  - --sets [SETS ...]

    **Format Options:**

    - '*' → all sets
    - 'NNN' → single set (e.g., 001)
    - 'NNN-MMM' → range (e.g., 001-025)
    - 'NNN-' → from NNN to end
    - Multiple explicit sets → e.g., 001 002 004 006

- **Output Formatting:**
  - --suffix SUFFIX → optional; auto prepends _
  - --ext {png|jpg} → default: png

- **ESRGAN Model Selection:**
  - --model {x4v3,x4plus,net_x4plus,x2plus,x4plus_anime_6B} → default: x4v3
  - --model_help

- **Enhancement:**
  - --face_enhance → apply GFPGAN after upscaling

- **Tiling and Scale:**
  - --tile TILE → default: 800
  - --tile_pad TILE_PAD → default: 10
  - --outscale OUTSCALE → default: 1.0

- **GPU:**
  - --gpu_id GPU_ID → default: 0

---

## Dependencies

Minimal requirements for this CLI wrapper:

    PyQt5>=5.15

Argparse is part of Python’s standard library. ESRGAN and optional face enhancement (GFPGAN) require additional dependencies; these will be listed in a separate section.

## Installation

### 1. Clone the repository
Clone the ESRGAN-APP project into your desired directory:

    git clone https://github.com/YourUser/esrgan-APP.git
    cd esrgan-APP

### 2. Create a virtual environment
It is recommended to use a Python virtual environment:

    python3 -m venv venv
    source venv/bin/activate   # Linux/macOS
    venv\Scripts\activate      # Windows

### 3. Install dependencies
Install the minimal requirements for the CLI wrapper:

    pip install -r requirements.txt

Note: ESRGAN itself requires additional dependencies (GPU libraries, GFPGAN, etc.) which is outside the scope of this guide. See the [ESRGAN project page](https://github.com/xinntao/Real-ESRGAN) for installation instructions. 

### 4. Make the CLI script accessible
The CLI is invoked via `esrgan_app.sh`. To make it easy to run from anywhere, create a symlink to a directory on your PATH (e.g., `/usr/local/bin`):

    ln -s /path/to/esrgan-APP/esrgan_app.sh /usr/local/bin/esrgan_app.sh

Now you can run the program simply by typing:

    esrgan_app.sh --cli --help

### 5. Verify installation
Run the help command to confirm everything is working:

    esrgan_app.sh --cli --help

## GUI Installation

The GUI requires a small additional setup step to install its resources.

- Run the provided `install.sh` script:

      ./install.sh

- This script copies the GUI assets into your local share directory:

      ~/.local/share/esrgan-APP

- The assets include:
  - Icons (from the `Resources` directory in the repo)
  - Help documentation (`help.txt` from the `Help` directory)

⚠️ Note: This installation is **per-user**. Advanced users who prefer a system-wide setup can review the simple `install.sh` script and adapt it as needed.

### Run the GUI
Run the GUI via the `esrgan_app.sh` script:

    esrgan_app.sh

The GUI should appear after a few seconds. Please note that the GUI was developed and tested on Linux only.  Additionally, the GUI was developed using 4K-UHD displays so YMMV on lower resolution displays. No attempt has been made to optimize the GUI for smaller displays.

The rest of the documentation is currently work in progress.
