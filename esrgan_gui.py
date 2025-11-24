#  esrgan_gui.py Copyright (c) 2025 Nikki Cooper
#
#  This program and the accompanying materials are made available under the
#  terms of the GNU Lesser General Public License, version 3.0 which is available at
#  https://www.gnu.org/licenses/gpl-3.0.html#license-text
#
import sys
import subprocess
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QComboBox,
                             QFileDialog, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit,
                             QProgressBar, QMessageBox, QListWidget,
                             QDialog, QDialogButtonBox,QTextEdit)
from PyQt5.QtCore import (Qt, QThread, QSize, pyqtSignal, QCoreApplication, QCommandLineParser, QCommandLineOption, QMutex, QWaitCondition)
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QIcon
from GUI_User_Setup import REAL_ESRGAN_SCRIPT, DEFAULT_OUTPUT_DIR, DEFAULT_ROOT_DIR
try:
    from GUI_User_Setup_local import REAL_ESRGAN_SCRIPT, DEFAULT_OUTPUT_DIR, DEFAULT_ROOT_DIR
except ImportError:
    pass

class ImageProcessor(QThread):
    # Define the signals at the class level
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    error_recovery_signal = pyqtSignal(str)

    def __init__(self, input_paths, output_dir, model_name, outscale, tile,
                 tile_pad, gpu_id, face_enhance, suffix="AI", ext="png"):
        super().__init__()
        self.input_paths = input_paths
        self.output_root = Path(output_dir)
        self.model_name = model_name
        self.outscale = str(outscale)
        self.tile = str(tile)
        self.tile_pad = str(tile_pad)
        self.gpu_id = str(gpu_id)
        self.face_enhance = face_enhance
        self.suffix = suffix
        self.ext = ext
        self.is_cancelled = False
        self.current_process = None
        self.current_img_num = 0            # The number of images processed so far in the current set.
        self.total_num_images = 0           # The total number of images to be processed in the current set.
        self.completed_sets_counter = 0     # How many sets have been processed successfully.
        self.total_num_sets_to_process = 0  # Total number of sets to process.
        self.current_set = None             # Contains the name of the current set being processed.

        self.is_paused = False
        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()

    def pause(self):
        self.mutex.lock()
        self.is_paused = True
        self.mutex.unlock()

    def resume(self):
        self.mutex.lock()
        self.is_paused = False
        self.wait_condition.wakeAll()
        self.mutex.unlock()

    def check_paused(self):
        self.mutex.lock()
        if self.is_paused:
            self.progress.emit('<font color="#ff55ff">Processing Paused...</font>')
            self.wait_condition.wait(self.mutex)
            if not self.is_cancelled:
                self.progress.emit('<font color="#00ff00">Processing Resumed...</font>')
        self.mutex.unlock()

    def cancel(self):
        self.is_cancelled = True
        self.resume()  # Resume if paused so thread can exit
        if self.current_process:
            self.current_process.terminate()

    def process_single_file(self, img_path, root_dir):
        self.check_paused()

        if self.is_cancelled:
            return False

        # Create output directory maintaining the structure Studio/Model/Set
        rel_path = img_path.relative_to(root_dir)
        if len(rel_path.parts) >= 4:  # Studio/Model/Set/image.jpg
            studio, model, set_name = rel_path.parts[-4:-1]
            output_dir = self.output_root / studio / model / set_name
        else:  # Model/Set/image.jpg
            model, set_name = rel_path.parts[-3:-1]
            output_dir = self.output_root / model / set_name

        self.current_set = set_name
        # Create directory only when we're about to process files in it
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", REAL_ESRGAN_SCRIPT,
            "-n", self.model_name,
            "-i", str(img_path),
            "-o", str(output_dir),
            "--outscale", self.outscale,
            "--gpu-id", self.gpu_id,
            "--ext", self.ext,
            "--suffix", self.suffix,
            "--tile", self.tile,
            "--tile_pad", self.tile_pad
        ]

        if self.face_enhance:
            cmd.append("--face_enhance")

        # Create a progress message for the current image
        proc_img_msg = (
            f'<p style="text-align:center;color: #0080ff; font-size: 16px; font-weight: bold; padding-inline: 50px;">'
            f'<font color="#ff55ff">Processing image </font>'
            f'(<b><font color="#00ff00">{self.current_img_num}</font></b> '
            f'of <b><font color="#ffff00">{self.total_num_images}</font></b>)'
            f'<font color="#ff55ff"> in SET </font><b><font color="#00ff00">{self.current_set}</font></b> | '
            f'<b><font color="#0080ff"> {img_path.name} </font></b> '
        )

        # Create a progress message for completed sets
        proc_set_msg = (
            f'<font color="#ff55ff">Completed SETS: </font>'
            f'(<b><font color="#00ff00">{self.completed_sets_counter}</font></b> '
            f'of <b><font color="#ffff00">{self.total_num_sets_to_process}</font></b>)'
            f'</p>'
        )

        full_msg = proc_img_msg + "  |  " + proc_set_msg

        self.progress.emit(full_msg)

        try:
            # Use Popen to allow cancellation and pipe capture
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait for the process to finish and capture output
            stdout, stderr = self.current_process.communicate()
            return_code = self.current_process.returncode

            # If cancellation was requested (and handled by terminate in cancel()), stop here.
            if self.is_cancelled:
                self.current_process = None
                return False

            # Determine failure:
            # 1. Non-zero return code
            # 2. Error keywords in output (even if return code is 0)
            failed = (return_code != 0)

            combined_output = (stderr or "") + (stdout or "")

            # Check for specific error markers in the output
            # This catches cases where the script prints an error but exits with 0
            error_markers = ["CUDA out of memory", "Traceback (most recent call last)", "RuntimeError:", "Error:"]
            if not failed:
                for marker in error_markers:
                    if marker in combined_output:
                        failed = True
                        break

            if failed:
                # Pause the thread to wait for user input
                self.mutex.lock()
                self.is_paused = True
                self.mutex.unlock()

                error_details = (
                    f"--- STDERR ---\n{stderr}\n\n"
                    f"--- STDOUT ---\n{stdout}"
                )

                if not combined_output.strip():
                    error_details = f"Process failed with no output (Return Code: {return_code})"

                # Emit signal to show dialog
                self.error_recovery_signal.emit(f"Error processing {img_path.name}:\n\n{error_details}")

                # Wait until the user clicks Continue (resumes) or Cancel (cancels+resumes)
                self.check_paused()

                self.current_process = None
                return False

            self.current_process = None
            self.progress.emit(f"Completed: {img_path.name}")
            return True

        except Exception as e:
            self.mutex.lock()
            self.is_paused = True
            self.mutex.unlock()

            self.error_recovery_signal.emit(f"Error processing {img_path.name}: {str(e)}")
            self.check_paused()

            self.current_process = None
            return False

    def run(self):
        """
        Executes the process of iterating through input paths, identifying image files,
        and processing them individually. Handles directory structures to locate
        images and emits signals for progress updates, errors, or completion.

        The method identifies image files in the specified `input_paths` based on
        their extensions and processes them while maintaining progress tracking
        and responsiveness to cancellation requests.

        :raises Exception: If an unexpected error occurs during processing.

        """
        try:
            self.output_root.mkdir(parents=True, exist_ok=True)
            image_exts = {".jpg", ".jpeg", ".png"}

            # Process all selected Sets.
            self.total_num_sets_to_process = len(self.input_paths)

            for input_path in self.input_paths:
                if self.is_cancelled:
                    break

                path = Path(input_path)
                root_dir = path
                while len(root_dir.parts) > 2:
                    root_dir = root_dir.parent

                if path.is_dir():
                    self.total_num_images = 0
                    self.current_img_num = 0
                    images = sorted(path.glob("*.*"))
                    self.total_num_images = len(images)
                    print(f"Number of images in {path}: {self.total_num_images}")
                    # Process all images in Set.
                    for img_path in images:
                        if self.is_cancelled:
                            break
                        if img_path.is_file() and img_path.suffix.lower() in image_exts:
                            self.current_img_num += 1
                            self.process_single_file(img_path, root_dir)

                    # Increment completed sets counter after processing all images in the set.
                    if not self.is_cancelled:
                        self.completed_sets_counter += 1

        except Exception as e:
            self.error.emit(str(e))
        finally:
            if self.is_cancelled:
                self.progress.emit("Processing cancelled")
            self.finished.emit()

class ESRGANGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = None
        self.root_dir = None
        self.current_studio = None
        self.current_model = None
        self.is_cancelling = False
        self.is_paused = False
        self.controls = []  # List to hold all controls that should be disabled during processing
        self.image_count = 0
        self.checkmark = Path.home() / '.local/share/esrgan-APP/Resources/checkmark_white-8x8.png'
        self.initUI()

    def initUI(self):
        """
        Initializes the User Interface (UI) for the ESRGAN Image Processor application.

        This method sets up the main layout and components of the graphical user interface,
        including directory selection, studio and model selection, list widgets for batch
        selections, parameter inputs, and output configurations.

        :raises AttributeError: If any UI component could not initialize properly or was
        inaccessible during the setup of the interface.

        :return: None
        :rtype: None
        """
        self.setWindowTitle('ESRGAN Image Processor by Nikki Cooper')
        self.setGeometry(100, 100, 800, 700)


        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.setFixedWidth(800)
        self.setFixedHeight(715)
        layout = QVBoxLayout(central_widget)

        # Root directory selection
        folder_icon = Path.home() / '.local/share/esrgan-APP/Resources/folder.png'
        root_dir_layout = QHBoxLayout()
        self.root_dir_path = QLineEdit()
        self.root_dir_path.setToolTip("""
            <div style='white-space: nowrap;'>
            <h3 style='color: #ff55ff; text-align:center;'><b>Root Directory:</b></h3>
            Enter the root directory where your <b>Studio/Model/Sets</b> are located.
            </div>
         """)
        self.root_dir_path.setPlaceholderText('Select root directory')
        self.root_dir_path.setText(str(DEFAULT_ROOT_DIR))
        browse_root_dir_btn = QPushButton('Browse')
        browse_root_dir_btn.setToolTip("Browse to Select <b>Root Directory</b>")
        browse_root_dir_btn.setIcon(QIcon(str(folder_icon)))
        browse_root_dir_btn.setIconSize(QSize(16, 16))
        browse_root_dir_btn.setFixedWidth(110)
        browse_root_dir_btn.clicked.connect(self.browse_root_dir)
        root_dir_layout.addWidget(QLabel('Root Directory:'))
        root_dir_layout.addWidget(self.root_dir_path)
        root_dir_layout.addWidget(browse_root_dir_btn)
        layout.addLayout(root_dir_layout)

        # Studio and Model selection
        selection_layout = QHBoxLayout()

        # Studio ComboBox
        self.studio_combo = QComboBox()
        self.studio_combo.setToolTip("Select Studio or Agency")
        self.studio_combo.setFixedWidth(250)
        self.studio_combo.currentTextChanged.connect(self.on_studio_changed)
        studio_label = QLabel('Studio:')
        studio_label.setFixedWidth(75)
        selection_layout.addWidget(studio_label)
        selection_layout.addWidget(self.studio_combo)
        selection_layout.setSpacing(75)
        selection_layout.addStretch()

        # Model ComboBox
        self.model_combo_name = QComboBox()
        self.model_combo_name.setToolTip("Select Model")
        self.model_combo_name.setFixedWidth(310)
        self.model_combo_name.currentTextChanged.connect(self.on_model_changed)
        m_label = QLabel('Model:')
        m_label.setFixedWidth(75)
        selection_layout.addWidget(m_label)
        selection_layout.addWidget(self.model_combo_name)
        selection_layout.setSpacing(25)
        selection_layout.addStretch()
        layout.addLayout(selection_layout)

        # Sets list for multiple selection
        list_layout = QHBoxLayout()  # Create horizontal layout for the list
        self.sets_list = QListWidget()
        self.sets_list.setToolTip("""
                                  <div style='white-space: nowrap;'>
                                  <br>Select <b>Set</b> or <b>Sets</b> to process<br>
                                  </div>
                                  """)
        self.sets_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.sets_list.setFixedWidth(405)
        self.sets_list.setFixedHeight(206)

        # Create label layout for better alignment
        label_layout = QHBoxLayout()
        self.list_label = QLabel('Available Sets (Ctrl+click or Shift+click to select multiple):')
        self.list_label.setToolTip('Select set or Sets to process')
        label_layout.addStretch()
        label_layout.addWidget(self.list_label)
        label_layout.addStretch()
        layout.addLayout(label_layout)

        # Add stretches before and after the list widget to center it
        list_layout.addStretch()
        list_layout.addWidget(self.sets_list)
        list_layout.addStretch()

        layout.addLayout(list_layout)  # Add the list layout to the main layout


        # ESRGAN Model selection with Help button
        model_layout = QHBoxLayout()
        model_label = QLabel('ESRGAN Model:')
        model_label.setFixedWidth(115)
        model_layout.addWidget(model_label)
        model_layout.addStretch()  # This pushes the combo box and help button to the right

        self.model_combo = QComboBox()
        self.model_combo.setToolTip("""
        <div style='white-space: nowrap;'>
            <h3 style='color: #ff55ff; text-align:center;'><b>Available Models:</b></h3>
            • realesr-general-x4v3: Fast, general purpose<br>
            • RealESRGAN_x4plus: High quality, slower<br>
            • RealESRNet_x4plus: Conservative enhancement<br>
            • RealESRGAN_x2plus: 2x upscaling<br>
            • RealESRGAN_x4plus_anime_6B: Optimized for anime
        </div>
        """)
        self.model_combo.addItems([
            'realesr-general-x4v3',
            'RealESRGAN_x4plus',
            'RealESRNet_x4plus',
            'RealESRGAN_x2plus',
            'RealESRGAN_x4plus_anime_6B'
        ])
        self.model_combo.setFixedWidth(275)  # Adjust this value as needed

        help_icon = Path.home() / '.local/share/esrgan-APP/Resources/help.png'
        # Add Help button
        self.model_help_btn = QPushButton('Help')
        self.model_help_btn.setToolTip('More Real-ESRGAN Models and Information')
        self.model_help_btn.setIcon(QIcon(str(help_icon)))
        self.model_help_btn.setIconSize(QSize(16, 16))
        self.model_help_btn.setFixedWidth(80)
        self.model_help_btn.clicked.connect(self.show_model_help)

        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.model_help_btn)
        model_layout.setSpacing(115)
        #model_layout.addStretch()
        layout.addLayout(model_layout)

        # Output directory
        output_layout = QHBoxLayout()
        self.output_path = QLineEdit()
        self.output_path.setToolTip("""
        <div style='white-space: nowrap;'>
            <h3 style='color: #ff55ff; text-align:center;'><b>Output_Root:</b></h3>
            Enter the Output_Root directory.<br>
            The final output path will be:<br>
            <b>Output_Root</b><i>/Studio/Model/Set/</i><br>
        </div>
        """)

        self.output_path.setText(str(DEFAULT_OUTPUT_DIR))
        self.output_path.setFixedWidth(475)
        browse_output_btn = QPushButton('Browse')
        browse_output_btn.setToolTip("Browse to select Output_Root directory")
        browse_output_btn.setIcon(QIcon(str(folder_icon)))
        browse_output_btn.setIconSize(QSize(16, 16))
        browse_output_btn.setFixedWidth(110)
        browse_output_btn.clicked.connect(self.browse_output)
        output_label = QLabel('Output Root:')
        output_label.setFixedWidth(100)
        output_layout.setSpacing(50)
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_path)
        output_layout.addStretch()
        output_layout.addWidget(browse_output_btn)
        #output_layout.setSpacing(80)
        #output_layout.addStretch()
        layout.addLayout(output_layout)

        # Parameters section
        params_layout = QVBoxLayout()

        # Suffix
        suffix_layout = QHBoxLayout()
        self.suffix_input = QLineEdit()
        self.suffix_input.setToolTip("""
            <div style='white-space: nowrap;'>
            <h3 style='color: #ff55ff; text-align:center;'><b>Optional suffix</b></h3>
            (e.g. V1, V2)<br>
            _Suffix will be appended to the output file name:<br>
            filename<b>_Suffix</b>.ext<br>        
            </div>   
            """)

        self.suffix_input.setPlaceholderText('Optional suffix (e.g. V1, V2)')
        self.suffix_input.setFixedWidth(475)
        suffix_label = QLabel('Suffix:')
        suffix_label.setFixedWidth(100)
        suffix_layout.addWidget(suffix_label)
        suffix_layout.addWidget(self.suffix_input)
        suffix_layout.setSpacing(50)
        suffix_layout.addStretch()
        params_layout.addLayout(suffix_layout)

        # Output Format
        format_layout = QHBoxLayout()
        self.format_combo = QComboBox()
        self.format_combo.setToolTip("Select output image format")
        self.format_combo.addItems(['png', 'jpg', 'jpeg'])
        self.format_combo.setFixedWidth(78)
        self.format_combo.setFixedHeight(30)
        output_format_label = QLabel('Output Format:')
        output_format_label.setFixedWidth(300)
        format_layout.addWidget(output_format_label)
        format_layout.addWidget(self.format_combo)
        format_layout.setSpacing(100)
        format_layout.addStretch()
        params_layout.addLayout(format_layout)

        # Create a consistent width for all labels
        label_width = 100  # Adjust this value as needed
        widget_spacing = 300

        # Outscale
        outscale_layout = QHBoxLayout()
        outscale_label = QLabel('Outscale:')
        outscale_label.setFixedWidth(label_width)
        self.outscale_spin = QDoubleSpinBox()
        self.outscale_spin.setToolTip("""
        <div style='white-space: nowrap;'>
            <h3 style='color: #ff55ff; text-align:center;'>Outscale</h3>
            • Defines the final upscaling factor applied to the image.<br>
            • <span style='color:#00ff7f;'><b>1</span></b> = keep original resolution, but still apply restoration/denoising.<br>
            • Very useful if you prefer your GPU or monitor hardware to handle the final upscale 
            (e.g., 1080p sources displayed on 4K screens).<br>
            • <span style='color:#00ff7f;'><b>2</span></b> = double width/height (4× total pixels).<br>
            • <span style='color:#00ff7f;'><b>4</span></b> = quadruple width/height (16× total pixels).<br>
            • Higher values = larger output and higher VRAM usage.<br>
            • Supports decimal values for precise scaling (e.g. 1.5 = 150%).<br>
            • <span style='color: #ff55ff';>Default:</span> <span style='color: #00ff7f;'><b>1</b></span><br>
        </div>
        """)

        self.outscale_spin.setRange(0.1, 4.0)       # Allow values from 0.1 to 4.0
        self.outscale_spin.setSingleStep(0.01)              # Set step size to 0.01
        self.outscale_spin.setDecimals(2)                   # Show 2 decimal places
        self.outscale_spin.setValue(1.0)                    # Set default value to 1.0
        outscale_layout.addWidget(outscale_label)
        outscale_layout.addWidget(self.outscale_spin)
        outscale_layout.setSpacing(widget_spacing)
        outscale_layout.addStretch()
        params_layout.addLayout(outscale_layout)

        # Tile
        tile_layout = QHBoxLayout()
        tile_label = QLabel('Tile:')
        tile_label.setFixedWidth(label_width)
        self.tile_spin = QSpinBox()
        self.tile_spin.setToolTip("""
        <div style='white-space: nowrap;'>
             <h3 style="color: #ff55ff; text-align:center;">Tile Size</h3>

            • Defines the width/height of each chunk (e.g.,256,512).<br>
            • If set to 0, no tiling is used; The whole image is processed at once.<br>
            • Smaller values reduce GPU memory usage but may cause seams and be slower.<br>
            • Larger values use more GPU memory but reduce seams and can be faster.<br>            
            • <span style="color: #ff55ff";>Default:</span> <span style="color: #00ff7f;">800</span><br> 
        </div>  
        """)
        self.tile_spin.setRange(0, 1000)
        self.tile_spin.setValue(800)
        tile_layout.addWidget(tile_label)
        tile_layout.addWidget(self.tile_spin)
        tile_layout.setSpacing(widget_spacing)
        tile_layout.addStretch()
        params_layout.addLayout(tile_layout)

        # Tile Pad
        tile_pad_layout = QHBoxLayout()
        tile_pad_label = QLabel('Tile Pad:')
        tile_pad_label.setFixedWidth(label_width)
        self.tile_pad_spin = QSpinBox()
        self.tile_pad_spin.setToolTip("""
        <div style='white-space: nowrap;'>
            <h3 style="color: #ff55ff; text-align:center;">Tile Pad</h3>
            • Adds overlap (padding) around each tile so edges blend smoothly when stitched.<br>
            • Too low → visible seams (especially in hair or fine textures).<br>
            • Too high → slightly more VRAM use, but safer.<br>
            • <span style="color: #ff55ff";>Default:</span> <span style="color: #00ff7f;">10 pixels</span><br>
        </div>
        """)

        self.tile_pad_spin.setRange(0, 100)
        self.tile_pad_spin.setValue(10)
        tile_pad_layout.addWidget(tile_pad_label)
        tile_pad_layout.addWidget(self.tile_pad_spin)
        tile_pad_layout.setSpacing(widget_spacing)
        tile_pad_layout.addStretch()
        params_layout.addLayout(tile_pad_layout)

        # GPU ID
        gpu_layout = QHBoxLayout()
        gpu_label = QLabel('GPU ID:')
        gpu_label.setFixedWidth(label_width)
        self.gpu_spin = QSpinBox()
        self.gpu_spin.setToolTip("GPU device ID to use for processing.\nUse 0 for primary GPU.")
        self.gpu_spin.setRange(0, 8)
        self.gpu_spin.setValue(0)
        gpu_layout.addWidget(gpu_label)
        gpu_layout.addWidget(self.gpu_spin)
        gpu_layout.setSpacing(widget_spacing)
        gpu_layout.addStretch()
        params_layout.addLayout(gpu_layout)

        # Face Enhance
        self.face_enhance_check = QCheckBox('Face Enhance')
        self.face_enhance_check.setStyleSheet(f"""
            QCheckBox::indicator:checked {{
                background-color: #3e738f;
                border: 1px solid #64c8ff;
                image: url({self.checkmark});
           }}
           QCheckBox::indicator {{
                width: 16px;
                height: 16px;
        }}

        """)
        self.face_enhance_check.setToolTip("""
        <div style='white-space: nowrap;'>
            <h3 style="color: #ff55ff; text-align:center;">Face Enhance</h3>
            <p><span style='text-align: left;'>
            • Uses GFPGAN, a separate AI model, to reconstruct and enhance detected faces.<br>
            • The enhanced face is blended back into the upscaled image.<br>
            • May introduce seams where the face meets hair or background.<br>
            </span></p>

            <h4 style="color: #ff55ff">When To Use</h4>
            • Low-res portraits where faces are <u><i>blurry</i></u> or <u><i>smudged</i></u>.<br>
            • Old photos where facial features need reconstruction.<br>
            • Group shots where you want faces to "<u><i>pop</i></u>" more clearly.<br>

            <h4 style="color: #ff55ff">When Not To Use</h4>
            • Already high-quality photos (can over-smooth or “plasticize” faces).<br>
            • Artistic or stylized portraits (may “normalize” them into generic faces).<br>
            • Large images with many faces (slower, inconsistent results).<br>
            • Situations where hair, hats, or background overlap the face region.<br>
        </div>
        """)

        params_layout.addWidget(self.face_enhance_check)

        layout.addLayout(params_layout)

        # Progress and control section
        progress_layout = QHBoxLayout()

        # Progress display
        self.progress_label = QLabel()
        layout.addWidget(self.progress_label)

        button_layout = QHBoxLayout()

        checkmark_icon = Path.home() / '.local/share/esrgan-APP/Resources/checkmark.png'
        # Single button for Process/Cancel
        self.process_btn = QPushButton('Process Selected Sets')
        self.process_btn.setToolTip("Click to process selected sets")
        self.process_btn.setIcon(QIcon(str(checkmark_icon)))
        self.process_btn.setFixedWidth(230)
        self.process_btn.clicked.connect(self.process_button_clicked)

        # Pause Button
        self.pause_btn = QPushButton('Pause')
        self.pause_btn.setToolTip("Pause/Resume processing")
        self.pause_btn.setFixedWidth(100)
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.toggle_pause)

        # Add stretches before and after the button to center it
        button_layout.addStretch()
        button_layout.addWidget(self.process_btn)
        button_layout.addWidget(self.pause_btn)
        button_layout.addStretch()

        # Add the button layout to the main layout
        layout.addLayout(button_layout)
        layout.addStretch()

        # Initialize root directory
        self.root_dir = DEFAULT_ROOT_DIR
        self.refresh_studios()
        # Add all controls to the list for easy enabling/disabling
        self.controls = [
            self.root_dir_path,     # Add root directory text input
            self.studio_combo,      # Add studio dropdown
            self.model_combo_name,  # Add model dropdown
            self.model_combo,
            self.sets_list,
            self.output_path,
            self.outscale_spin,
            self.tile_spin,
            self.tile_pad_spin,
            self.gpu_spin,
            self.face_enhance_check,
            self.format_combo,
            self.suffix_input,
            self.process_btn,
            # Add browse buttons
            browse_output_btn,
            browse_root_dir_btn     # Add root directory browse button
        ]

    def process_button_clicked(self):
        """Handle the process/cancel button clicks"""
        if self.process_btn.text() == 'Process Selected Sets':
            self.process_images()
        else:
            self.cancel_processing()

    def toggle_pause(self):
            if not self.processor:
                return

            if self.pause_btn.text() == 'Pause':
                self.processor.pause()
                self.pause_btn.setText('Resume')
                self.is_paused = True
            else:
                self.processor.resume()
                self.pause_btn.setText('Pause')
                self.is_paused = False

    def disable_controls(self):
        """Disable all controls during processing"""
        self.list_label.setEnabled(False)
        # Create a mapping of controls to their labels
        control_label_pairs = [
            (self.root_dir_path, 'Root Directory:'),
            (self.studio_combo, 'Studio:'),
            (self.model_combo_name, 'Model:'),
            (self.model_combo, 'ESRGAN Model:'),
            (self.output_path, 'Output Root:'),
            (self.suffix_input, 'Suffix:'),
            (self.format_combo, 'Output Format:'),
            (self.outscale_spin, 'Outscale:'),
            (self.tile_spin, 'Tile:'),
            (self.tile_pad_spin, 'Tile Pad:'),
            (self.gpu_spin, 'GPU ID:')
        ]

        # Disable each control and find its corresponding label
        for control, label_text in control_label_pairs:
            control.setEnabled(False)
            # Find the label by its text
            for label in self.findChildren(QLabel):
                if label.text() == label_text:
                    label.setEnabled(False)
                    break

        # Disable other controls
        self.sets_list.setEnabled(False)
        self.face_enhance_check.setEnabled(False)
        self.process_btn.setEnabled(True)  # Keep this enabled for cancellation
        self.process_btn.setText('Cancel Processing')
        self.process_btn.setToolTip('Click to cancel processing')

        # Buttons
        self.process_btn.setEnabled(False)
        for button in self.findChildren(QPushButton):
            button.setEnabled(False)

        self.progress_label.setText("Processing... Please wait.")

    def enable_controls(self):
        """Re-enable all controls after processing"""

        self.list_label.setEnabled(True)
        self.root_dir_path.setEnabled(True)
        # Re-enable each control and its label
        control_label_pairs = [
            (self.root_dir_path, 'Root Directory:'),
            (self.studio_combo, 'Studio:'),
            (self.model_combo_name, 'Model:'),
            (self.model_combo, 'ESRGAN Model:'),
            (self.output_path, 'Output Root:'),
            (self.suffix_input, 'Suffix:'),
            (self.format_combo, 'Output Format:'),
            (self.outscale_spin, 'Outscale:'),
            (self.tile_spin, 'Tile:'),
            (self.tile_pad_spin, 'Tile Pad:'),
            (self.gpu_spin, 'GPU ID:')
        ]

        for control, label_text in control_label_pairs:
            control.setEnabled(True)
            # Find the label by its text
            for label in self.findChildren(QLabel):
                if label.text() == label_text:
                    label.setEnabled(True)
                    break

        # Re-enable other controls
        self.sets_list.setEnabled(True)
        self.face_enhance_check.setEnabled(True)
        self.process_btn.setEnabled(True)
        self.process_btn.setText('Process Selected Sets')

        # Buttons
        self.process_btn.setEnabled(True)
        for button in self.findChildren(QPushButton):
            button.setEnabled(True)

        self.pause_btn.setText('Pause')
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText('Pause')
        self.is_paused = False

    def browse_root_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, 'Select Root Directory', str(DEFAULT_ROOT_DIR))
        if dir_path:
            self.root_dir = Path(dir_path)
            self.root_dir_path.setText(str(dir_path))
            self.refresh_studios()

    def refresh_studios(self):
        """Populate the studios combo box"""
        self.studio_combo.clear()
        if self.root_dir and self.root_dir.exists():
            studios = [d.name for d in self.root_dir.iterdir()
                       if d.is_dir() and not d.name.startswith('.')]
            self.studio_combo.addItems(sorted(studios))

    def on_studio_changed(self, studio_name):
        """Handle studio selection change"""
        self.current_studio = studio_name
        self.refresh_models()
        # Ensure the first model is properly selected and its sets are loaded
        if self.model_combo_name.count() > 0:
            first_model = self.model_combo_name.itemText(0)
            self.current_model = first_model
            self.refresh_sets_list()

    def on_model_changed(self, model_name):
        """Handle model selection change"""
        if not model_name:  # Skip if empty model name
            return
        self.current_model = model_name
        # Force refresh of sets list
        self.refresh_sets_list()

    def refresh_models(self):
        """Populate the models combo box based on selected studio"""
        self.model_combo_name.clear()
        if self.root_dir and self.current_studio:
            studio_path = self.root_dir / self.current_studio
            if studio_path.exists():
                models = [d.name for d in studio_path.iterdir()
                          if d.is_dir() and not d.name.startswith('.')]
                self.model_combo_name.addItems(sorted(models))

    def refresh_sets_list(self):
        """Populate the sets list based on selected model"""
        self.sets_list.clear()
        if not all([self.root_dir, self.current_studio, self.current_model]):
            print(f"Missing required paths: root_dir={bool(self.root_dir)}, studio={bool(self.current_studio)}, model={bool(self.current_model)}")
            return

        model_path = self.root_dir / self.current_studio / self.current_model
        if not model_path.exists():
            print(f"Model path does not exist: {model_path}")
            return

        image_exts = {".jpg", ".jpeg", ".png"}
        sets = []
        for set_dir in sorted(model_path.iterdir()):
            if set_dir.is_dir():
                # Check for image files in the directory
                has_images = False
                self.image_count = 0
                for f in set_dir.iterdir():
                    if f.is_file() and f.suffix.lower() in image_exts:
                        has_images = True
                        #break
                        self.image_count += 1

                if has_images:
                    sets.append(set_dir.name)
                    #print(f"Found set: {set_dir.name} in {model_path} with {self.image_count} images.")
                #else:
                #    print(f"No valid images found in set: {set_dir.name}")

        # Print the final list of sets found
        #print(f"Total sets found: {len(sets)}")
        if sets:
            self.sets_list.addItems(sets)
        else:
            print("No sets found with valid images")

    def browse_output(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, 'Select Output Directory', str(DEFAULT_OUTPUT_DIR))
        if dir_path:
            self.output_path.setText(dir_path)

    def cancel_processing(self):
        if self.processor:
            self.processor.cancel()
            self.process_btn.setEnabled(False)  # Disable during cancellation
            self.progress_label.setText("Cancelling...")
            self.is_cancelling = True

    def process_images(self):
        selected_items = self.sets_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, 'Error', 'Please select at least one set to process.')
            return

        if not all([self.root_dir, self.current_studio, self.current_model]):
            QMessageBox.warning(self, 'Error', 'Please select Studio and Model first.')
            return

        # Get full paths for selected sets
        selected_paths = []
        for item in selected_items:
            set_name = item.text()
            input_set_path = self.root_dir / self.current_studio / self.current_model / set_name
            if input_set_path.exists():
                selected_paths.append(input_set_path)

        # Disable controls except process button and change its text
        self.disable_controls()
        self.process_btn.setEnabled(True)
        self.process_btn.setText('Cancel Processing')

        self.pause_btn.setEnabled(True)

        self.processor = ImageProcessor(
            input_paths=selected_paths,
            output_dir=self.output_path.text(),
            model_name=self.model_combo.currentText(),
            outscale=self.outscale_spin.value(),
            tile=self.tile_spin.value(),
            tile_pad=self.tile_pad_spin.value(),
            gpu_id=self.gpu_spin.value(),
            face_enhance=self.face_enhance_check.isChecked(),
            suffix=self.suffix_input.text(),
            ext=self.format_combo.currentText()
        )

        self.processor.progress.connect(self.update_progress)
        self.processor.error.connect(self.show_error)
        self.processor.error_recovery_signal.connect(self.show_recovery_error)
        self.processor.finished.connect(self.processing_finished)
        self.processor.start()

    def processing_finished(self):
        self.enable_controls()
        self.process_btn.setText('Process Selected Sets')  # Reset button text
        if hasattr(self, 'is_cancelling') and self.is_cancelling:
            self.progress_label.setText("Processing cancelled!")
            self.is_cancelling = False
        else:
            self.progress_label.setText("Processing complete!")

    def update_progress(self, message):
        self.progress_label.setText(message)
        # Enable rich text interpretation
        self.progress_label.setTextFormat(Qt.RichText)

    def show_error(self, message):
        if hasattr(self, 'is_cancelling') and self.is_cancelling:
            # If we're cancelling, just reset the button without re-enabling other controls
            self.process_btn.setText('Process Selected Sets')
            self.process_btn.setEnabled(True)
        else:
            # Regular error handling
            self.enable_controls()
            # Only show error message if we're not cancelling
            QMessageBox.critical(self, 'Error', message)

    def show_recovery_error(self, message):
        """
        Display a modal error dialog allowing the user to Continue or Cancel.
        Triggered by ImageProcessor when a subprocess error occurs.
        """
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Real-ESRGAN Inference Error")
        msg.setText("An error occurred during processing.")
        msg.setInformativeText(message)

        continue_btn = msg.addButton("Continue", QMessageBox.ActionRole)
        cancel_btn = msg.addButton("Cancel", QMessageBox.RejectRole)

        msg.exec_()

        if msg.clickedButton() == cancel_btn:
            self.cancel_processing()
        else:
            # User chose to continue; resume the worker thread
            if self.processor:
                self.processor.resume()

    def show_model_help(self):
        try:
            help_file = Path.home() / '.local/share/esrgan-APP/Help/help.txt'
            if not help_file.exists():
                # Create directories if they don't exist
                help_file.parent.mkdir(parents=True, exist_ok=True)
                # Create help file with content
                help_content = """Real ESRGAN Model Help:
x4v3 (realesr-general-x4v3):
 • Small, efficient model optimized for general use
 • Best balance of speed and quality
 • Excellent for photos and natural images
 • Recommended starting point for most uses

x4plus (RealESRGAN_x4plus):
 • High-quality model with superior detail recovery
 • Better preservation of textures and fine details
 • Slower but produces premium results
 • Ideal for important images where quality is priority

net_x4plus (RealESRNet_x4plus):
 • Conservative enhancement approach
 • Maintains high fidelity to source material
 • Reduces risk of over-processing artifacts
 • Perfect for subtle improvements and restoration

x2plus (RealESRGAN_x2plus):
 • 2x upscaling with gentle enhancement
 • Minimizes artificial-looking results
 • More natural-looking output
 • Best for slight resolution increases

anime_6B (RealESRGAN_x4plus_anime_6B):
 • Specialized for anime and artwork
 • Preserves sharp edges and line art
 • Maintains color consistency
 • Ideal for illustrations and stylized content

Processing Tips:
 • Start with x4v3 for quick results
 • Try x4plus if you need more detail
 • Use x2plus for minimal artifacts
 • Choose anime_6B for artwork
 • Consider net_x4plus for natural preservation"""

                help_file.write_text(help_content)

            # Read help content
            help_text = help_file.read_text()

            # Convert text to HTML with proper line breaks and bold headings
            lines = help_text.split('\n')
            formatted_lines = []
            for line in lines:
                if line.strip() == 'Real ESRGAN Model Help:':  # Main header
                    line = f'<center><h2 style="color: #a553f8;">Real ESRGAN Model Help:</h2></center>'
                elif ':' in line and not line.startswith(' '):  # This catches the headings
                    line = f'<b><font color="#0080ff">{line}</font></b>'
                elif line.startswith(' •'):  # Bullet points
                    bullet = '<span style="color: #ffffff">•</span>'  # White bullet
                    text = line[2:]  # Get text after bullet
                    line = f' {bullet} <span style="color: #ff55ff">{text}</span>'  # Dark gray text
                if line.strip() == '':  # Empty line
                    formatted_lines.append('<br>')
                else:
                    formatted_lines.append(line)
            formatted_text = '<br>'.join(formatted_lines)

            # Create custom dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("ESRGAN Model Help")
            dialog.setMinimumWidth(625)
            dialog.setMinimumHeight(800)

            # Create layout
            layout = QVBoxLayout(dialog)

            # Create text display widget using QTextEdit
            text_display = QTextEdit()
            text_display.setAcceptRichText(True)
            text_display.setFont(QFont("Monospace"))
            text_display.setReadOnly(True)
            text_display.setHtml(formatted_text)

            # Add to layout
            layout.addWidget(text_display)

            # Add OK button at bottom
            button_box = QDialogButtonBox(QDialogButtonBox.Ok)
            button_box.accepted.connect(dialog.accept)
            layout.addWidget(button_box)

            dialog.exec_()

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not load help file: {str(e)}")


def main(argv=None):
    """
    Main entry point of the application.

    This function initializes the Qt application, defines a custom style sheet for
    the application user interface, and applies it. It then creates an instance
    of the main GUI, shows it, and starts the application's event loop.

    The style sheet customizes the appearance of various Qt widgets, such as
    QMainWindow, QLabel, QPushButton, QComboBox, QListWidget, QLineEdit,
    QSpinBox, QCheckBox, and QProgressBar. The customization includes colors,
    borders, hover and pressed states, and other style properties.

    The function also ensures the proper termination of the application by
    calling `sys.exit` with the result of the application's main event loop.

    :raises SystemExit: If the application fails to run or exits.
    """

    if argv is None:
        argv = sys.argv
    app = QApplication(sys.argv)

    QCoreApplication.setApplicationName("ESRGAN GUI")
    QCoreApplication.setApplicationVersion("0.5.0")

    parser = QCommandLineParser()
    parser.addHelpOption()
    parser.addVersionOption()
    parser.process(app)

    # Define the style sheet
    '''
    The style sheet defines the appearance of various Qt widgets. It includes
    colors, borders, hover and pressed states, and other style propertie.
    
    The colors used are:
        Main background: #2b2b2b (dark gray)
        Widget background: #3c3c3c (lighter gray)
        Accent color: #3d8ec9 (blue)
        Text: #ffffff (white)
        Borders: #474d54 (medium gray)  
        
        You can customize any of these colors by changing the hex codes. Some tips for customizing:
        
        1. To change the main theme color, replace #3d8ec9 with your preferred color
        2. For lighter/darker variants, adjust the hex values up/down
        3. Test contrast ratios to ensure text remains readable
        4. Use CSS-like selectors to target specific widgets
    '''
    style_sheet = """
        QMainWindow {
            background-color: #2a2e32;
        }

        QLabel {
            color: #ffffff;
            font-size: 16px;
        }

        QLabel:disabled {
            color: #3e4247;
        }

        QPushButton {
            background-color: #31363b;
            color: white;
            border: 1px solid #6e7175;
            padding: 5px 15px;
            border-radius: 4px;
            font-weight: bold;
        }

        QPushButton:hover {
            background-color: #31363b;
            border: 1px solid #3d8ec9;
        }

        QPushButton:pressed {
            background-color: #206592;
        }

        QPushButton:disabled {
            background-color: #2f3338;
            color: #3e4247;
        }

        QComboBox {
            background-color: #2f3338;
            color: white;
            border: 1px solid #474d54;
            border-radius: 3px;
            padding: 3px;
        }

        QComboBox:hover {
            border: 1px solid #3d8ec9;
        }

        QComboBox:disabled {
            background-color: #2f3338;
            color: #3e4247;
        }

        QListWidget {
            background-color: #1b1e20;
            color: white;
            border: 1px solid #474d54;
            border-radius: 3px;
        }

        QListWidget::item:selected {
            background-color: #3d8ec9;
        }

        QListWidget::item:hover {
            border: 1px solid #3d8ec9;
            background-color: #213e4c;
        }
        
        QListWidget:disabled {
            background-color: #2f3338;
            border: 1px solid #3e4247;
        }

        QListWidget::item:disabled {
            color: #3e4247;  /* Gray out unselected items */
        }

        QListWidget::item:disabled:selected {
            color: #a0a0a0;  /* Keep selected items more visible but still indicate disabled */
            background-color: #2a5773;  /* Darker version of the selection color */
        }

        QListWidget:hover {
            border: 1px solid #3d8ec9;
            
        }


        QLineEdit {
            background-color: #1b1e20;
            color: white;
            border: 1px solid #474d54;
            border-radius: 3px;
            padding: 3px;
        }

        QLineEdit:disabled {
            background-color: #2f3338;
            color: #3e4247;       
        }

        QLineEdit:focus {
            border: 1px solid #3d8ec9;
        }

        QSpinBox {
            background-color: #1b1e20;
            color: white;
            border: 1px solid #474d54;
            border-radius: 3px;
            padding: 3px;
            min-width: 60px;
            max-width: 70px;
            border-right: none;
        }
        
        QSpinBox:disabled {
            background-color: #2f3338;
            color: #3e4247;
            border: 1px solid #3e4247;
        }
        
        QSpinBox::up-button {
            background-color: #2d3235;
            subcontrol-origin: border;
            subcontrol-position: top right;
            width: 14px;
            color: white;
            /* Add border to match main spinbox */
            border: 1px solid #474d54;
            border-left: none;
            border-bottom: none;
            border-top-right-radius: 3px;
        }

        QSpinBox::down-button {
            background-color: #2d3235;
            subcontrol-origin: border;
            subcontrol-position: bottom right;
            width: 14px;
            color: white;
            /* Add border to match main spinbox */
            border: 1px solid #474d54;
            border-left: none;
            border-bottom-right-radius: 3px;
        }

        QSpinBox::up-button:disabled, QSpinBox::down-button:disabled {
            background-color: #2f3338;
            border: 1px solid #3e4247;
        }

        QSpinBox::up-arrow {
            background-color: transparent;
            border-left: 3px solid none;
            border-right: 3px solid none;
            border-bottom: 3px solid white;
            width: 0px;
            height: 0px;
        }
        
        QSpinBox::down-arrow {
            background-color: transparent;
            border-left: 3px solid none;
            border-right: 3px solid none;
            border-top: 3px solid white;
            width: 0px;
            height: 0px;
        }
        
        QSpinBox::up-arrow:disabled {
            border-bottom: 3px solid #3e4247;
        }
        
        QSpinBox::down-arrow:disabled {
            border-top: 3px solid #3e4247;
        }

        QSpinBox:hover {
            border: 1px solid #3d8ec9;
            background-color: #3d8ec9;
            padding: 3px;
            border-radius: 3px;
        }                    

        QDoubleSpinBox {
            background-color: #1b1e20;
            color: white;
            border: 1px solid #474d54;
            border-radius: 3px;
            padding: 3px;
            min-width: 60px;
            max-width: 70px;
            border-right: none;
        }
        
        QDoubleSpinBox:disabled {
            background-color: #2f3338;
            color: #3e4247;
            border: 1px solid #3e4247;
        }
        
        QDoubleSpinBox::up-button {
            background-color: #2d3235;
            subcontrol-origin: border;
            subcontrol-position: top right;
            width: 14px;
            color: white;
            border: 1px solid #474d54;
            border-left: none;
            border-bottom: none;
            border-top-right-radius: 3px;
        }

        QDoubleSpinBox::down-button {
            background-color: #2d3235;
            subcontrol-origin: border;
            subcontrol-position: bottom right;
            width: 14px;
            color: white;
            border: 1px solid #474d54;
            border-left: none;
            border-bottom-right-radius: 3px;
        }

        QDoubleSpinBox::up-button:disabled, QDoubleSpinBox::down-button:disabled {
            background-color: #2f3338;
            border: 1px solid #3e4247;
        }

        QDoubleSpinBox::up-arrow {
            background-color: transparent;
            border-left: 3px solid none;
            border-right: 3px solid none;
            border-bottom: 3px solid white;
            width: 0px;
            height: 0px;
        }
        
        QDoubleSpinBox::down-arrow {
            background-color: transparent;
            border-left: 3px solid none;
            border-right: 3px solid none;
            border-top: 3px solid white;
            width: 0px;
            height: 0px;
        }
        
        QDoubleSpinBox::up-arrow:disabled {
            border-bottom: 3px solid #3e4247;
        }
        
        QDoubleSpinBox::down-arrow:disabled {
            border-top: 3px solid #3e4247;
        }

        QDoubleSpinBox:hover {
            border: 1px solid #3d8ec9;
            background-color: #3d8ec9;
            padding: 3px;
            border-radius: 3px;
        }

        QCheckBox {
            color: white;
            spacing: 5px;
        }

        QCheckBox:disabled {
            background-color: #2f3338;
            color: #3e4247;            
        }

        QCheckBox::indicator {
            width: 16px;
            height: 16px;
        }

        QCheckBox::indicator:unchecked {
            background-color: #1b1e20;
            border: 1px solid #474d54;
        }

        /*
        QCheckBox::indicator:checked {
            background-color: #3e738f;
            border: 1px solid #64c8ff;
            image: url(self.checkmark);
        }
        */

        QProgressBar {
            border: 1px solid #474d54;
            border-radius: 3px;
            background-color: #2a2e32;
            color: white;
            text-align: center;
        }

        QProgressBar::chunk {
            background-color: #3d8ec9;
        }
    """

    # Apply the style sheet to the application
    app.setStyleSheet(style_sheet)

    gui = ESRGANGui()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()