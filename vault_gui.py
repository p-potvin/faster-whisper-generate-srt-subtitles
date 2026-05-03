import sys
import os
import time
from typing import List, Optional
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QCheckBox, QComboBox, 
    QProgressBar, QTextEdit, QFileDialog, QFrame, QSpinBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt, QThread, Signal

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vault-themes"))
from qt_exporter import QtThemeExporter

# Import core logic
from vault_enhancer import core, utils, media

# VaultWares Theme Tokens are centralized in vault-themes module.

class TranscriptionWorker(QThread):
    finished = Signal(list)
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            self.progress.emit("--- Initializing Media Pipeline ---")
            output_paths = core.transcribe_video(**self.params)
            self.finished.emit(output_paths)
        except Exception as e:
            self.error.emit(str(e))

class VaultWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vault Video Enhancer")
        self.setMinimumSize(900, 800)
        self.exporter = QtThemeExporter()
        self.themes = self.exporter.get_all_themes()
        self.current_theme = self.themes[0]  #Default:Solarized Light Revisited
        self.init_ui()
        self.apply_vault_styles()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)

        # Header
        header_layout = QHBoxLayout()
        logo_label = QLabel("V")
        logo_label.setFixedSize(40, 40)
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setObjectName("LogoLabel")
        
        self.title_label = QLabel(f"Media <span style='color: {self.current_theme.accent}'>Transcriber</span>")
        self.title_label.setStyleSheet("font-size: 24px; font-weight: 500;")
        self.title_label.setTextFormat(Qt.RichText)

        self.theme_combo = QComboBox()
        for t in self.themes:
            self.theme_combo.addItem(t.name)
        self.theme_combo.setCurrentText(self.current_theme.name)
        self.theme_combo.currentTextChanged.connect(self.change_theme)

        header_layout.addWidget(logo_label)
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(QLabel("Theme:"))
        header_layout.addWidget(self.theme_combo)
        main_layout.addLayout(header_layout)

        # Main Split
        content_layout = QHBoxLayout()
        
        # --- Config Column ---
        config_scroll = QFrame()
        config_scroll.setObjectName("ConfigPanel")
        config_layout = QVBoxLayout(config_scroll)
        config_layout.setSpacing(15)

        config_title = QLabel("PIPELINE CONFIGURATION")
        config_title.setObjectName("SectionTitleConfig")
        config_layout.addWidget(config_title)

        # Input Path
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Path to video/audio or folder...")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_input)
        
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.input_edit)
        path_layout.addWidget(browse_btn)
        config_layout.addWidget(QLabel("Input File / Scan Directory"))
        config_layout.addLayout(path_layout)

        # Core Options Grid
        core_grid = QHBoxLayout()
        
        # Languages
        v_lang = QVBoxLayout()
        self.lang_edit = QLineEdit("en")
        v_lang.addWidget(QLabel("Target Languages"))
        v_lang.addWidget(self.lang_edit)
        core_grid.addLayout(v_lang)

        # Engine
        v_engine = QVBoxLayout()
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["parakeet", "whisper"])
        self.engine_combo.setCurrentText("parakeet")
        v_engine.addWidget(QLabel("Engine"))
        v_engine.addWidget(self.engine_combo)
        core_grid.addLayout(v_engine)
        
        config_layout.addLayout(core_grid)

        # Advanced Settings
        config_layout.addWidget(self.create_separator())

        # Translate API & Mode
        trans_grid = QHBoxLayout()
        
        v_api = QVBoxLayout()
        self.api_combo = QComboBox()
        self.api_combo.addItems(["deep-translator", "googletrans"])
        v_api.addWidget(QLabel("Translator Backend"))
        v_api.addWidget(self.api_combo)
        trans_grid.addLayout(v_api)

        v_mode = QVBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["all", "non-target"])
        v_mode.addWidget(QLabel("Translate Mode"))
        v_mode.addWidget(self.mode_combo)
        trans_grid.addLayout(v_mode)
        
        config_layout.addLayout(trans_grid)

        # Source Language & Max Duration
        # Source Language & Max Duration
        limit_grid = QHBoxLayout()
        
        v_src = QVBoxLayout()
        self.src_lang_edit = QLineEdit()
        self.src_lang_edit.setPlaceholderText("Auto-detect")
        v_src.addWidget(QLabel("Source Language (Override)"))
        v_src.addWidget(self.src_lang_edit)
        limit_grid.addLayout(v_src)
        limit_grid.setSpacing(20) # Spacing between Source Language and Max Duration

        v_dur = QVBoxLayout()
        self.max_duration = QSpinBox()
        self.max_duration.setRange(0, 86400)
        self.max_duration.setValue(7200)
        self.max_duration.setSuffix("s")
        v_dur.addWidget(QLabel("Max Media Duration"))
        v_dur.addWidget(self.max_duration)
        limit_grid.addLayout(v_dur)

        config_layout.addLayout(limit_grid)
        config_layout.addSpacing(10) # Spacing before toggles

        # Toggles Grid
        toggles_grid = QHBoxLayout()
        
        v_toggles_l = QVBoxLayout()
        self.vocal_check = QCheckBox("Vocal Isolation (Demucs)")
        self.vocal_check.setChecked(True)
        self.skip_orig_check = QCheckBox("Skip Original SRT")
        v_toggles_l.addWidget(self.vocal_check)
        v_toggles_l.addWidget(self.skip_orig_check)
        
        v_toggles_r = QVBoxLayout()
        self.overwrite_check = QCheckBox("Overwrite Files")
        self.continue_err_check = QCheckBox("Continue on Error")
        v_toggles_r.addWidget(self.overwrite_check)
        v_toggles_r.addWidget(self.continue_err_check)
        
        toggles_grid.addLayout(v_toggles_l)
        toggles_grid.addLayout(v_toggles_r)
        config_layout.addLayout(toggles_grid)

        # Start Button
        self.start_btn = QPushButton("INITIATE PIPELINE")
        self.start_btn.setObjectName("PrimaryBtn")
        self.start_btn.setFixedHeight(55)
        self.start_btn.clicked.connect(self.start_processing)
        config_layout.addWidget(self.start_btn)

        config_layout.addStretch()
        content_layout.addWidget(config_scroll, 4)

        # --- Monitor Column ---
        monitor_frame = QFrame()
        monitor_frame.setObjectName("MonitorPanel")
        monitor_layout = QVBoxLayout(monitor_frame)
        
        monitor_title = QLabel("VAULT ACTIVITY MONITOR")
        monitor_title.setObjectName("SectionTitleMonitor")
        monitor_layout.addWidget(monitor_title)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setObjectName("LogArea")
        monitor_layout.addWidget(self.log_area)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(12)
        monitor_layout.addWidget(self.progress_bar)

        content_layout.addWidget(monitor_frame, 5)
        main_layout.addLayout(content_layout)

        # Footer
        self.footer = QLabel("© 2026 VaultWares — Built under VaultWares Enterprise Guidelines")
        self.footer.setAlignment(Qt.AlignCenter)
        self.footer.setObjectName("FooterLabel")
        main_layout.addWidget(self.footer)

    def create_separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet(f"background-color: rgba(74, 84, 89, 0.1); max-height: 1px;")
        return line

    def change_theme(self, theme_name):
        for t in self.themes:
            if t.name == theme_name:
                self.current_theme = t
                break
        self.apply_vault_styles()

    def apply_vault_styles(self):
        self.setStyleSheet(self.exporter.generate_qss(self.current_theme))
        self.title_label.setText(f"Media <span style='color: {self.current_theme.accent}'>Transcriber</span>")
        # Update colors that need manual override if QSS doesn't catch them or for rich text
        pass

    def browse_input(self):
        path = QFileDialog.getOpenFileName(self, "Select Media", "", "Media Files (*.mp4 *.mkv *.avi *.mov *.flv *.webm *.mp3 *.wav *.m4a);;All Files (*)")[0]
        if path:
            self.input_edit.setText(path)

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_area.append(f"<span style='color: {VAULT_MUTED}'>[{timestamp}]</span> {message}")

    def start_processing(self):
        input_path = self.input_edit.text()
        if not input_path:
            self.log("<span style='color: {VAULT_BURGUNDY}'>Error: No input path specified.</span>")
            return

        params = {
            "input_file": input_path,
            "languages": [l.strip() for l in self.lang_edit.text().split(",") if l.strip()],
            "engine": self.engine_combo.currentText(),
            "translate_api": self.api_combo.currentText(),
            "translate_mode": self.mode_combo.currentText(),
            "skip_vocal_isolation": not self.vocal_check.isChecked(),
            "skip_original": self.skip_orig_check.isChecked(),
            "max_duration": self.max_duration.value(),
            "source_language": self.src_lang_edit.text().strip() or None,
            "overwrite": self.overwrite_check.isChecked(),
            # "continue_on_error" is used in the script's loop, core.py handle single files. 
            # If scan_dir was implemented in GUI, we'd use it there.
        }

        self.start_btn.setEnabled(False)
        self.progress_bar.setValue(10)
        self.log(f"Initiating Vault Pipeline for: {os.path.basename(input_path)}")
        
        self.worker = TranscriptionWorker(params)
        self.worker.progress.connect(self.log)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_error(self, message):
        self.log(f"<span style='color: {VAULT_BURGUNDY}'>ERROR: {message}</span>")
        self.start_btn.setEnabled(True)
        self.progress_bar.setValue(0)

    def on_finished(self, outputs):
        self.log(f"<span style='color: {VAULT_GREEN}'>SUCCESS: Generated {len(outputs)} files.</span>")
        for p in outputs:
            self.log(f" &nbsp;&nbsp;• {os.path.basename(p)}")
        self.start_btn.setEnabled(True)
        self.progress_bar.setValue(100)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VaultWindow()
    window.show()
    sys.exit(app.exec())
