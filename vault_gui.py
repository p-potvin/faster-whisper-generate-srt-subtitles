import sys
import os
import time
from typing import List, Optional
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QCheckBox, QComboBox,
    QProgressBar, QTextEdit, QFileDialog, QFrame, QSpinBox, QDoubleSpinBox,
    QSizePolicy, QScrollArea, QSpacerItem
)
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QPixmap, QIcon, QFont, QColor, QPalette,  QKeySequence, QShortcut

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vault-themes"))
from qt_exporter import QtThemeExporter

from vault_enhancer import core, utils, media


# ─────────────────────────────────────────────────────────────────────────────
# Worker thread
# ─────────────────────────────────────────────────────────────────────────────

class TranscriptionWorker(QThread):
    finished = Signal(list)
    progress = Signal(str)
    progress_percent = Signal(int)
    error = Signal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.is_running = True

    def run(self):
        try:
            input_path = self.params.get('input_file', '')
            if not input_path:
                self.error.emit("No input path provided.")
                return

            # Determine files to process
            files_to_process = []
            if os.path.isdir(input_path):
                extensions = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm', '.mp3', '.wav', '.m4a']
                files_to_process = list(media.find_media_files(input_path, extensions))
                if not files_to_process:
                    self.error.emit(f"No media files found in directory: {input_path}")
                    return
            else:
                files_to_process = [input_path]

            total_files = len(files_to_process)
            all_outputs = []

            for i, file_path in enumerate(files_to_process):
                if not self.is_running:
                    break

                current_params = dict(self.params)
                current_params['input_file'] = file_path
                
                file_basename = os.path.basename(file_path)
                batch_prefix = f"[{i+1}/{total_files}] {file_basename}: " if total_files > 1 else ""
                
                self.progress.emit(f"{batch_prefix}Initializing…")

                def progress_cb(text, percent):
                    if text:
                        self.progress.emit(f"{batch_prefix}{text}")
                    if percent is not None:
                        # Scale percent to be within the file's portion of the total progress
                        overall_pct = int((i * 100 + percent) / total_files)
                        self.progress_percent.emit(overall_pct)

                current_params['progress_callback'] = progress_cb
                current_params.pop('continue_on_error', None)
                
                try:
                    output_paths = core.transcribe_video(**current_params)
                    all_outputs.extend(output_paths)
                except Exception as e:
                    if self.params.get('continue_on_error', False):
                        self.progress.emit(f"Error processing {file_basename}: {str(e)}")
                        continue
                    else:
                        raise e
                finally:
                    # Cleanup GPU state between files in batch to prevent illegal memory access
                    try:
                        import torch
                        if torch.cuda.is_available():
                            try:
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                            except:
                                # Context might already be corrupted; ignore and move on
                                pass
                    except ImportError:
                        pass
                    time.sleep(2.0)

            self.finished.emit(all_outputs)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")
        except BaseException as e:
            self.error.emit(f"Critical crash: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# Log stream redirect
# ─────────────────────────────────────────────────────────────────────────────

class LogStream:
    def __init__(self, log_fn):
        self.log_fn = log_fn

    def write(self, text):
        if text.strip():
            import re
            clean = re.sub(r'\x1b\[[0-9;]*[mK]', '', text)
            if clean.strip():
                self.log_fn(clean.strip())

    def flush(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

class VaultWindow(QMainWindow):
    log_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vault Video Enhancer")
        self.setMinimumSize(1100, 700)
        self.setWindowState(Qt.WindowMaximized)

        self.exporter = QtThemeExporter()
        self.themes = self.exporter.get_all_themes()
        self.current_theme = self.themes[0]

        self.init_ui()
        self.apply_vault_styles()
        self.setAcceptDrops(True)

        self.log_signal.connect(self.log)
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = LogStream(self.log_signal.emit)
        sys.stderr = LogStream(self.log_signal.emit)

    # ── UI construction ──────────────────────────────────────────────────────

    def init_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 10, 16, 10)
        root_layout.setSpacing(10)

        # ── Header ────────────────────────────────────────────────────────
        root_layout.addWidget(self._build_header())
        root_layout.addWidget(self._make_separator())

        # ── Main split (config | monitor) ────────────────────────────────
        split = QHBoxLayout()
        split.setSpacing(12)
        split.addWidget(self._build_config_panel(), stretch=4)
        split.addWidget(self._build_monitor_panel(), stretch=6)
        root_layout.addLayout(split, stretch=1)

        # ── Footer ────────────────────────────────────────────────────────
        root_layout.addWidget(self._make_separator())
        footer = QLabel("© 2026 VaultWares — All processing is local. No data leaves your machine.")
        footer.setAlignment(Qt.AlignCenter)
        footer.setObjectName("FooterLabel")
        root_layout.addWidget(footer)

    def _build_header(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 4, 0, 4)

        # Logo
        logo_label = QLabel()
        logo_path = "vault-themes/Brand/minimal-logos/vaultwares-minimal-gold-filled.png"
        logo_pix = QPixmap(logo_path)
        if not logo_pix.isNull():
            logo_label.setPixmap(logo_pix.scaled(28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.setWindowIcon(QIcon(logo_path))
        logo_label.setObjectName("LogoLabel")
        logo_label.setFixedSize(36, 36)
        logo_label.setAlignment(Qt.AlignCenter)

        # Title
        self.title_label = QLabel()
        self.title_label.setTextFormat(Qt.RichText)
        self.title_label.setStyleSheet("font-size: 20px; font-weight: 300; letter-spacing: 1px;")

        # Separator spacer
        spacer = QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)

        # Theme selector
        theme_label = QLabel("Theme")
        theme_label.setObjectName("StatusLabel")
        self.theme_combo = QComboBox()
        self.theme_combo.setFixedWidth(220)
        for t in self.themes:
            self.theme_combo.addItem(t.name)
        self.theme_combo.setCurrentText(self.current_theme.name)
        self.theme_combo.currentTextChanged.connect(self.change_theme)

        layout.addWidget(logo_label)
        layout.addSpacing(10)
        layout.addWidget(self.title_label)
        layout.addItem(spacer)
        layout.addWidget(theme_label)
        layout.addSpacing(6)
        layout.addWidget(self.theme_combo)

        return w

    def _build_config_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("ConfigPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)

        # Title
        title = QLabel("PIPELINE CONFIGURATION")
        title.setObjectName("SectionTitleConfig")
        layout.addWidget(title)

        # Input path ─────────────────────────────────────────────────────
        layout.addWidget(self._field_label("Input File or Directory"))
        path_row = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Drop a video/audio file or folder path…")
        
        browse_file_btn = QPushButton("File…")
        browse_file_btn.setFixedWidth(70)
        browse_file_btn.clicked.connect(self.browse_input_file)
        
        browse_folder_btn = QPushButton("Folder…")
        browse_folder_btn.setFixedWidth(70)
        browse_folder_btn.clicked.connect(self.browse_input_folder)
        
        path_row.addWidget(self.input_edit)
        path_row.addWidget(browse_file_btn)
        path_row.addWidget(browse_folder_btn)
        layout.addLayout(path_row)

        layout.addWidget(self._make_separator())

        # Core options row ────────────────────────────────────────────────
        core_row = QHBoxLayout()
        core_row.setSpacing(12)

        v_engine = QVBoxLayout()
        v_engine.addWidget(self._field_label("ASR Engine"))
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["parakeet", "whisper"])
        self.engine_combo.setCurrentText("parakeet")
        v_engine.addWidget(self.engine_combo)
        core_row.addLayout(v_engine)

        v_lang = QVBoxLayout()
        v_lang.addWidget(self._field_label("Target Languages (comma-separated)"))
        self.lang_edit = QLineEdit("en")
        self.lang_edit.setPlaceholderText("en, fr, de…")
        v_lang.addWidget(self.lang_edit)
        core_row.addLayout(v_lang, stretch=2)

        layout.addLayout(core_row)

        # Translation row ─────────────────────────────────────────────────
        trans_row = QHBoxLayout()
        trans_row.setSpacing(12)

        v_api = QVBoxLayout()
        v_api.addWidget(self._field_label("Translator Backend"))
        self.api_combo = QComboBox()
        self.api_combo.addItems(["deep-translator", "googletrans"])
        v_api.addWidget(self.api_combo)
        trans_row.addLayout(v_api)

        v_mode = QVBoxLayout()
        v_mode.addWidget(self._field_label("Translate Mode"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["all", "non-target"])
        v_mode.addWidget(self.mode_combo)
        trans_row.addLayout(v_mode)

        layout.addLayout(trans_row)

        # Limits row ──────────────────────────────────────────────────────
        limits_row = QHBoxLayout()
        limits_row.setSpacing(12)

        v_src = QVBoxLayout()
        v_src.addWidget(self._field_label("Source Language"))
        self.src_lang_edit = QLineEdit()
        self.src_lang_edit.setPlaceholderText("Auto-detect")
        v_src.addWidget(self.src_lang_edit)
        limits_row.addLayout(v_src)

        v_dur = QVBoxLayout()
        v_dur.addWidget(self._field_label("Max Duration"))
        self.max_duration = QSpinBox()
        self.max_duration.setRange(0, 86400)
        self.max_duration.setValue(7200)
        self.max_duration.setSuffix(" s")
        v_dur.addWidget(self.max_duration)
        limits_row.addLayout(v_dur)

        v_delay = QVBoxLayout()
        v_delay.addWidget(self._field_label("Audio Delay"))
        self.delay_spin = QSpinBox()
        self.delay_spin.setRange(-10000, 10000)
        self.delay_spin.setValue(0)
        self.delay_spin.setSuffix(" ms")
        v_delay.addWidget(self.delay_spin)
        limits_row.addLayout(v_delay)

        layout.addLayout(limits_row)

        layout.addWidget(self._make_separator())

        # Toggles ─────────────────────────────────────────────────────────
        toggles_row = QHBoxLayout()
        toggles_row.setSpacing(16)

        col_l = QVBoxLayout()
        self.vocal_check = QCheckBox("Vocal Isolation (Demucs)")
        self.vocal_check.setChecked(True)
        self.skip_orig_check = QCheckBox("Skip Original SRT")
        col_l.addWidget(self.vocal_check)
        col_l.addWidget(self.skip_orig_check)

        col_r = QVBoxLayout()
        self.overwrite_check = QCheckBox("Overwrite Existing Files")
        self.continue_err_check = QCheckBox("Continue on Error")
        col_r.addWidget(self.overwrite_check)
        col_r.addWidget(self.continue_err_check)

        toggles_row.addLayout(col_l)
        toggles_row.addLayout(col_r)
        layout.addLayout(toggles_row)

        layout.addStretch()

        # Start button ────────────────────────────────────────────────────
        self.start_btn = QPushButton("▶  INITIATE PIPELINE")
        self.start_btn.setObjectName("PrimaryBtn")
        self.start_btn.setFixedHeight(52)
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.clicked.connect(self.start_processing)
        layout.addWidget(self.start_btn)

        return panel

    def _build_monitor_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("MonitorPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        # Header row
        monitor_row = QHBoxLayout()
        title = QLabel("VAULT ACTIVITY MONITOR")
        title.setObjectName("SectionTitleMonitor")
        monitor_row.addWidget(title)
        monitor_row.addStretch()

        self.status_badge = QLabel("IDLE")
        self.status_badge.setObjectName("TagBadge")
        monitor_row.addWidget(self.status_badge)
        layout.addLayout(monitor_row)

        # Log area
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setObjectName("LogArea")
        layout.addWidget(self.log_area, stretch=1)

        # Progress row
        progress_row = QHBoxLayout()
        progress_row.setSpacing(10)

        self.progress_label = QLabel("Ready")
        self.progress_label.setObjectName("StatusLabel")
        self.progress_label.setFixedWidth(200)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(22)

        progress_row.addWidget(self.progress_label)
        progress_row.addWidget(self.progress_bar, stretch=1)
        layout.addLayout(progress_row)

        return panel

    # ── Helpers ──────────────────────────────────────────────────────────────

        # --- Accessibility & Tooltips ---
        self.theme_combo.setToolTip("Select UI Theme")
        self.input_edit.setToolTip("Path to the video/audio file to process")
        browse_btn.setToolTip("Browse for media file (Ctrl+O)")
        self.lang_edit.setToolTip("Comma-separated target languages for translation (e.g., en, fr, es)")
        self.engine_combo.setToolTip("Transcription engine to use")
        self.api_combo.setToolTip("Translation backend service")
        self.mode_combo.setToolTip("'all' translates everything; 'non-target' only translates if original isn't the target")
        self.src_lang_edit.setToolTip("Force a specific source language if auto-detect fails")
        self.max_duration.setToolTip("Skip media longer than this (seconds)")
        self.vocal_check.setToolTip("Use Demucs to isolate vocals before transcription for better accuracy in noisy audio")
        self.skip_orig_check.setToolTip("Do not generate the SRT file for the original spoken language")
        self.overwrite_check.setToolTip("Overwrite existing SRT files if they exist")
        self.continue_err_check.setToolTip("Continue processing next files if one fails (Scan Mode only)")
        self.start_btn.setToolTip("Initiate transcription pipeline (Ctrl+Return)")

        # --- Tab Order ---
        QWidget.setTabOrder(self.theme_combo, self.input_edit)
        QWidget.setTabOrder(self.input_edit, browse_btn)
        QWidget.setTabOrder(browse_btn, self.lang_edit)
        QWidget.setTabOrder(self.lang_edit, self.engine_combo)
        QWidget.setTabOrder(self.engine_combo, self.api_combo)
        QWidget.setTabOrder(self.api_combo, self.mode_combo)
        QWidget.setTabOrder(self.mode_combo, self.src_lang_edit)
        QWidget.setTabOrder(self.src_lang_edit, self.max_duration)
        QWidget.setTabOrder(self.max_duration, self.vocal_check)
        QWidget.setTabOrder(self.vocal_check, self.skip_orig_check)
        QWidget.setTabOrder(self.skip_orig_check, self.overwrite_check)
        QWidget.setTabOrder(self.overwrite_check, self.continue_err_check)
        QWidget.setTabOrder(self.continue_err_check, self.start_btn)

        # Keyboard Shortcuts
        self.shortcut_browse = QShortcut(QKeySequence("Ctrl+O"), self)
        self.shortcut_browse.activated.connect(self.browse_input)

        self.shortcut_start = QShortcut(QKeySequence("Ctrl+Return"), self)
        self.shortcut_start.activated.connect(self.start_processing)

        self.shortcut_start_enter = QShortcut(QKeySequence("Ctrl+Enter"), self)
        self.shortcut_start_enter.activated.connect(self.start_processing)

        self.shortcut_clear = QShortcut(QKeySequence("Esc"), self)
        self.shortcut_clear.activated.connect(lambda: self.input_edit.clear())

        # Footer
        self.footer = QLabel("© 2026 VaultWares — Built under VaultWares Enterprise Guidelines")
        self.footer.setAlignment(Qt.AlignCenter)
        self.footer.setObjectName("FooterLabel")
        main_layout.addWidget(self.footer)


    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                # Get the first URL and convert to local file path
                path = urls[0].toLocalFile()
                self.input_edit.setText(path)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def _field_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("StatusLabel")
        return lbl

    def _make_separator(self) -> QFrame:
        line = QFrame()
        line.setObjectName("Separator")
        line.setFrameShape(QFrame.HLine)
        line.setFixedHeight(1)
        return line

    # ── Theme ─────────────────────────────────────────────────────────────────

    def change_theme(self, theme_name: str):
        for t in self.themes:
            if t.name == theme_name:
                self.current_theme = t
                break
        self.apply_vault_styles()

    def apply_vault_styles(self):
        t = self.current_theme
        self.setStyleSheet(self.exporter.generate_qss(t))
        self.title_label.setText(
            f"<span style='font-weight:300'>Vault</span>"
            f"<span style='color:{t.accent}; font-weight:600'> Video Enhancer</span>"
        )

    # ── Actions ──────────────────────────────────────────────────────────────

    def browse_input_file(self):
        path = QFileDialog.getOpenFileName(
            self, "Select Media File", "",
            "Media Files (*.mp4 *.mkv *.avi *.mov *.flv *.webm *.mp3 *.wav *.m4a);;All Files (*)"
        )[0]
        if path:
            self.input_edit.setText(path)

    def browse_input_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Folder", "")
        if path:
            self.input_edit.setText(path)

    def log(self, message: str):
        t = self.current_theme
        ts = time.strftime("%H:%M:%S")

        # Color-code known message patterns
        if any(k in message.lower() for k in ("error:", "failed", "traceback", "exception")):
            color = t.error
        elif any(k in message.lower() for k in ("success", "completed", "generated", "done")):
            color = t.success
        elif any(k in message.lower() for k in ("warning", "warn:", "skipping")):
            color = t.warning
        elif any(k in message.lower() for k in ("step ", "initiating", "extracting", "transcrib")):
            color = t.accent
        else:
            color = t.text

        ts_html = f"<span style='color:{t.muted}'>[{ts}]</span>"
        msg_html = f"<span style='color:{color}'>{message}</span>"
        self.log_area.append(f"{ts_html} {msg_html}")

    def start_processing(self):
        input_path = self.input_edit.text().strip()
        if not input_path:
            self.log(f"<span style='color:{self.current_theme.error}'>Error: No input path specified.</span>")
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
            "delay_ms": self.delay_spin.value(),
            "source_language": self.src_lang_edit.text().strip() or None,
            "overwrite": self.overwrite_check.isChecked(),
            "continue_on_error": self.continue_err_check.isChecked(),
        }

        self.start_btn.setEnabled(False)
        self.status_badge.setText("RUNNING")
        self.status_badge.setStyleSheet(
            f"background-color: {self.current_theme.success}; "
            f"color: {self.current_theme.text_inverse}; "
            "border-radius: 4px; padding: 1px 8px; font-size: 10px; font-weight: 700;"
        )
        self.progress_bar.setValue(5)
        self.progress_label.setText("Initializing…")
        self.log(f"Pipeline started — {os.path.basename(input_path)}")

        self.worker = TranscriptionWorker(params)
        self.worker.progress.connect(self._on_progress_text)
        self.worker.progress_percent.connect(self._on_progress_pct)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def _on_progress_text(self, text: str):
        self.progress_label.setText(text[:40] + "…" if len(text) > 40 else text)
        self.log(text)

    def _on_progress_pct(self, pct: int):
        self.progress_bar.setValue(pct)

    def on_error(self, message: str):
        self.log(f"<span style='color:{self.current_theme.error}'>ERROR: {message}</span>")
        self.status_badge.setText("FAILED")
        self.status_badge.setStyleSheet(
            f"background-color: {self.current_theme.error}; "
            f"color: {self.current_theme.text_inverse}; "
            "border-radius: 4px; padding: 1px 8px; font-size: 10px; font-weight: 700;"
        )
        self.start_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Error — see log")

    def on_finished(self, outputs: list):
        t = self.current_theme
        self.log(
            f"<span style='color:{t.success}'>✓ Pipeline complete — "
            f"{len(outputs)} file(s) generated.</span>"
        )
        for p in outputs:
            self.log(f"<span style='color:{t.text_muted}'>  • {os.path.basename(p)}</span>")

        self.status_badge.setText("DONE")
        self.status_badge.setStyleSheet(
            f"background-color: {t.accent}; "
            f"color: {t.text_inverse}; "
            "border-radius: 4px; padding: 1px 8px; font-size: 10px; font-weight: 700;"
        )
        self.start_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.progress_label.setText("Complete")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = VaultWindow()
    window.show()
    sys.exit(app.exec())
