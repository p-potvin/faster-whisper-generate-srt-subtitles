import sys
import os
import time
from typing import List, Optional
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QCheckBox, QComboBox,
    QProgressBar, QTextEdit, QFileDialog, QFrame, QSpinBox, QDoubleSpinBox,
    QSizePolicy, QScrollArea, QSpacerItem, QSplitter
)
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QPixmap, QIcon, QFont, QColor, QPalette,  QKeySequence, QShortcut, QTextCursor

themes_dir = os.path.join(os.path.dirname(__file__), "vaultwares-themes")

sys.path.insert(0, themes_dir)
from qt_exporter import QtThemeExporter

from vaultwares_media_processing import core, utils, media


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
                    if "Stylization" in current_params.get("task_mode", ""):
                        output_paths = core.stylize_video(
                            input_file=current_params["input_file"], 
                            prompt=current_params.get("prompt", ""), 
                            negative_prompt=current_params.get("neg_prompt", ""),
                            progress_callback=current_params.get("progress_callback")
                        )
                    else:
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
            clean = clean.strip()
            if not clean:
                return
            
            # Suppress NeMo, PyTorch, and OneLogger warnings
            lower_clean = clean.lower()
            if "[nemo w" in lower_clean or "megatron_init" in lower_clean or "nv_one_logger" in lower_clean:
                return
            if "warning:nv_one_logger" in lower_clean or "[nemo i" in lower_clean:
                return

            self.log_fn(clean)

    def flush(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────


# ── Localization ─────────────────────────────────────────────────────────

UI_STRINGS = {
    "en": {
        "theme": "Theme",
        "lang_switch": "EN",
        "config_title": "PIPELINE CONFIGURATION",
        "input_label": "Input File or Directory",
        "input_placeholder": "Drop a video/audio file or folder path…",
        "file_btn": "File…",
        "folder_btn": "Folder…",
        "engine_label": "ASR Engine",
        "target_lang_label": "Target Languages (comma-separated)",
        "target_lang_placeholder": "en, fr, de…",
        "api_label": "Translator Backend",
        "mode_label": "Translate Mode",
        "src_lang_label": "Source Language",
        "src_lang_placeholder": "Auto-detect",
        "max_dur_label": "Max Duration",
        "max_dur_special": "None",
        "delay_label": "Audio Delay",
        "isolate_check": "Isolate Vocals",
        "skip_orig_check": "Skip Original SRT",
        "overwrite_check": "Overwrite Existing Files",
        "continue_check": "Continue on Error",
        "start_btn": "▶  INITIATE PIPELINE",
        "monitor_title": "ACTIVITY MONITOR",
        "ready": "Ready",
        "footer": "© 2026 VaultWares — Built under VaultWares Enterprise Guidelines",
        "idle": "IDLE",
        "hide": "Hide",
        "show": "Show",
        "tt_theme": "Select UI Theme",
        "tt_input": "Path to the video/audio file to process",
        "tt_file": "Browse for media file (Ctrl+O)",
        "tt_folder": "Browse for folder",
        "tt_lang": "Comma-separated target languages for translation (e.g., en, fr, es)",
        "tt_engine": "Transcription engine to use",
        "tt_api": "Translation backend service",
        "tt_mode": "'all' translates everything; 'non-target' only translates if original isn't the target",
        "tt_src": "Force a specific source language if auto-detect fails",
        "tt_dur": "Skip media longer than this (seconds)",
        "tt_iso": "Use Demucs to isolate vocals before transcription for better accuracy in noisy audio",
        "tt_skip": "Do not generate the SRT file for the original spoken language",
        "tt_over": "Overwrite existing SRT files if they exist",
        "tt_cont": "Continue processing next files if one fails (Scan Mode only)",
        "tt_start": "Initiate transcription pipeline (Ctrl+Return)",
        "running": "RUNNING",
        "failed": "FAILED",
        "done": "DONE"
    },
    "qc": {
        "theme": "Thème",
        "lang_switch": "QC",
        "config_title": "CONFIGURATION DU PIPELINE",
        "input_label": "Fichier ou dossier source",
        "input_placeholder": "Déposer un fichier ou un dossier…",
        "file_btn": "Fichier…",
        "folder_btn": "Dossier…",
        "engine_label": "Moteur ASR",
        "target_lang_label": "Langues cibles (séparées par virgules)",
        "target_lang_placeholder": "en, fr, de…",
        "api_label": "Service de traduction",
        "mode_label": "Mode de traduction",
        "src_lang_label": "Langue source",
        "src_lang_placeholder": "Détection auto",
        "max_dur_label": "Durée max",
        "max_dur_special": "Aucune",
        "delay_label": "Délai audio",
        "isolate_check": "Isoler les voix",
        "skip_orig_check": "Ignorer le SRT original",
        "overwrite_check": "Écraser les fichiers existants",
        "continue_check": "Continuer sur erreur",
        "start_btn": "▶  LANCER LE PIPELINE",
        "monitor_title": "MONITEUR D'ACTIVITÉ",
        "ready": "Prêt",
        "footer": "© 2026 VaultWares — Tout le traitement est local. Aucune donnée ne quitte votre machine.",
        "idle": "INACTIF",
        "hide": "Cacher",
        "show": "Montrer",
        "tt_theme": "Sélectionner le thème de l'interface",
        "tt_input": "Chemin du fichier/dossier à traiter",
        "tt_file": "Parcourir pour un fichier (Ctrl+O)",
        "tt_folder": "Parcourir pour un dossier",
        "tt_lang": "Langues cibles séparées par des virgules (ex: en, fr, es)",
        "tt_engine": "Moteur de transcription à utiliser",
        "tt_api": "Service de traduction à utiliser",
        "tt_mode": "'all' traduit tout; 'non-target' traduit seulement si l'original n'est pas la cible",
        "tt_src": "Forcer une langue source spécifique si la détection automatique échoue",
        "tt_dur": "Ignorer les médias plus longs que ceci (secondes)",
        "tt_iso": "Utiliser Demucs pour isoler les voix avant la transcription pour une meilleure précision",
        "tt_skip": "Ne pas générer le fichier SRT pour la langue parlée originale",
        "tt_over": "Écraser les fichiers SRT existants s'ils existent",
        "tt_cont": "Continuer avec les fichiers suivants si l'un échoue (Mode scan uniquement)",
        "tt_start": "Lancer le pipeline de transcription (Ctrl+Entrée)",
        "running": "EN COURS",
        "failed": "ÉCHOUÉ",
        "done": "TERMINÉ"
    }
}

class ThemeComboBox(QComboBox):
    def setCurrentText(self, text: str):
        idx = self.findText(text)
        if idx != -1:
            self.setCurrentIndex(idx)
            return
        for i in range(self.count()):
            if text.lower() in self.itemText(i).lower():
                self.setCurrentIndex(i)
                return
        super().setCurrentText(text)


class VaultWindow(QMainWindow):
    log_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vault Video Enhancer")
        self.setMinimumSize(400, 300)
        self.setWindowState(Qt.WindowMaximized)

        self.exporter = QtThemeExporter()

        self.themes = self.exporter.get_all_themes()
        self.current_theme = self.themes[0]
        self.current_lang = "en"

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
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.setCentralWidget(self.scroll_area)

        root = QWidget()
        self.scroll_area.setWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 10, 16, 10)
        root_layout.setSpacing(10)

        # ── Header ────────────────────────────────────────────────────────
        root_layout.addWidget(self._build_header())
        root_layout.addWidget(self._make_separator())

        # ── Main split (config | monitor) ────────────────────────────────
        self.split = QSplitter(Qt.Horizontal)
        self.config_panel = self._build_config_panel()
        self.monitor_panel = self._build_monitor_panel()
        self.split.addWidget(self.config_panel)
        self.split.addWidget(self.monitor_panel)
        self.split.setStretchFactor(0, 4)
        self.split.setStretchFactor(1, 6)
        
        # When layout resizes, we might need to fold
        root_layout.addWidget(self.split, stretch=1)

        # ── Footer ────────────────────────────────────────────────────────
        root_layout.addWidget(self._make_separator())
        self.footer = QLabel(UI_STRINGS[self.current_lang]["footer"])
        self.footer.setAlignment(Qt.AlignCenter)
        self.footer.setObjectName("FooterLabel")
        root_layout.addWidget(self.footer)

        self._setup_accessibility()


    def _build_header(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 4, 0, 4)

        # Logo
        logo_label = QLabel()
        logo_path = os.path.join(themes_dir, "assets", "logos", "vaultwares-minimal-gold-filled.png")
        if not os.path.exists(logo_path):
            logo_path = "vault-themes/assets/logos/vaultwares-minimal-gold-filled.png"
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


        # Mode switch
        self.mode_btn = QPushButton("🌙/☀️")
        self.mode_btn.setToolTip("Toggle Light/Dark Theme")
        self.mode_btn.setFixedSize(40, 24)
        self.mode_btn.clicked.connect(self.toggle_mode)

        # Language switch
        self.lang_combo = QComboBox()
        self.lang_combo.setFixedWidth(60)
        self.lang_combo.addItems(["EN", "QC"])
        self.lang_combo.currentTextChanged.connect(self.change_language)

        # Theme selector
        self.theme_label = QLabel(UI_STRINGS[self.current_lang]["theme"])
        self.theme_label.setObjectName("StatusLabel")
        self.theme_combo = ThemeComboBox()
        self.theme_combo.setFixedWidth(220)
        for t in self.themes:
            self.theme_combo.addItem(t.name)

        # OS Default theme selection
        is_dark = True
        try:
            from PySide6.QtGui import QPalette
            import PySide6.QtWidgets
            app_inst = PySide6.QtWidgets.QApplication.instance()
            if app_inst:
                win_color = app_inst.palette().color(QPalette.Window).value()
                if win_color > 128:  # Simple brightness check (0-255)
                    is_dark = False
        except Exception:
            pass

        default_theme_name = "Golden Slate" if is_dark else "Codex Solar Light"
        for i, t in enumerate(self.themes):
            if default_theme_name.lower() in t.name.lower():
                self.theme_combo.setCurrentIndex(i)
                self.current_theme = t
                break

        self.theme_combo.currentTextChanged.connect(self.change_theme)
        self.header_mode_combo = self.theme_combo

        layout.addWidget(logo_label)
        layout.addSpacing(10)
        layout.addWidget(self.title_label)
        layout.addItem(spacer)
        layout.addWidget(self.mode_btn)
        layout.addSpacing(6)
        layout.addWidget(self.lang_combo)
        layout.addSpacing(6)
        layout.addWidget(self.theme_label)
        layout.addSpacing(6)
        layout.addWidget(self.theme_combo)

        return w


    def _build_config_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("ConfigPanel")
        panel.setMinimumHeight(400)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)

        # Title
        self.config_title = QLabel(UI_STRINGS[self.current_lang]["config_title"])
        title = self.config_title
        title.setObjectName("SectionTitleConfig")
        layout.addWidget(title)

        # Input path ─────────────────────────────────────────────────────
        self.input_label = self._field_label(UI_STRINGS[self.current_lang]["input_label"])
        layout.addWidget(self.input_label)
        path_row = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText(UI_STRINGS[self.current_lang]["input_placeholder"])
        
        self.browse_file_btn = QPushButton(UI_STRINGS[self.current_lang]["file_btn"])
        browse_file_btn = self.browse_file_btn
        browse_file_btn.setFixedWidth(70)
        browse_file_btn.clicked.connect(self.browse_input_file)
        
        self.browse_folder_btn = QPushButton(UI_STRINGS[self.current_lang]["folder_btn"])
        browse_folder_btn = self.browse_folder_btn
        browse_folder_btn.setFixedWidth(70)
        browse_folder_btn.clicked.connect(self.browse_input_folder)

        path_row.addWidget(self.input_edit)
        path_row.addWidget(browse_file_btn)
        path_row.addWidget(browse_folder_btn)
        layout.addLayout(path_row)

        layout.addWidget(self._make_separator())

        # Task Mode ────────────────────────────────────────────────────────
        self.task_mode_label = self._field_label("Task Mode")
        layout.addWidget(self.task_mode_label)
        self.task_mode_combo = QComboBox()
        self.task_mode_combo.addItems(["Transcription (Parakeet)", "Video Stylization (StreamDiffusion)"])
        self.task_mode_combo.currentTextChanged.connect(self._on_task_mode_changed)
        layout.addWidget(self.task_mode_combo)

        layout.addWidget(self._make_separator())

        # Stylization Row ──────────────────────────────────────────────────
        self.stylize_widget = QWidget()
        stylize_layout = QVBoxLayout(self.stylize_widget)
        stylize_layout.setContentsMargins(0, 0, 0, 0)

        self.prompt_label = self._field_label("Stylization Prompt")
        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText("e.g. 1girl, cyberpunk, masterpiece, 4k...")
        stylize_layout.addWidget(self.prompt_label)
        stylize_layout.addWidget(self.prompt_edit)

        self.neg_prompt_label = self._field_label("Negative Prompt")
        self.neg_prompt_edit = QLineEdit()
        self.neg_prompt_edit.setPlaceholderText("e.g. low quality, bad anatomy...")
        stylize_layout.addWidget(self.neg_prompt_label)
        stylize_layout.addWidget(self.neg_prompt_edit)

        layout.addWidget(self.stylize_widget)
        self.stylize_widget.setVisible(False)  # Default hidden

        # Transcription Container (Wrap the rest)
        self.transcription_widget = QWidget()
        transLayout = QVBoxLayout(self.transcription_widget)
        transLayout.setContentsMargins(0, 0, 0, 0)

        # Core options row ────────────────────────────────────────────────
        core_row = QHBoxLayout()
        core_row.setSpacing(12)

        v_engine = QVBoxLayout()
        self.engine_label = self._field_label(UI_STRINGS[self.current_lang]["engine_label"])
        v_engine.addWidget(self.engine_label)
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["parakeet"])
        self.engine_combo.setCurrentText("parakeet")
        self.engine_combo.setEnabled(False)
        v_engine.addWidget(self.engine_combo)
        core_row.addLayout(v_engine)

        v_lang = QVBoxLayout()
        self.target_lang_label = self._field_label(UI_STRINGS[self.current_lang]["target_lang_label"])
        v_lang.addWidget(self.target_lang_label)
        self.lang_edit = QLineEdit("en")
        self.lang_edit.setPlaceholderText(UI_STRINGS[self.current_lang]["target_lang_placeholder"])
        v_lang.addWidget(self.lang_edit)
        core_row.addLayout(v_lang, stretch=2)

        layout.addLayout(core_row)

        # Translation row ─────────────────────────────────────────────────
        trans_row = QHBoxLayout()
        trans_row.setSpacing(12)

        v_api = QVBoxLayout()
        self.api_label = self._field_label(UI_STRINGS[self.current_lang]["api_label"])
        v_api.addWidget(self.api_label)
        self.api_combo = QComboBox()
        self.api_combo.addItems(["deep-translator", "googletrans"])
        v_api.addWidget(self.api_combo)
        trans_row.addLayout(v_api)

        v_mode = QVBoxLayout()
        self.mode_label = self._field_label(UI_STRINGS[self.current_lang]["mode_label"])
        v_mode.addWidget(self.mode_label)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["all", "non-target"])
        v_mode.addWidget(self.mode_combo)
        trans_row.addLayout(v_mode)

        layout.addLayout(trans_row)

        # Limits row ──────────────────────────────────────────────────────
        limits_row = QHBoxLayout()
        limits_row.setSpacing(12)

        v_src = QVBoxLayout()
        self.src_lang_label = self._field_label(UI_STRINGS[self.current_lang]["src_lang_label"])
        v_src.addWidget(self.src_lang_label)
        self.src_lang_edit = QLineEdit()
        self.src_lang_edit.setPlaceholderText(UI_STRINGS[self.current_lang]["src_lang_placeholder"])
        v_src.addWidget(self.src_lang_edit)
        limits_row.addLayout(v_src)

        v_dur = QVBoxLayout()
        self.max_dur_label = self._field_label(UI_STRINGS[self.current_lang]["max_dur_label"])
        v_dur.addWidget(self.max_dur_label)
        self.max_duration = QSpinBox()
        self.max_duration.setRange(0, 1440)
        self.max_duration.setValue(0)
        self.max_duration.setSpecialValueText(UI_STRINGS[self.current_lang]["max_dur_special"])
        self.max_duration.setSuffix(" min")
        v_dur.addWidget(self.max_duration)
        limits_row.addLayout(v_dur)

        v_delay = QVBoxLayout()
        self.delay_label = self._field_label(UI_STRINGS[self.current_lang]["delay_label"])
        v_delay.addWidget(self.delay_label)
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
        self.vocal_check = QCheckBox(UI_STRINGS[self.current_lang]["isolate_check"])
        self.vocal_check.setChecked(True)
        self.skip_orig_check = QCheckBox(UI_STRINGS[self.current_lang]["skip_orig_check"])
        col_l.addWidget(self.vocal_check)
        col_l.addWidget(self.skip_orig_check)

        col_r = QVBoxLayout()
        self.overwrite_check = QCheckBox(UI_STRINGS[self.current_lang]["overwrite_check"])
        self.continue_err_check = QCheckBox(UI_STRINGS[self.current_lang]["continue_check"])
        col_r.addWidget(self.overwrite_check)
        col_r.addWidget(self.continue_err_check)

        toggles_row.addLayout(col_l)
        toggles_row.addLayout(col_r)
        transLayout.addLayout(toggles_row)

        layout.addWidget(self.transcription_widget)
        layout.addStretch()

        # Start button ────────────────────────────────────────────────────
        self.start_btn = QPushButton(UI_STRINGS[self.current_lang]["start_btn"])
        self.start_btn.setObjectName("PrimaryBtn")
        self.start_btn.setFixedHeight(52)
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.clicked.connect(self.start_processing)
        layout.addWidget(self.start_btn)

        return panel

    def _build_monitor_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("MonitorPanel")
        panel.setMinimumHeight(300)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        # Header row
        monitor_row = QHBoxLayout()
        self.monitor_title = QLabel(UI_STRINGS[self.current_lang]["monitor_title"])
        title = self.monitor_title
        title.setObjectName("SectionTitleMonitor")
        monitor_row.addWidget(title)
        monitor_row.addStretch()

        self.status_badge = QLabel(UI_STRINGS[self.current_lang]["idle"])
        self.status_badge.setObjectName("TagBadge")
        monitor_row.addWidget(self.status_badge)
        
        self.toggle_monitor_btn = QPushButton()
        self.toggle_monitor_btn.setText(UI_STRINGS[self.current_lang]["hide"])
        self.toggle_monitor_btn.setFixedSize(50, 24)
        self.toggle_monitor_btn.setStyleSheet("font-size: 10px; padding: 2px 6px; border-radius: 4px;")
        self.toggle_monitor_btn.clicked.connect(self.toggle_monitor)
        monitor_row.addWidget(self.toggle_monitor_btn)
        
        layout.addLayout(monitor_row)

        # Log area
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setObjectName("LogArea")
        layout.addWidget(self.log_area, stretch=1)

        # Progress row
        progress_row = QHBoxLayout()
        progress_row.setSpacing(10)

        self.progress_label = QLabel(UI_STRINGS[self.current_lang]["ready"])
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

    def toggle_monitor(self):
        # Hide/show the monitor panel, and change splitter orientation
        if self.split.orientation() == Qt.Horizontal:
            # Hide monitor: switch to vertical, only show progress bar under config
            self.split.setOrientation(Qt.Vertical)
            self.monitor_panel.setMaximumHeight(120)
            self.log_area.hide()
            self.progress_label.show()
            self.progress_bar.show()
            self.toggle_monitor_btn.setText(UI_STRINGS[self.current_lang]["show"])
        else:
            # Show monitor: restore horizontal, show all
            self.split.setOrientation(Qt.Horizontal)
            self.monitor_panel.setMaximumHeight(16777215)
            self.log_area.show()
            self.progress_label.show()
            self.progress_bar.show()
            self.toggle_monitor_btn.setText(UI_STRINGS[self.current_lang]["hide"])

    # ── Helpers ──────────────────────────────────────────────────────────────

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not hasattr(self, 'config_panel') or not hasattr(self, 'monitor_panel') or not hasattr(self, 'split'):
            return
        if self.width() < 800:
            if self.split.orientation() != Qt.Vertical:
                self.split.setOrientation(Qt.Vertical)
                # Config on top (index 0), monitor on bottom (index 1)
                self.split.insertWidget(0, self.config_panel)
                self.split.insertWidget(1, self.monitor_panel)
                
            self.split.setMinimumHeight(self.monitor_panel.minimumHeight() + self.config_panel.minimumHeight() + 10)
        else:
            if self.split.orientation() != Qt.Horizontal:
                self.split.setOrientation(Qt.Horizontal)
                self.split.insertWidget(0, self.config_panel)
                self.split.insertWidget(1, self.monitor_panel)
                
            self.split.setMinimumHeight(max(self.monitor_panel.minimumHeight(), self.config_panel.minimumHeight()))
            
        # Ensure scroll area content resizes correctly to prevent layout auditor bounds error
        if hasattr(self, 'scroll_area') and self.scroll_area.widget():
            self.scroll_area.widget().setMinimumHeight(self.split.minimumHeight() + 150)

    def _setup_accessibility(self):
        # --- Tab Order ---
        QWidget.setTabOrder(self.theme_combo, self.input_edit)
        QWidget.setTabOrder(self.input_edit, self.browse_file_btn)
        QWidget.setTabOrder(self.browse_file_btn, self.lang_edit)
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
        self.shortcut_browse.activated.connect(self.browse_input_file)

        self.shortcut_start = QShortcut(QKeySequence("Ctrl+Return"), self)
        self.shortcut_start.activated.connect(self.start_processing)

        self.shortcut_start_enter = QShortcut(QKeySequence("Ctrl+Enter"), self)
        self.shortcut_start_enter.activated.connect(self.start_processing)

        self.shortcut_clear = QShortcut(QKeySequence("Esc"), self)
        self.shortcut_clear.activated.connect(lambda: self.input_edit.clear())

        self.change_language(self.current_lang)


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


    def change_language(self, lang_code: str):
        self.current_lang = lang_code.lower()
        s = UI_STRINGS[self.current_lang]

        self.theme_label.setText(s["theme"])
        self.config_title.setText(s["config_title"])
        self.input_label.setText(s["input_label"])
        self.input_edit.setPlaceholderText(s["input_placeholder"])
        self.browse_file_btn.setText(s["file_btn"])
        self.browse_folder_btn.setText(s["folder_btn"])
        self.engine_label.setText(s["engine_label"])
        self.target_lang_label.setText(s["target_lang_label"])
        self.lang_edit.setPlaceholderText(s["target_lang_placeholder"])
        self.api_label.setText(s["api_label"])
        self.mode_label.setText(s["mode_label"])
        self.src_lang_label.setText(s["src_lang_label"])
        self.src_lang_edit.setPlaceholderText(s["src_lang_placeholder"])
        self.max_dur_label.setText(s["max_dur_label"])
        self.max_duration.setSpecialValueText(s["max_dur_special"])
        self.delay_label.setText(s["delay_label"])
        self.vocal_check.setText(s["isolate_check"])
        self.skip_orig_check.setText(s["skip_orig_check"])
        self.overwrite_check.setText(s["overwrite_check"])
        self.continue_err_check.setText(s["continue_check"])
        self.start_btn.setText(s["start_btn"])
        self.monitor_title.setText(s["monitor_title"])
        self.progress_label.setText(s["ready"])
        self.footer.setText(s["footer"])

        if self.status_badge.text() in ["IDLE", "INACTIF"]:
            self.status_badge.setText(s["idle"])

        if self.toggle_monitor_btn.text() in ["Hide", "Cacher"]:
            self.toggle_monitor_btn.setText(s["hide"])
        elif self.toggle_monitor_btn.text() in ["Show", "Montrer"]:
            self.toggle_monitor_btn.setText(s["show"])

        # Tooltips
        self.theme_combo.setToolTip(s["tt_theme"])
        self.input_edit.setToolTip(s["tt_input"])
        self.browse_file_btn.setToolTip(s["tt_file"])
        self.browse_folder_btn.setToolTip(s["tt_folder"])
        self.lang_edit.setToolTip(s["tt_lang"])
        self.engine_combo.setToolTip(s["tt_engine"])
        self.api_combo.setToolTip(s["tt_api"])
        self.mode_combo.setToolTip(s["tt_mode"])
        self.src_lang_edit.setToolTip(s["tt_src"])
        self.max_duration.setToolTip(s["tt_dur"])
        self.vocal_check.setToolTip(s["tt_iso"])
        self.skip_orig_check.setToolTip(s["tt_skip"])
        self.overwrite_check.setToolTip(s["tt_over"])
        self.continue_err_check.setToolTip(s["tt_cont"])
        self.start_btn.setToolTip(s["tt_start"])

    # ── Theme ─────────────────────────────────────────────────────────────────



    def toggle_mode(self):
        curr_name = getattr(self.current_theme, "name", "")
        for i, t in enumerate(self.themes):
            if "Golden" in curr_name and ("Light" in t.name or "Solar" in t.name):
                self.theme_combo.setCurrentIndex(i)
                break
            elif ("Light" in curr_name or "Solar" in curr_name) and ("Golden" in t.name or "Slate" in t.name):
                self.theme_combo.setCurrentIndex(i)
                break

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

    def log(self, message: str, is_progress: bool = False):
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
        full_html = f"{ts_html} {msg_html}"

        if is_progress:
            if getattr(self, '_last_was_progress', False):
                # Overwrite the last line for progress updates
                cursor = self.log_area.textCursor()
                cursor.movePosition(QTextCursor.End)
                cursor.select(QTextCursor.BlockUnderCursor)
                cursor.removeSelectedText()
                self.log_area.setTextCursor(cursor)
                self.log_area.insertHtml(full_html)
            else:
                self.log_area.append(full_html)
            self._last_was_progress = True
        else:
            self.log_area.append(full_html)
            self._last_was_progress = False

        # Ensure scroll stays at bottom
        cursor = self.log_area.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_area.setTextCursor(cursor)

    def _on_task_mode_changed(self, text):
        is_stylize = "Stylization" in text
        self.stylize_widget.setVisible(is_stylize)
        self.transcription_widget.setVisible(not is_stylize)

    def start_processing(self):
        input_path = self.input_edit.text().strip()
        if not input_path:
            self.log(f"<span style='color:{self.current_theme.error}'>Error: No input path specified.</span>")
            return

        duration_val = self.max_duration.value()
        max_dur = duration_val * 60 if duration_val > 0 else None

        params = {
            "task_mode": self.task_mode_combo.currentText(),
            "prompt": self.prompt_edit.text(),
            "neg_prompt": self.neg_prompt_edit.text(),
            "input_file": input_path,
            "languages": [l.strip() for l in self.lang_edit.text().split(",") if l.strip()],
            "engine": self.engine_combo.currentText(),
            "translate_api": self.api_combo.currentText(),
            "translate_mode": self.mode_combo.currentText(),
            "skip_vocal_isolation": not self.vocal_check.isChecked(),
            "skip_original": self.skip_orig_check.isChecked(),
            "max_duration": max_dur,
            "delay_ms": self.delay_spin.value(),
            "source_language": self.src_lang_edit.text().strip() or None,
            "overwrite": self.overwrite_check.isChecked(),
            "continue_on_error": self.continue_err_check.isChecked(),
        }

        self.start_btn.setEnabled(False)
        self.status_badge.setText(UI_STRINGS[self.current_lang]["running"])
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
        self.log(text, is_progress=True)

    def _on_progress_pct(self, pct: int):
        self.progress_bar.setValue(pct)

    def on_error(self, message: str):
        self.log(f"<span style='color:{self.current_theme.error}'>ERROR: {message}</span>")
        self.status_badge.setText(UI_STRINGS[self.current_lang]["failed"])
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

        self.status_badge.setText(UI_STRINGS[self.current_lang]["done"])
        self.status_badge.setStyleSheet(
            f"background-color: {t.accent}; "
            f"color: {t.text_inverse}; "

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
