"""PyQt5 main window for the Watermark Remover Suite."""

from __future__ import annotations

import logging
from logging import FileHandler
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from config import DEFAULT_CONFIG_PATH, load_config
from core import (
    BatchWatermarkProcessor,
    ImageWatermarkRemover,
    VideoWatermarkRemover,
)
from core.logger import setup_logging

try:
    from PyQt5.QtCore import QObject, QRunnable, Qt, QThreadPool, pyqtSignal
    from PyQt5.QtGui import QCloseEvent, QPixmap
    from PyQt5.QtWidgets import (
        QApplication,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QProgressBar,
        QSizePolicy,
        QStatusBar,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    PYQT_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback handled elsewhere
    PYQT_AVAILABLE = False
    QApplication = object  # type: ignore
    QMainWindow = object  # type: ignore


logger = logging.getLogger(__name__)
GUI_VALIDATION_LOG = Path("verification_reports/gui_validation.log")


def _ensure_gui_log_handler() -> None:
    GUI_VALIDATION_LOG.parent.mkdir(parents=True, exist_ok=True)
    existing = [
        handler
        for handler in logger.handlers
        if isinstance(handler, FileHandler)
        and Path(getattr(handler, "baseFilename", "")).resolve() == GUI_VALIDATION_LOG.resolve()
    ]
    if not existing:
        file_handler = FileHandler(GUI_VALIDATION_LOG, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(file_handler)


class _WorkerSignals(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)


class _ProcessingWorker(QRunnable):
    def __init__(self, task: Callable[[], Any]) -> None:
        super().__init__()
        self.task = task
        self.signals = _WorkerSignals()

    def run(self) -> None:  # pragma: no cover - invoked in separate thread
        try:
            self.signals.progress.emit(0)
            result = self.task()
            self.signals.progress.emit(100)
            self.signals.finished.emit(result)
        except Exception as exc:  # pragma: no cover - error path
            logger.exception("Background task failed: %s", exc)
            self.signals.error.emit(str(exc))


class _LogEmitter(QObject):
    message = pyqtSignal(str)


class _QtLogHandler(logging.Handler):
    def __init__(self, emitter: _LogEmitter) -> None:
        super().__init__()
        self.emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - simple bridge
        msg = self.format(record)
        self.emitter.message.emit(msg)


class MainWindow(QMainWindow):  # pragma: no cover - exercised via integration tests
    def __init__(self, config_path: Optional[Path] = None) -> None:
        if not PYQT_AVAILABLE:
            logger.error("PyQt5 is not available. GUI cannot be launched.")
            raise RuntimeError("PyQt5 is not available. Cannot launch GUI.")

        super().__init__()
        self.setWindowTitle("Watermark Remover Suite")
        self.resize(900, 600)

        self.config = load_config(config_path or DEFAULT_CONFIG_PATH)
        setup_logging(self.config.get("logging", {}), force=True)
        _ensure_gui_log_handler()
        logger.info("MainWindow initialized with config from %s", config_path or DEFAULT_CONFIG_PATH)

        self.thread_pool = QThreadPool.globalInstance()
        self.image_remover = ImageWatermarkRemover.from_config(self.config)
        self.video_remover = VideoWatermarkRemover.from_config(
            self.config, image_remover=self.image_remover
        )
        self.batch_processor = BatchWatermarkProcessor(
            image_remover=self.image_remover,
            video_remover=self.video_remover,
            config=self.config,
        )

        self._log_emitter = _LogEmitter()
        self._log_handler = _QtLogHandler(self._log_emitter)
        self._log_handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
        logging.getLogger().addHandler(self._log_handler)

        self._create_widgets()
        self._set_status_help()
        self._connect_signals()

    # UI Construction -----------------------------------------------------------------
    def _create_widgets(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        layout.addWidget(self._build_input_group())
        layout.addWidget(self._build_preview_group())
        layout.addWidget(self._build_log_group())

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        status = QStatusBar(self)
        self.setStatusBar(status)
        status.showMessage("Ready")

    def _build_input_group(self) -> QGroupBox:
        group = QGroupBox("Inputs", self)
        grid = QGridLayout(group)

        # Image inputs
        self.image_input = QLineEdit(self)
        image_browse = QPushButton("Browse...", self)
        image_browse.clicked.connect(lambda: self._browse_file(self.image_input, "Select Image", image=True))

        self.image_output = QLineEdit(self)
        image_output_browse = QPushButton("Save As...", self)
        image_output_browse.clicked.connect(lambda: self._browse_save(self.image_output, "Select Output Image", image=True))

        self.image_mask = QLineEdit(self)
        image_mask_browse = QPushButton("Mask...", self)
        image_mask_browse.clicked.connect(lambda: self._browse_file(self.image_mask, "Select Mask Image", image=True))

        self.image_process_btn = QPushButton("Process Image", self)
        self.image_process_btn.clicked.connect(self._on_process_image)

        grid.addWidget(QLabel("Image Input:"), 0, 0)
        grid.addWidget(self.image_input, 0, 1)
        grid.addWidget(image_browse, 0, 2)
        grid.addWidget(QLabel("Image Output:"), 1, 0)
        grid.addWidget(self.image_output, 1, 1)
        grid.addWidget(image_output_browse, 1, 2)
        grid.addWidget(QLabel("Mask (optional):"), 2, 0)
        grid.addWidget(self.image_mask, 2, 1)
        grid.addWidget(image_mask_browse, 2, 2)
        grid.addWidget(self.image_process_btn, 3, 1, 1, 2)

        # Video inputs
        self.video_input = QLineEdit(self)
        video_browse = QPushButton("Browse...", self)
        video_browse.clicked.connect(lambda: self._browse_file(self.video_input, "Select Video", image=False))

        self.video_output = QLineEdit(self)
        video_output_browse = QPushButton("Save As...", self)
        video_output_browse.clicked.connect(lambda: self._browse_save(self.video_output, "Select Output Video", image=False))

        self.video_mask = QLineEdit(self)
        video_mask_browse = QPushButton("Mask...", self)
        video_mask_browse.clicked.connect(lambda: self._browse_file(self.video_mask, "Select Mask Image", image=True))

        self.video_process_btn = QPushButton("Process Video", self)
        self.video_process_btn.clicked.connect(self._on_process_video)

        grid.addWidget(QLabel("Video Input:"), 4, 0)
        grid.addWidget(self.video_input, 4, 1)
        grid.addWidget(video_browse, 4, 2)
        grid.addWidget(QLabel("Video Output:"), 5, 0)
        grid.addWidget(self.video_output, 5, 1)
        grid.addWidget(video_output_browse, 5, 2)
        grid.addWidget(QLabel("Mask (optional):"), 6, 0)
        grid.addWidget(self.video_mask, 6, 1)
        grid.addWidget(video_mask_browse, 6, 2)
        grid.addWidget(self.video_process_btn, 7, 1, 1, 2)

        return group

    def _build_preview_group(self) -> QGroupBox:
        group = QGroupBox("Preview", self)
        layout = QHBoxLayout(group)

        self.before_label = QLabel("Before", self)
        self.before_label.setAlignment(Qt.AlignCenter)
        self.before_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.after_label = QLabel("After", self)
        self.after_label.setAlignment(Qt.AlignCenter)
        self.after_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addWidget(self.before_label)
        layout.addWidget(self.after_label)
        return group

    def _build_log_group(self) -> QGroupBox:
        group = QGroupBox("Log", self)
        vbox = QVBoxLayout(group)
        self.log_console = QTextEdit(self)
        self.log_console.setReadOnly(True)
        vbox.addWidget(self.log_console)
        return group

    def _browse_file(self, line_edit: QLineEdit, title: str, *, image: bool) -> None:
        filters = "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)" if image else "Video Files (*.mp4 *.avi *.mov *.mkv)"
        path, _ = QFileDialog.getOpenFileName(self, title, "", filters)
        if path:
            line_edit.setText(path)
            if image and line_edit is self.image_input:
                self._load_preview(Path(path), self.before_label)

    def _browse_save(self, line_edit: QLineEdit, title: str, *, image: bool) -> None:
        filters = "PNG Image (*.png)" if image else "MP4 Video (*.mp4)"
        path, _ = QFileDialog.getSaveFileName(self, title, "", filters)
        if path:
            line_edit.setText(path)

    def _connect_signals(self) -> None:
        self._log_emitter.message.connect(self._append_log)

    def _append_log(self, message: str) -> None:
        self.log_console.append(message)

    def _set_status_help(self) -> None:
        self.statusBar().showMessage(
            "Need help? See docs/user_guide.md or press F1.", 5000
        )

    def _set_busy(self, busy: bool) -> None:
        self.image_process_btn.setDisabled(busy)
       self.video_process_btn.setDisabled(busy)
        if busy:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()

    # Processing -----------------------------------------------------------------------
    def _on_process_image(self) -> None:
        input_path = Path(self.image_input.text())
        if not input_path.exists():
            QMessageBox.warning(self, "Invalid Input", "Please select a valid image file.")
            return

        output_path = (
            Path(self.image_output.text())
            if self.image_output.text()
            else input_path.with_name(f"{input_path.stem}_restored{input_path.suffix}")
        )
        self.image_output.setText(str(output_path))

        mask = Path(self.image_mask.text()) if self.image_mask.text() else None

        def task() -> Tuple[Path, Path]:
            return self.image_remover.process_file(input_path, output_path, mask_path=mask)

        self._execute_task(task, self._handle_image_result)

    def _handle_image_result(self, result: Tuple[Path, Path]) -> None:
        output_path, mask_path = result
        self.statusBar().showMessage(f"Image processed: {output_path}", 5000)
        self._load_preview(output_path, self.after_label, is_before=False)
        if self.image_input.text():
            self._load_preview(Path(self.image_input.text()), self.before_label, is_before=True)
        logger.info("Mask saved to %s", mask_path)

    def _on_process_video(self) -> None:
        input_path = Path(self.video_input.text())
        if not input_path.exists():
            QMessageBox.warning(self, "Invalid Input", "Please select a valid video file.")
            return

        output_path = (
            Path(self.video_output.text())
            if self.video_output.text()
            else input_path.with_name(f"{input_path.stem}_restored.mp4")
        )
        self.video_output.setText(str(output_path))

        mask = Path(self.video_mask.text()) if self.video_mask.text() else None

        def task() -> Path:
            return self.video_remover.process_file(input_path, output_path, mask_path=mask)

        self._execute_task(task, self._handle_video_result)

    def _handle_video_result(self, output_path: Path) -> None:
        self.statusBar().showMessage(f"Video processed: {output_path}", 5000)
        logger.info("Video saved to %s", output_path)

    def _execute_task(self, task: Callable[[], Any], on_success: Callable[[Any], None]) -> None:
        worker = _ProcessingWorker(task)
        worker.signals.progress.connect(self.progress_bar.setValue)
        worker.signals.finished.connect(lambda result: self._on_task_finished(on_success, result))
        worker.signals.error.connect(self._on_task_error)
        self._set_busy(True)
        self.thread_pool.start(worker)

    def _on_task_finished(self, callback: Callable[[Any], None], result: Any) -> None:
        self._set_busy(False)
        self.progress_bar.setValue(100)
        callback(result)

    def _on_task_error(self, message: str) -> None:
        self._set_busy(False)
        self.progress_bar.setValue(0)
        QMessageBox.critical(self, "Processing Failed", message)

    def _load_preview(self, path: Path, label: QLabel, *, is_before: bool = True) -> None:
        if not path.exists():
            return
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return
        scaled = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)
        label.setText("" if not pixmap.isNull() else ("Before" if is_before else "After"))
        label.setProperty("description", "Before" if is_before else "After")

    # Lifecycle -----------------------------------------------------------------------
    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self.before_label.pixmap():
            self.before_label.setPixmap(
                self.before_label.pixmap().scaled(self.before_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        if self.after_label.pixmap():
            self.after_label.setPixmap(
                self.after_label.pixmap().scaled(self.after_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        logging.getLogger().removeHandler(self._log_handler)
        self._log_handler.close()
        QApplication.restoreOverrideCursor()
        logger.info("MainWindow closed cleanly.")
        super().closeEvent(event)


def run_gui(config_path: Optional[Path] = None, *, show: bool = True, log_validation: bool = False) -> int:
    if not PYQT_AVAILABLE:
        raise RuntimeError("PyQt5 is required to run the GUI.")

    import sys

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(config_path=config_path)
    if show:
        window.show()
        logger.info("GUI event loop starting.")
        try:
            return app.exec()
        except AttributeError:  # PyQt5 < 5.15 compatibility
            return app.exec_()

    if log_validation:
        GUI_VALIDATION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(GUI_VALIDATION_LOG, "a", encoding="utf-8") as log_file:
            log_file.write("GUI validation executed successfully.\n")
    logger.info("Headless GUI validation completed successfully.")
    return 0


# === CLI Entrypoint ===============================================================
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Watermark Remover Suite GUI Launcher")
    parser.add_argument("--show", action="store_true",
                        help="Launch the full GUI instead of running headless validation")
    args = parser.parse_args()

    if args.show:
        sys.exit(run_gui(show=True))

    sys.exit(run_gui(show=False, log_validation=True))

