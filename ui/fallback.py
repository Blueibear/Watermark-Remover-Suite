"""Tkinter fallback UI for environments without PyQt5."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from watermark_remover.config import DEFAULT_CONFIG_PATH, load_config
from watermark_remover.core.logger import setup_logging

from watermark_remover.core import ImageWatermarkRemover, VideoWatermarkRemover

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    TK_AVAILABLE = True
except ImportError:  # pragma: no cover
    TK_AVAILABLE = False


logger = logging.getLogger(__name__)


class FallbackApp:
    """Simplified GUI leveraging Tkinter when PyQt5 is unavailable."""

    def __init__(self, root: Optional["tk.Tk"] = None, config_path: Optional[Path] = None) -> None:
        if not TK_AVAILABLE:
            raise RuntimeError("Tkinter is not available. Cannot launch fallback GUI.")

        self.root = root or tk.Tk()
        self.root.title("Watermark Remover Suite (Fallback)")
        self.root.geometry("600x420")

        self.config = load_config(config_path or DEFAULT_CONFIG_PATH)
        setup_logging(self.config.get("logging", {}), force=True)

        self.image_remover = ImageWatermarkRemover.from_config(self.config)
        self.video_remover = VideoWatermarkRemover.from_config(self.config, image_remover=self.image_remover)

        self._create_widgets()
        self._lock_ui(False)

    def _create_widgets(self) -> None:
        padding = {"padx": 10, "pady": 5}

        # Notebook for image/video tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, **padding)

        self.image_frame = ttk.Frame(notebook)
        self.video_frame = ttk.Frame(notebook)

        notebook.add(self.image_frame, text="Image")
        notebook.add(self.video_frame, text="Video")

        self._build_image_tab()
        self._build_video_tab()

        # Log area
        log_frame = ttk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill="both", expand=True, **padding)
        self.log_text = tk.Text(log_frame, height=8, state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

    def _build_image_tab(self) -> None:
        padding = {"padx": 8, "pady": 4}

        ttk.Label(self.image_frame, text="Input Image").grid(row=0, column=0, sticky="w", **padding)
        self.image_input_var = tk.StringVar()
        ttk.Entry(self.image_frame, textvariable=self.image_input_var, width=45).grid(row=0, column=1, **padding)
        ttk.Button(self.image_frame, text="Browse", command=self._browse_image_input).grid(row=0, column=2, **padding)

        ttk.Label(self.image_frame, text="Output Image").grid(row=1, column=0, sticky="w", **padding)
        self.image_output_var = tk.StringVar()
        ttk.Entry(self.image_frame, textvariable=self.image_output_var, width=45).grid(row=1, column=1, **padding)
        ttk.Button(self.image_frame, text="Save As", command=self._browse_image_output).grid(row=1, column=2, **padding)

        ttk.Label(self.image_frame, text="Mask (optional)").grid(row=2, column=0, sticky="w", **padding)
        self.image_mask_var = tk.StringVar()
        ttk.Entry(self.image_frame, textvariable=self.image_mask_var, width=45).grid(row=2, column=1, **padding)
        ttk.Button(self.image_frame, text="Browse", command=self._browse_image_mask).grid(row=2, column=2, **padding)

        self.image_progress = ttk.Progressbar(self.image_frame, mode="determinate", maximum=100)
        self.image_progress.grid(row=3, column=0, columnspan=3, sticky="ew", **padding)

        ttk.Button(self.image_frame, text="Process Image", command=self._on_process_image).grid(
            row=4, column=0, columnspan=3, pady=10
        )

    def _build_video_tab(self) -> None:
        padding = {"padx": 8, "pady": 4}

        ttk.Label(self.video_frame, text="Input Video").grid(row=0, column=0, sticky="w", **padding)
        self.video_input_var = tk.StringVar()
        ttk.Entry(self.video_frame, textvariable=self.video_input_var, width=45).grid(row=0, column=1, **padding)
        ttk.Button(self.video_frame, text="Browse", command=self._browse_video_input).grid(row=0, column=2, **padding)

        ttk.Label(self.video_frame, text="Output Video").grid(row=1, column=0, sticky="w", **padding)
        self.video_output_var = tk.StringVar()
        ttk.Entry(self.video_frame, textvariable=self.video_output_var, width=45).grid(row=1, column=1, **padding)
        ttk.Button(self.video_frame, text="Save As", command=self._browse_video_output).grid(row=1, column=2, **padding)

        ttk.Label(self.video_frame, text="Mask (optional)").grid(row=2, column=0, sticky="w", **padding)
        self.video_mask_var = tk.StringVar()
        ttk.Entry(self.video_frame, textvariable=self.video_mask_var, width=45).grid(row=2, column=1, **padding)
        ttk.Button(self.video_frame, text="Browse", command=self._browse_video_mask).grid(row=2, column=2, **padding)

        self.video_progress = ttk.Progressbar(self.video_frame, mode="determinate", maximum=100)
        self.video_progress.grid(row=3, column=0, columnspan=3, sticky="ew", **padding)

        ttk.Button(self.video_frame, text="Process Video", command=self._on_process_video).grid(
            row=4, column=0, columnspan=3, pady=10
        )

    # Browsers ------------------------------------------------------------------------
    def _browse_image_input(self) -> None:
        path = filedialog.askopenfilename(title="Select Image", filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
        if path:
            self.image_input_var.set(path)

    def _browse_image_output(self) -> None:
        path = filedialog.asksaveasfilename(title="Save Image", defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            self.image_output_var.set(path)

    def _browse_image_mask(self) -> None:
        path = filedialog.askopenfilename(title="Select Mask", filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
        if path:
            self.image_mask_var.set(path)

    def _browse_video_input(self) -> None:
        path = filedialog.askopenfilename(title="Select Video", filetypes=[("Videos", "*.mp4;*.mov;*.avi;*.mkv")])
        if path:
            self.video_input_var.set(path)

    def _browse_video_output(self) -> None:
        path = filedialog.asksaveasfilename(title="Save Video", defaultextension=".mp4", filetypes=[("MP4", "*.mp4")])
        if path:
            self.video_output_var.set(path)

    def _browse_video_mask(self) -> None:
        path = filedialog.askopenfilename(title="Select Mask", filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
        if path:
            self.video_mask_var.set(path)

    # Logging -------------------------------------------------------------------------
    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{message}\n")
        self.log_text.configure(state="disabled")
        self.log_text.see("end")

    def _lock_ui(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        for widget in self.image_frame.winfo_children():
            if isinstance(widget, ttk.Button):
                widget.configure(state=state)
        for widget in self.video_frame.winfo_children():
            if isinstance(widget, ttk.Button):
                widget.configure(state=state)
        if not busy:
            self.image_progress["value"] = 0
            self.video_progress["value"] = 0

    # Processing ----------------------------------------------------------------------
    def _on_process_image(self) -> None:
        input_path = Path(self.image_input_var.get())
        if not input_path.exists():
            messagebox.showwarning("Invalid Input", "Please choose a valid image file.")
            return
        output_path = Path(self.image_output_var.get()) if self.image_output_var.get() else input_path.with_name(f"{input_path.stem}_restored{input_path.suffix}")
        self.image_output_var.set(str(output_path))
        mask_path = Path(self.image_mask_var.get()) if self.image_mask_var.get() else None

        def task() -> None:
            self._lock_ui(True)
            self.image_progress["value"] = 10
            output, mask = self.image_remover.process_file(input_path, output_path, mask_path=mask)
            self.image_progress["value"] = 100
            self._append_log(f"Image processed: {output}")
            if mask:
                self._append_log(f"Mask saved: {mask}")
            messagebox.showinfo("Done", f"Image processed successfully.\nSaved to {output}")
            self._lock_ui(False)

        threading.Thread(target=self._run_task_safe, args=(task,), daemon=True).start()

    def _on_process_video(self) -> None:
        input_path = Path(self.video_input_var.get())
        if not input_path.exists():
            messagebox.showwarning("Invalid Input", "Please choose a valid video file.")
            return
        output_path = Path(self.video_output_var.get()) if self.video_output_var.get() else input_path.with_name(f"{input_path.stem}_restored.mp4")
        self.video_output_var.set(str(output_path))
        mask_path = Path(self.video_mask_var.get()) if self.video_mask_var.get() else None

        def task() -> None:
            self._lock_ui(True)
            self.video_progress["value"] = 10
            output = self.video_remover.process_file(input_path, output_path, mask_path=mask_path)
            self.video_progress["value"] = 100
            self._append_log(f"Video processed: {output}")
            messagebox.showinfo("Done", f"Video processed successfully.\nSaved to {output}")
            self._lock_ui(False)

        threading.Thread(target=self._run_task_safe, args=(task,), daemon=True).start()

    def _run_task_safe(self, func) -> None:
        try:
            func()
        except Exception as exc:  # pragma: no cover - error path
            logger.exception("Processing failed: %s", exc)
            self._append_log(f"Error: {exc}")
            messagebox.showerror("Error", str(exc))
            self._lock_ui(False)

    def run(self) -> None:  # pragma: no cover - manual launch
        self.root.mainloop()


def run_fallback_gui(config_path: Optional[Path] = None) -> None:  # pragma: no cover
    if not TK_AVAILABLE:
        raise RuntimeError("Tkinter is required for the fallback GUI.")
    app = FallbackApp(config_path=config_path)
    app.run()
