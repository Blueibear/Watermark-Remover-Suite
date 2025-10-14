# GUI Architecture Overview

The Watermark Remover Suite provides a desktop interface built with PyQt5 (with a Tkinter fallback) to compliment the CLI.

## Objectives
- Offer intuitive access to image and video watermark removal workflows.
- Surface configuration options such as auto mask parameters without overwhelming casual users.
- Provide real-time progress feedback and preview of results.

## Window Layout
1. **Toolbar**: Quick actions for loading inputs, running processing, and opening output folders.
2. **Input Panel**: File selectors for image/video inputs and optional mask selection.
3. **Preview Panel**: Displays before/after thumbnails when processing images.
4. **Progress Panel**: Shows task status, log tail, and errors.

## Processing Flow
```text
User Input → Validation → Background Worker (QThread) → Core Processors → UI update signals
```

## Key Components
- `MainWindow` (`QMainWindow`): Hosts toolbar, panels, and status bar.
- `ProcessingController`: Bridges UI events to core processors, handles threading.
- `PreviewWidget`: Renders image previews with lazy loading.
- `LogConsole`: Captures log messages routed from the application's logging handlers.

## Threading Strategy
- CPU-intensive processing runs in a `QThreadPool` using `QRunnable` tasks.
- Signals report progress and status back to the UI.
- UI remains responsive by avoiding direct processing on the main thread.

## Fallback Behaviour
- When PyQt5 is unavailable, a simplified Tkinter interface launches with basic input and progress feedback.
- Shared controller logic ensures consistent behaviour across interfaces.

## Assets
- Icons stored under `assets/icons/`:
  - `app_icon.png`: application icon.
  - `process.png`: process action icon.
  - `open_folder.png`: open output folder.

## Next Steps
- Implement `ui/main_window.py` with PyQt5.
- Provide Tkinter fallback in `ui/fallback.py`.
- Integrate CLI commands for automation parity.
- Link UI help to `docs/user_guide.md` and `docs/developer_guide.md`.
