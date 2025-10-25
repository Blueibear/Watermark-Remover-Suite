import os
import unittest
from importlib import import_module

main_window_module = import_module("ui.main_window")
fallback_module = import_module("ui.fallback")


class TestPyQtMainWindow(unittest.TestCase):
    @unittest.skipUnless(main_window_module.PYQT_AVAILABLE, "PyQt5 not available")
    def test_main_window_initialises(self) -> None:
        from PyQt5.QtWidgets import QApplication

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QApplication.instance() or QApplication([])
        window = main_window_module.MainWindow()
        try:
            self.assertEqual(window.windowTitle(), "Watermark Remover Suite")
            self.assertTrue(window.image_process_btn.isEnabled())
        finally:
            window.close()
            if hasattr(app, "quit"):
                app.quit()


class TestTkFallback(unittest.TestCase):
    @unittest.skipUnless(fallback_module.TK_AVAILABLE, "Tkinter not available")
    @unittest.skipIf(os.environ.get("DISPLAY", "") == "", "No display available in CI")
    def test_fallback_app_initialises(self) -> None:
        root = fallback_module.tk.Tk()
        root.withdraw()
        app = fallback_module.FallbackApp(root=root)
        try:
            self.assertEqual(root.title(), "Watermark Remover Suite (Fallback)")
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
