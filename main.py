# main.py - Clean entry point following SRP

import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.application import AvAApplication


def main():
    """Clean main entry point - ONLY creates app and runs it"""

    app = QApplication(sys.argv)
    app.setApplicationName("AvA")
    app.setApplicationDisplayName("AvA - AI Development Assistant")

    # Create and run AvA application
    ava_app = AvAApplication()
    ava_app.initialize()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())