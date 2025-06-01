# main.py - Simplified Working Version

import sys
import os
from pathlib import Path

# Add the current directory to Python path so imports work
sys.path.insert(0, str(Path(__file__).parent))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont

from gui.main_window import AvAMainWindow


def setup_fonts():
    """Setup application fonts"""
    app = QApplication.instance()

    # Try to set a nice font
    font = QFont("Segoe UI", 9)
    if not font.exactMatch():
        # Fallback fonts
        for font_name in ["Arial", "Helvetica", "sans-serif"]:
            font = QFont(font_name, 9)
            if font.exactMatch():
                break

    app.setFont(font)


def main():
    """Simple main function that just shows the UI"""

    # Create Qt application
    app = QApplication(sys.argv)

    # Setup application properties
    app.setApplicationName("AvA")
    app.setApplicationDisplayName("AvA - AI Development Assistant")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("AvA Development")

    # Setup fonts
    setup_fonts()

    # Create and show main window
    window = AvAMainWindow()

    # Connect the chat message to a simple handler for now
    window.workflow_requested.connect(handle_chat_message)

    window.show()

    # Start the application
    return app.exec()


def handle_chat_message(message):
    """Simple handler for chat messages"""
    print(f"User message: {message}")

    # Simulate some processing with a timer
    def simulate_response():
        print("Simulated response completed")

    QTimer.singleShot(2000, simulate_response)  # 2 second delay


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"Error starting AvA: {e}")
        sys.exit(1)