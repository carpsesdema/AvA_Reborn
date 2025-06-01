import sys
import asyncio
from PySide6.QtWidgets import QApplication
from gui import AvAGui
from utils.logger import init_logging

"""
Entry point for AvA.  Initializes logging, RAG index (if needed), and starts the GUI.
"""

def main():
    init_logging()
    app = QApplication(sys.argv)
    window = AvAGui()
    window.show()
    # Start asyncio event loop alongside Qt
    loop = asyncio.get_event_loop()
    loop.run_until_complete(app.exec())

if __name__ == "__main__":
    main()