# main.py

import sys
import asyncio
import logging
from pathlib import Path
from PySide6.QtWidgets import QApplication
import qasync

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.application import AvAApplication
from gui.main_window import AvAMainWindow
from utils.logger import init_logging


# Global exception handler
def handle_exception(exc_type, exc_value, exc_traceback):
    """Log unhandled exceptions."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.critical("Unhandled top-level exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

# --- Asynchronous Main Logic ---
async def main_async_logic():
    """
    The main asynchronous entry point for the application.
    Initializes the event loop, creates windows, and starts the app logic.
    """
    logging.info("main_async_logic: Coroutine started.")
    app_quit_future = asyncio.get_event_loop().create_future()

    def on_about_to_quit():
        logging.info("main.py: QApplication.aboutToQuit signal received.")
        if not app_quit_future.done():
            app_quit_future.set_result(True)

    QApplication.instance().aboutToQuit.connect(on_about_to_quit)
    logging.info("main_async_logic: Connected aboutToQuit signal.")

    # Create the main GUI window.
    window = AvAMainWindow()

    # Create the main application logic class, passing it the main window.
    ava_app = AvAApplication(window)

    # Launch the application's asynchronous initialization as a background task.
    # This task will run on the qasync event loop without blocking UI setup.
    init_task = asyncio.create_task(ava_app.initialize_async())

    # Show the main window immediately. It will become fully functional
    # once the initialization task completes.
    window.show()
    logging.info("main_async_logic: MainWindow shown.")

    # Wait for the application to be closed.
    await app_quit_future
    logging.info("main_async_logic: app_quit_future completed.")

    # Cleanly handle the init_task on exit
    if not init_task.done():
        init_task.cancel()
        try:
            await init_task
        except asyncio.CancelledError:
            logging.info("Initialization task cancelled on exit.")

    logging.info("main_async_logic: Coroutine ending.")

# --- Main Entry Point ---
if __name__ == "__main__":
    exit_code = 0
    try:
        init_logging()
        app = QApplication(sys.argv)
        qasync.run(main_async_logic())
    except Exception as e:
        logging.critical(f"Unhandled top-level exception during startup: {e}", exc_info=True)
        exit_code = 1

    logging.info(f"main.py: Application exiting with code {exit_code}.")
    sys.exit(exit_code)