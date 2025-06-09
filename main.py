# main.py
import sys
import asyncio
import logging
from pathlib import Path
from PySide6.QtWidgets import QApplication
import qasync

from core.application import AvAApplication

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

  # Import the AvAApplication class

ava_app_instance_container = []  # Global to hold the instance for shutdown
app_quit_future = None  # Global future to signal application quit


def handle_about_to_quit():
    """Slot to be called when QApplication is about to quit."""
    global app_quit_future
    logging.info("main.py: QApplication.aboutToQuit signal received.")
    if app_quit_future and not app_quit_future.done():
        logging.info("main.py: Setting result for app_quit_future.")
        app_quit_future.set_result(True)
    else:
        logging.info("main.py: app_quit_future is None or already done.")


async def main_async_logic():
    """The main async part of the application startup."""
    logging.info("main_async_logic: Coroutine started.")
    global ava_app_instance_container, app_quit_future

    app = QApplication.instance()
    if not app:
        logging.error("main_async_logic: QApplication instance not found. This is unexpected.")
        return

    app_quit_future = asyncio.Future()
    app.aboutToQuit.connect(handle_about_to_quit)
    logging.info("main_async_logic: Connected aboutToQuit signal.")

    # THE FIX IS HERE
    ava_app = AvAApplication()
    # THE FIX IS HERE

    ava_app_instance_container.append(ava_app)

    def on_fully_initialized():
        print("AvAApplication fully_initialized_signal received: Async components should be complete.")
        if ava_app_instance_container:
            current_ava_app = ava_app_instance_container[0]
            try:
                status = current_ava_app.get_status()
                print(f"Status Check (on_fully_initialized) - LLM Models: {status.get('llm_models', 'N/A')}")
                rag_status_info = status.get('rag', {})
                print(f"Status Check (on_fully_initialized) - RAG Info: {rag_status_info}")
            except Exception as e:
                # Use the logger for exceptions, which is safer
                logging.error("Error getting status in on_fully_initialized", exc_info=True)

    ava_app.fully_initialized_signal.connect(on_fully_initialized)

    try:
        logging.info("main_async_logic: About to call and await ava_app.initialize().")
        await ava_app.initialize()
        logging.info("main_async_logic: ava_app.initialize() completed.")

        logging.info("main_async_logic: Application initialized. Waiting for quit signal...")
        await app_quit_future
        logging.info("main_async_logic: app_quit_future completed. Application is quitting.")

    except Exception as e:
        logging.critical(f"main_async_logic: Error during application lifecycle: {e}", exc_info=True)
        if app_quit_future and not app_quit_future.done():
            app_quit_future.set_exception(e)
        if QApplication.instance():
            QApplication.instance().quit()
    finally:
        logging.info("main_async_logic: Exiting.")


if __name__ == "__main__":
    # It's important to have logging configured before the app starts for startup issues.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    app = QApplication(sys.argv)
    app.setApplicationName("AvA")
    app.setApplicationVersion("2.1")

    exit_code = 0
    try:
        logging.info("main.py: About to call qasync.run(main_async_logic()).")
        qasync.run(main_async_logic())
        logging.info("main.py: qasync.run() has completed.")

    except Exception as e:
        logging.critical(f"main.py: Unhandled top-level exception: {e}", exc_info=True)
        exit_code = 1
    finally:
        logging.info("main.py: Main finally block.")
        if ava_app_instance_container:
            ava_app_instance = ava_app_instance_container[0]
            if hasattr(ava_app_instance, 'shutdown'):
                logging.info("main.py: Calling AvAApplication.shutdown().")
                ava_app_instance.shutdown()

        logging.info(f"main.py: Script is ending. Intended exit code: {exit_code}")