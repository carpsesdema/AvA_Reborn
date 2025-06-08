# main.py
import sys
import asyncio
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QCoreApplication  # For aboutToQuit
import qasync  # Make sure qasync is imported

from core.application import AvAApplication

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

  # Import the AvAApplication class

ava_app_instance_container = []  # Global to hold the instance for shutdown
app_quit_future = None  # Global future to signal application quit


def handle_about_to_quit():
    """Slot to be called when QApplication is about to quit."""
    global app_quit_future
    print("main.py: QApplication.aboutToQuit signal received.")
    if app_quit_future and not app_quit_future.done():
        print("main.py: Setting result for app_quit_future.")
        app_quit_future.set_result(True)
    else:
        print("main.py: app_quit_future is None or already done.")


async def main_async_logic():
    """The main async part of the application startup."""
    print("main_async_logic: Coroutine started.")
    global ava_app_instance_container, app_quit_future

    # Ensure QApplication instance exists before trying to connect to its signals
    app = QApplication.instance()
    if not app:
        print("main_async_logic: QApplication instance not found. This is unexpected.")
        return

    app_quit_future = asyncio.Future()
    app.aboutToQuit.connect(handle_about_to_quit)
    print("main_async_logic: Connected aboutToQuit signal.")

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
                print(f"Error getting status in on_fully_initialized: {e}", exc_info=True)

    ava_app.fully_initialized_signal.connect(on_fully_initialized)

    try:
        print("main_async_logic: About to call and await ava_app.initialize().")
        await ava_app.initialize()
        print("main_async_logic: ava_app.initialize() completed.")

        # Now, wait for the application to quit
        print("main_async_logic: Application initialized. Waiting for quit signal...")
        await app_quit_future
        print("main_async_logic: app_quit_future completed. Application is quitting.")

    except Exception as e:
        print(f"main_async_logic: Error during application lifecycle: {e}")
        import traceback
        traceback.print_exc()
        if app_quit_future and not app_quit_future.done():
            app_quit_future.set_exception(e)  # Signal error to the future if it's still pending
        if QApplication.instance():
            QApplication.instance().quit()  # Attempt to quit if an error occurs during init
        # Do not re-raise here if qasync.run is expected to handle process exit
    finally:
        print("main_async_logic: Exiting.")


if __name__ == "__main__":
    # QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling) # Optional: for High DPI displays
    app = QApplication(sys.argv)
    app.setApplicationName("AvA")
    app.setApplicationDisplayName("AvA - AI Development Assistant")
    app.setApplicationVersion("2.0")

    # It's crucial that the QApplication instance exists before qasync.run,
    # so that main_async_logic can connect to its signals.

    exit_code = 0
    try:
        print("main.py: About to call qasync.run(main_async_logic()).")
        qasync.run(main_async_logic())
        print("main.py: qasync.run() has completed.")
        # qasync.run() blocks until the asyncio loop it manages is stopped.
        # This happens when main_async_logic completes (due to app_quit_future).
        # The QApplication might still be running its own event processing briefly.
        # We usually let Qt handle the final exit code.
        # If app.exec_() was called inside qasync or if qasync manages exit, this might be redundant.

    except SystemExit as e:
        print(f"main.py: SystemExit caught with code {e.code}")
        exit_code = e.code if isinstance(e.code, int) else 1
    except RuntimeError as e:
        print(f"main.py: Critical RuntimeError: {e}")
        import traceback

        traceback.print_exc()
        exit_code = 1  # Indicate an error
    except Exception as e:
        print(f"main.py: Unhandled top-level exception: {e}")
        import traceback

        traceback.print_exc()
        exit_code = 1  # Indicate an error
    finally:
        print("main.py: Main finally block.")
        if ava_app_instance_container:
            ava_app_instance = ava_app_instance_container[0]
            if hasattr(ava_app_instance, 'shutdown'):
                print("main.py: Calling AvAApplication.shutdown().")
                ava_app_instance.shutdown()

        print(f"main.py: Script is ending. Intended exit code: {exit_code}")
        # sys.exit(exit_code) # Let the script end naturally or Qt manage the exit.
        # Explicit sys.exit here might interfere with Qt's cleanup
        # if qasync.run() doesn't already cause a process exit.