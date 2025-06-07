# launcher.py

import sys
import os
import json
import requests
import zipfile
import subprocess
from pathlib import Path
from packaging.version import parse as parse_version

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QProgressBar
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QFont, QIcon  # Assuming you might want an icon later

# --- Configuration ---
# This is the URL to the raw version.json file on your GitHub repo.
# IMPORTANT: Make sure it's the "raw" link!
VERSION_URL = "https://raw.githubusercontent.com/your-username/your-repo/main/version.json"
APP_DIR = Path("./app")
CURRENT_VERSION_FILE = Path("./current_version.txt")
MAIN_APP_SCRIPT = APP_DIR / "main.py"


class UpdateWorker(QThread):
    """
    Runs the update check and download in a separate thread to keep the UI responsive.
    """
    status_updated = Signal(str)
    progress_updated = Signal(int)
    update_finished = Signal(bool, str)  # success (bool), message (str)

    def run(self):
        try:
            # 1. Get current version
            self.status_updated.emit("Checking local version...")
            if CURRENT_VERSION_FILE.exists():
                current_v_str = CURRENT_VERSION_FILE.read_text().strip()
            else:
                current_v_str = "0.0.0"
            current_version = parse_version(current_v_str)
            self.status_updated.emit(f"Current version: {current_version}")

            # 2. Fetch latest version info from server
            self.status_updated.emit("Contacting update server...")
            response = requests.get(VERSION_URL, timeout=10)
            response.raise_for_status()  # Raises an exception for bad status codes
            latest_info = response.json()
            latest_v_str = latest_info["version"]
            latest_version = parse_version(latest_v_str)
            self.status_updated.emit(f"Latest version: {latest_version}")

            # 3. Compare versions
            if latest_version > current_version:
                self.status_updated.emit(f"New version available! Downloading {latest_version}...")
                download_url = latest_info["download_url"]

                # 4. Download the update
                zip_path = Path("./update.zip")
                with requests.get(download_url, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get('content-length', 0))
                    bytes_downloaded = 0
                    with open(zip_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                            bytes_downloaded += len(chunk)
                            if total_size > 0:
                                progress = int((bytes_downloaded / total_size) * 100)
                                self.progress_updated.emit(progress)

                self.status_updated.emit("Download complete. Extracting files...")
                self.progress_updated.emit(0)  # Reset progress for extraction

                # 5. Unzip the contents
                APP_DIR.mkdir(exist_ok=True)
                with zipfile.ZipFile(zip_path) as zf:
                    # This is tricky because GitHub zips include a root folder.
                    # We need to extract files from that root folder into our ./app/ dir.
                    file_list = zf.infolist()
                    # The first item is usually the root directory name, e.g., "AvA-Release-1.0.1/"
                    root_dir_name = file_list[0].filename
                    for member in file_list:
                        # Strip the root directory from the path
                        member.filename = member.filename.replace(root_dir_name, "", 1)
                        if member.filename:  # Don't extract the root dir itself
                            zf.extract(member, APP_DIR)

                # 6. Clean up and update version file
                zip_path.unlink()  # Delete the zip file
                CURRENT_VERSION_FILE.write_text(latest_v_str)
                self.status_updated.emit(f"Update to version {latest_v_str} complete!")
                self.update_finished.emit(True, "Update successful!")

            else:
                self.status_updated.emit("You are up to date!")
                self.update_finished.emit(True, "No update needed.")

        except Exception as e:
            self.status_updated.emit(f"Error during update: {e}")
            self.update_finished.emit(False, f"An error occurred: {e}")


class LauncherWindow(QWidget):
    def __init__(self):
        super().__init__()
        self._setup_ui()
        self.worker = UpdateWorker()
        self._connect_signals()

        # Start the update process shortly after the window appears
        QTimer.singleShot(500, self.worker.start)

    def _setup_ui(self):
        self.setWindowTitle("AvA Launcher")
        self.setFixedSize(400, 150)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        # Center the window
        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width() - self.width()) // 2, (screen.height() - self.height()) // 2)

        self.setStyleSheet("""
            background-color: #161b22;
            color: #c9d1d9;
            border-radius: 12px;
            border: 1px solid #30363d;
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title Label
        title_font = QFont("Segoe UI", 14, QFont.Weight.Bold)
        self.title_label = QLabel("AvA Launcher")
        self.title_label.setFont(title_font)
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)

        # Status Label
        status_font = QFont("Segoe UI", 10)
        self.status_label = QLabel("Initializing...")
        self.status_label.setFont(status_font)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #0d1117;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #58a6ff;
                border-radius: 4px;
            }
        """)
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

    def _connect_signals(self):
        self.worker.status_updated.connect(self.status_label.setText)
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.update_finished.connect(self.on_update_finished)

    def on_update_finished(self, success, message):
        if success:
            self.status_label.setText("Launching AvA...")
            self.launch_app()
        else:
            self.status_label.setText(f"Failed to start: {message}")
            # In a real app, you might show a close button here or auto-close after a delay
            QTimer.singleShot(5000, self.close)  # Close after 5 seconds on error

    def launch_app(self):
        if not MAIN_APP_SCRIPT.exists():
            self.status_label.setText(f"Error: Main app script not found at {MAIN_APP_SCRIPT}")
            QTimer.singleShot(5000, self.close)
            return

        try:
            # Use subprocess.Popen to launch the main application detached from the launcher
            # We need to find the python executable to ensure it runs correctly
            python_exe = sys.executable
            subprocess.Popen([python_exe, str(MAIN_APP_SCRIPT)], cwd=APP_DIR)

            # Close the launcher
            QTimer.singleShot(100, self.close)

        except Exception as e:
            self.status_label.setText(f"Error launching app: {e}")
            QTimer.singleShot(5000, self.close)


if __name__ == "__main__":
    # IMPORTANT: You must update the VERSION_URL at the top of this file!
    if "your-username" in VERSION_URL:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! CRITICAL: Please update the VERSION_URL in launcher.py !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = LauncherWindow()
    window.show()
    sys.exit(app.exec())