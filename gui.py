from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from PySide6.QtCore import Qt, Signal, Slot
from chat_widget import ChatWidget

class AvAGui(QMainWindow):
    """
    Main window: Chat interface is central. Buttons to open Code Viewer and Terminal windows.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AvA – AI Development Assistant")
        self.resize(900, 600)
        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget()
        layout = QVBoxLayout()

        # Chat widget
        self.chat_widget = ChatWidget()
        self.chat_widget.message_sent.connect(self._handle_user_message)

        # Buttons to launch other components
        button_layout = QHBoxLayout()
        self.open_code_viewer_btn = QPushButton("Open Code Viewer")
        self.open_code_viewer_btn.clicked.connect(self._open_code_viewer)
        self.open_terminal_btn = QPushButton("Open Terminal")
        self.open_terminal_btn.clicked.connect(self._open_terminal)
        button_layout.addWidget(self.open_code_viewer_btn)
        button_layout.addWidget(self.open_terminal_btn)

        layout.addWidget(self.chat_widget)
        layout.addLayout(button_layout)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Keep references to secondary windows
        self.code_viewer_window = None
        self.terminal_window = None

    @Slot(str)
    def _handle_user_message(self, message: str):
        # TODO: send 'message' to Planner LLM and display response in chat
        response = f"Planner: Received '{message}'"
        self.chat_widget.chat_list.addItem(response)

    @Slot()
    def _open_code_viewer(self):
        try:
            from code_viewer import CodeViewerWindow
            if self.code_viewer_window is None:
                self.code_viewer_window = CodeViewerWindow()
            self.code_viewer_window.show()
            self.code_viewer_window.raise_()
        except ImportError as e:
            self.chat_widget.chat_list.addItem(f"Error opening Code Viewer: {e}")

    @Slot()
    def _open_terminal(self):
        try:
            from terminals import TerminalWindow
            if self.terminal_window is None:
                self.terminal_window = TerminalWindow()
            self.terminal_window.show()
            self.terminal_window.raise_()
        except ImportError as e:
            self.chat_widget.chat_list.addItem(f"Error opening Terminal: {e}")

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = AvAGui()
    window.show()
    sys.exit(app.exec())
