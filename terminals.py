from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit
from PySide6.QtCore import QTimer

class StreamingTerminal(QWidget):
    """
    Aider-like terminal widget to stream LLM logs and code as they're generated.
    """
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setStyleSheet("background-color: #000; color: #0f0; font-family: 'Courier New';")
        layout.addWidget(self.text_area)
        self.setLayout(layout)
        # Example: refresh every 100ms to check for new logs or LLM output
        self.timer = QTimer()
        self.timer.timeout.connect(self._refresh)
        self.timer.start(100)

    def append_text(self, text: str):
        self.text_area.append(text)

    def _refresh(self):
        # Poll a shared log buffer or queue for new lines
        pass  # to be implemented