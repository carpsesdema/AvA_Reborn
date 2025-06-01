# gui/components.py - Professional UI Components

from PySide6.QtWidgets import QPushButton, QFrame, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class ModernButton(QPushButton):
    """Professional styled button with hover animations"""

    def __init__(self, text="", icon_path=None, button_type="primary"):
        super().__init__(text)
        self.button_type = button_type
        self.setMinimumHeight(36)
        self.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))

        if button_type == "primary":
            self.setStyleSheet("""
                ModernButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #0078d4, stop:1 #005a9e);
                    border: 1px solid #004578;
                    border-radius: 6px;
                    color: white;
                    padding: 8px 16px;
                    font-weight: 500;
                }
                ModernButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #106ebe, stop:1 #005a9e);
                    border-color: #0078d4;
                }
                ModernButton:pressed {
                    background: #004578;
                    border-color: #003d5b;
                }
                ModernButton:disabled {
                    background: #2d2d30;
                    border-color: #404040;
                    color: #6d6d6d;
                }
            """)
        elif button_type == "secondary":
            self.setStyleSheet("""
                ModernButton {
                    background: #2d2d30;
                    border: 1px solid #404040;
                    border-radius: 6px;
                    color: #cccccc;
                    padding: 8px 16px;
                }
                ModernButton:hover {
                    background: #3e3e42;
                    border-color: #0078d4;
                    color: white;
                }
                ModernButton:pressed {
                    background: #1e1e1e;
                }
            """)
        elif button_type == "accent":
            self.setStyleSheet("""
                ModernButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #00d7ff, stop:1 #0078d4);
                    border: 1px solid #0078d4;
                    border-radius: 6px;
                    color: #1e1e1e;
                    padding: 8px 16px;
                    font-weight: 600;
                }
                ModernButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #40e0ff, stop:1 #0078d4);
                }
            """)


class ModernPanel(QFrame):
    """Professional panel with subtle borders and shadows"""

    def __init__(self, title=""):
        super().__init__()
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("""
            ModernPanel {
                background: #252526;
                border: 1px solid #3e3e42;
                border-radius: 8px;
                margin: 2px;
            }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(8)

        if title:
            title_label = QLabel(title)
            title_label.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
            title_label.setStyleSheet("color: #0078d4; margin-bottom: 4px;")
            layout.addWidget(title_label)

        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(6)
        layout.addLayout(self.content_layout)

        self.setLayout(layout)

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)


class StatusIndicator(QLabel):
    """Professional status indicator with colored dots"""

    def __init__(self, status="ready"):
        super().__init__()
        self.setFixedSize(12, 12)
        self.status = status
        self.update_status(status)

    def update_status(self, status):
        self.status = status
        colors = {
            "ready": "#00d7ff",
            "working": "#ffb900",
            "success": "#107c10",
            "error": "#d13438",
            "offline": "#6d6d6d"
        }

        color = colors.get(status, "#6d6d6d")
        self.setStyleSheet(f"""
            StatusIndicator {{
                background: {color};
                border: 2px solid #1e1e1e;
                border-radius: 6px;
            }}
        """)