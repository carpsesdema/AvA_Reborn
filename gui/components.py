# gui/components_fixed.py - Clean, working components

from PySide6.QtWidgets import QPushButton, QFrame, QVBoxLayout, QLabel, QComboBox
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class ModernButton(QPushButton):
    """Clean, professional styled button"""

    def __init__(self, text="", icon_path=None, button_type="primary"):
        super().__init__(text)
        self.button_type = button_type
        self.setMinimumHeight(32)
        self.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
        self._apply_style()

    def _apply_style(self):
        """Apply button styling based on type"""

        if self.button_type == "primary":
            self.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #0078d4, stop:1 #005a9e);
                    border: 1px solid #004578;
                    border-radius: 6px;
                    color: white;
                    padding: 6px 12px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #106ebe, stop:1 #005a9e);
                    border-color: #0078d4;
                }
                QPushButton:pressed {
                    background: #004578;
                    border-color: #003d5b;
                }
            """)

        elif self.button_type == "secondary":
            self.setStyleSheet("""
                QPushButton {
                    background: #2d2d30;
                    border: 1px solid #404040;
                    border-radius: 6px;
                    color: #cccccc;
                    padding: 6px 12px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background: #3e3e42;
                    border-color: #00d7ff;
                    color: white;
                }
                QPushButton:pressed {
                    background: #1e1e1e;
                }
            """)

        elif self.button_type == "accent":
            self.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #00d7ff, stop:1 #0078d4);
                    border: 1px solid #0078d4;
                    border-radius: 6px;
                    color: #1e1e1e;
                    padding: 6px 12px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #40e0ff, stop:1 #0078d4);
                }
                QPushButton:pressed {
                    background: #0078d4;
                }
            """)


class StatusIndicator(QLabel):
    """Clean status indicator with colored dots"""

    def __init__(self, status="ready"):
        super().__init__()
        self.setFixedSize(12, 12)
        self.status = status
        self.update_status(status)

    def update_status(self, status):
        """Update status with appropriate color"""
        self.status = status
        colors = {
            "ready": "#00d7ff",  # Blue - ready
            "working": "#ffb900",  # Yellow - processing
            "success": "#107c10",  # Green - success
            "error": "#d13438",  # Red - error
            "offline": "#6d6d6d"  # Gray - offline
        }

        color = colors.get(status, "#6d6d6d")
        self.setStyleSheet(f"""
            StatusIndicator {{
                background: {color};
                border: 2px solid #1e1e1e;
                border-radius: 6px;
            }}
        """)


class ModernPanel(QFrame):
    """Clean panel with proper borders"""

    def __init__(self, title=""):
        super().__init__()
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("""
            ModernPanel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2a2a2e, stop:1 #252526);
                border: 2px solid #00d7ff;
                border-radius: 8px;
                margin: 2px;
            }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(8)

        if title:
            title_label = QLabel(title)
            title_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
            title_label.setStyleSheet("""
                QLabel {
                    color: #00d7ff;
                    background: transparent;
                    border: none;
                    margin-bottom: 4px;
                }
            """)
            layout.addWidget(title_label)

        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(6)
        layout.addLayout(self.content_layout)

        self.setLayout(layout)

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)

    def add_layout(self, layout):
        self.content_layout.addLayout(layout)


class StyledComboBox(QComboBox):
    """Clean styled combo box"""

    def __init__(self, items=None):
        super().__init__()
        if items:
            self.addItems(items)
        self._apply_style()

    def _apply_style(self):
        """Apply clean combo box styling"""
        self.setStyleSheet("""
            QComboBox {
                background: #1e1e1e;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 4px 8px;
                color: #cccccc;
                min-width: 100px;
                font-size: 10px;
            }
            QComboBox:hover {
                border-color: #00d7ff;
            }
            QComboBox::drop-down {
                border: none;
                width: 16px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid #cccccc;
                margin-right: 4px;
            }
            QComboBox QAbstractItemView {
                background: #2d2d30;
                border: 1px solid #00d7ff;
                selection-background-color: #00d7ff;
                selection-color: #1e1e1e;
                color: #cccccc;
                outline: none;
            }
        """)