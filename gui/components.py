# gui/components.py - Modern Design System for AvA (Updated existing file)

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QPushButton, QFrame, QVBoxLayout, QLabel, QComboBox, QWidget


# ================================
# DESIGN TOKENS
# ================================

class Colors:
    """Modern color palette inspired by VS Code & GitHub"""

    # Backgrounds
    PRIMARY_BG = "#0d1117"  # Deep dark background
    SECONDARY_BG = "#161b22"  # Secondary panels
    ELEVATED_BG = "#21262d"  # Elevated elements
    HOVER_BG = "#30363d"  # Hover states

    # Accent Colors
    ACCENT_BLUE = "#58a6ff"  # Primary accent
    ACCENT_PURPLE = "#a5a6ff"  # Secondary accent
    ACCENT_GREEN = "#3fb950"  # Success states
    ACCENT_ORANGE = "#d18616"  # Warning states
    ACCENT_RED = "#f85149"  # Error states

    # Text Colors
    TEXT_PRIMARY = "#f0f6fc"  # Primary text
    TEXT_SECONDARY = "#8b949e"  # Secondary text
    TEXT_MUTED = "#6e7681"  # Muted text

    # Borders
    BORDER_DEFAULT = "#30363d"  # Default borders
    BORDER_MUTED = "#21262d"  # Subtle borders
    BORDER_ACCENT = "#58a6ff"  # Accent borders


class Typography:
    """Typography system with proper hierarchy"""

    @staticmethod
    def get_font(size=12, weight=QFont.Weight.Normal, family="Segoe UI"):
        font = QFont(family, size, weight)
        font.setStyleHint(QFont.StyleHint.SansSerif)
        return font

    @staticmethod
    def heading_large():
        return Typography.get_font(16, QFont.Weight.Bold)

    @staticmethod
    def heading_medium():
        return Typography.get_font(14, QFont.Weight.DemiBold)

    @staticmethod
    def heading_small():
        return Typography.get_font(12, QFont.Weight.DemiBold)

    @staticmethod
    def body():
        return Typography.get_font(11, QFont.Weight.Normal)

    @staticmethod
    def body_small():
        return Typography.get_font(10, QFont.Weight.Normal)

    @staticmethod
    def code():
        return Typography.get_font(11, QFont.Weight.Normal, "JetBrains Mono")


# ================================
# CORE COMPONENTS
# ================================

class ModernButton(QPushButton):
    """Professional button with multiple variants and smooth animations"""

    def __init__(self, text="", button_type="primary", icon=None):
        super().__init__(text)
        self.button_type = button_type
        self.setMinimumHeight(36)
        self.setFont(Typography.body())
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Setup animations
        self._setup_animations()
        self._apply_style()

    def _setup_animations(self):
        """Setup smooth hover animations"""
        self.setProperty("hover", False)

    def _apply_style(self):
        """Apply modern styling based on button type"""

        base_style = f"""
            QPushButton {{
                border: 1px solid;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: 500;
                text-align: center;
                transition: all 0.2s ease;
            }}
            QPushButton:hover {{
                transform: translateY(-1px);
            }}
            QPushButton:pressed {{
                transform: translateY(0px);
            }}
        """

        if self.button_type == "primary":
            style = base_style + f"""
                QPushButton {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 {Colors.ACCENT_BLUE}, stop:1 #1f6feb);
                    border-color: {Colors.ACCENT_BLUE};
                    color: {Colors.TEXT_PRIMARY};
                }}
                QPushButton:hover {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #6cb6ff, stop:1 {Colors.ACCENT_BLUE});
                    border-color: #6cb6ff;
                    box-shadow: 0 4px 12px rgba(88, 166, 255, 0.3);
                }}
                QPushButton:pressed {{
                    background: #1f6feb;
                    border-color: #1f6feb;
                }}
                QPushButton:disabled {{
                    background: {Colors.BORDER_DEFAULT};
                    border-color: {Colors.BORDER_MUTED};
                    color: {Colors.TEXT_MUTED};
                }}
            """

        elif self.button_type == "secondary":
            style = base_style + f"""
                QPushButton {{
                    background: {Colors.SECONDARY_BG};
                    border-color: {Colors.BORDER_DEFAULT};
                    color: {Colors.TEXT_PRIMARY};
                }}
                QPushButton:hover {{
                    background: {Colors.HOVER_BG};
                    border-color: {Colors.BORDER_ACCENT};
                    color: {Colors.ACCENT_BLUE};
                }}
                QPushButton:pressed {{
                    background: {Colors.BORDER_DEFAULT};
                }}
            """

        elif self.button_type == "ghost":
            style = base_style + f"""
                QPushButton {{
                    background: transparent;
                    border-color: transparent;
                    color: {Colors.TEXT_SECONDARY};
                }}
                QPushButton:hover {{
                    background: {Colors.HOVER_BG};
                    border-color: {Colors.BORDER_DEFAULT};
                    color: {Colors.TEXT_PRIMARY};
                }}
            """

        elif self.button_type == "danger":
            style = base_style + f"""
                QPushButton {{
                    background: {Colors.ACCENT_RED};
                    border-color: {Colors.ACCENT_RED};
                    color: {Colors.TEXT_PRIMARY};
                }}
                QPushButton:hover {{
                    background: #ff6b6b;
                    border-color: #ff6b6b;
                }}
            """

        self.setStyleSheet(style)


class StatusIndicator(QWidget):
    """Modern status indicator with smooth color transitions"""

    def __init__(self, status="offline"):
        super().__init__()
        self.setFixedSize(12, 12)
        self._status = status
        self.update_status(status)

    def update_status(self, status: str):
        """Update the status with smooth color transition"""
        self._status = status

        colors = {
            "ready": Colors.ACCENT_BLUE,
            "working": Colors.ACCENT_ORANGE,
            "success": Colors.ACCENT_GREEN,
            "error": Colors.ACCENT_RED,
            "offline": Colors.TEXT_MUTED
        }

        color = colors.get(status, Colors.TEXT_MUTED)

        self.setStyleSheet(f"""
            StatusIndicator {{
                background: {color};
                border: 2px solid {Colors.PRIMARY_BG};
                border-radius: 6px;
            }}
        """)

        # Add subtle glow effect for active states
        if status in ["working", "ready"]:
            self.setStyleSheet(f"""
                StatusIndicator {{
                    background: {color};
                    border: 2px solid {Colors.PRIMARY_BG};
                    border-radius: 6px;
                }}
            """)


class ModernPanel(QFrame):
    """Sleek panel with subtle borders and shadows"""

    def __init__(self, title="", collapsible=False):
        super().__init__()
        self.title = title
        self.collapsible = collapsible
        self.is_collapsed = False

        self.setFrameStyle(QFrame.Shape.NoFrame)
        self._setup_ui()
        self._apply_style()

    def _setup_ui(self):
        """Setup the panel UI structure"""
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        if self.title:
            self._create_header()

        # Content area
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(16, 12, 16, 16)
        self.content_layout.setSpacing(12)

        self.main_layout.addWidget(self.content_widget)
        self.setLayout(self.main_layout)

    def _create_header(self):
        """Create collapsible header"""
        header_widget = QWidget()
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(16, 12, 16, 0)

        title_label = QLabel(self.title)
        title_label.setFont(Typography.heading_small())
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT_PRIMARY};
                background: transparent;
                border: none;
                padding: 4px 0px;
            }}
        """)

        header_layout.addWidget(title_label)
        self.main_layout.addWidget(header_widget)

    def _apply_style(self):
        """Apply modern panel styling"""
        self.setStyleSheet(f"""
            ModernPanel {{
                background: {Colors.SECONDARY_BG};
                border: 1px solid {Colors.BORDER_DEFAULT};
                border-radius: 12px;
                margin: 4px;
            }}
            ModernPanel:hover {{
                border-color: {Colors.BORDER_ACCENT};
            }}
        """)

    def add_widget(self, widget):
        """Add widget to content area"""
        self.content_layout.addWidget(widget)

    def add_layout(self, layout):
        """Add layout to content area"""
        self.content_layout.addLayout(layout)


class ModernComboBox(QComboBox):
    """Styled combobox matching the design system"""

    def __init__(self):
        super().__init__()
        self.setFont(Typography.body())
        self._apply_style()

    def _apply_style(self):
        """Apply modern combobox styling"""
        self.setStyleSheet(f"""
            QComboBox {{
                background: {Colors.ELEVATED_BG};
                border: 1px solid {Colors.BORDER_DEFAULT};
                border-radius: 6px;
                padding: 6px 12px;
                color: {Colors.TEXT_PRIMARY};
                min-width: 120px;
            }}
            QComboBox:hover {{
                border-color: {Colors.BORDER_ACCENT};
            }}
            QComboBox:focus {{
                border-color: {Colors.ACCENT_BLUE};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid {Colors.TEXT_SECONDARY};
                margin-right: 8px;
            }}
            QComboBox QAbstractItemView {{
                background: {Colors.ELEVATED_BG};
                border: 1px solid {Colors.BORDER_ACCENT};
                border-radius: 6px;
                selection-background-color: {Colors.HOVER_BG};
                selection-color: {Colors.TEXT_PRIMARY};
                color: {Colors.TEXT_PRIMARY};
                padding: 4px;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 8px 12px;
                border-radius: 4px;
                margin: 2px;
            }}
            QComboBox QAbstractItemView::item:selected {{
                background: {Colors.ACCENT_BLUE};
                color: {Colors.TEXT_PRIMARY};
            }}
        """)


# ================================
# LEGACY COMPATIBILITY
# ================================

# Keep old names for backward compatibility
StyledPanel = ModernPanel