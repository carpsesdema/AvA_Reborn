# gui/agent_node.py - Visual AI Agent Node for Workflow Monitor

from PySide6.QtCore import Qt, QRectF, QTimer, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QPainter, QPen, QBrush, QFont, QLinearGradient, QColor, QPainterPath
from PySide6.QtWidgets import QGraphicsObject, QGraphicsDropShadowEffect, QStyleOptionGraphicsItem

from gui.components import Colors, Typography


class AgentNode(QGraphicsObject):
    """
    Modern, animated visual representation of an AI agent in the workflow.
    Features smooth state transitions, animated borders, and clean typography.
    """

    # Node dimensions
    WIDTH = 180
    HEIGHT = 100
    BORDER_RADIUS = 12
    SHADOW_OFFSET = 4

    # Animation properties
    ANIMATION_DURATION = 300

    def __init__(self, agent_id: str, agent_name: str, icon: str, initial_status: str = "idle"):
        super().__init__()

        self.agent_id = agent_id
        self.agent_name = agent_name
        self.icon = icon
        self._status = initial_status
        self._status_text = "Ready"

        # Animation properties
        self._border_opacity = 0.0
        self._dash_offset = 0.0

        # Setup graphics effects
        self._setup_shadow_effect()

        # Setup animations
        self._setup_animations()

        # Start dash animation timer for working state
        self._dash_timer = QTimer()
        self._dash_timer.timeout.connect(self._update_dash_animation)
        self._dash_timer.setInterval(50)  # 20 FPS

        # Initial state setup
        self.set_status(initial_status, "Ready")

    def _setup_shadow_effect(self):
        """Add subtle drop shadow for depth"""
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(8)
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)

    def _setup_animations(self):
        """Setup smooth property animations"""
        self._border_animation = QPropertyAnimation(self, b"borderOpacity")
        self._border_animation.setDuration(self.ANIMATION_DURATION)
        self._border_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def boundingRect(self) -> QRectF:
        """Define the bounding rectangle with margin for shadow"""
        margin = self.SHADOW_OFFSET + 2
        return QRectF(-margin, -margin, self.WIDTH + 2 * margin, self.HEIGHT + 2 * margin)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget=None):
        """Custom paint method for modern, animated appearance"""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Main node rectangle
        node_rect = QRectF(0, 0, self.WIDTH, self.HEIGHT)

        # Background gradient
        bg_gradient = self._get_background_gradient()
        painter.setBrush(QBrush(bg_gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(node_rect, self.BORDER_RADIUS, self.BORDER_RADIUS)

        # Status border
        self._draw_status_border(painter, node_rect)

        # Content
        self._draw_content(painter, node_rect)

    def _get_background_gradient(self) -> QLinearGradient:
        """Create background gradient based on current status"""
        gradient = QLinearGradient(0, 0, 0, self.HEIGHT)

        if self._status == "idle":
            gradient.setColorAt(0, QColor(Colors.SECONDARY_BG))
            gradient.setColorAt(1, QColor(Colors.ELEVATED_BG))
        elif self._status == "working":
            gradient.setColorAt(0, QColor("#1a1f2e"))
            gradient.setColorAt(1, QColor("#2a2f3e"))
        elif self._status == "success":
            gradient.setColorAt(0, QColor("#1a2e1a"))
            gradient.setColorAt(1, QColor("#2a3e2a"))
        elif self._status == "error":
            gradient.setColorAt(0, QColor("#2e1a1a"))
            gradient.setColorAt(1, QColor("#3e2a2a"))

        return gradient

    def _draw_status_border(self, painter: QPainter, rect: QRectF):
        """Draw animated status border"""
        if self._status == "idle":
            # Subtle border for idle state
            painter.setPen(QPen(QColor(Colors.BORDER_DEFAULT), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect, self.BORDER_RADIUS, self.BORDER_RADIUS)
            return

        # Get status color
        status_colors = {
            "working": Colors.ACCENT_ORANGE,
            "success": Colors.ACCENT_GREEN,
            "error": Colors.ACCENT_RED
        }

        color = QColor(status_colors.get(self._status, Colors.BORDER_DEFAULT))
        color.setAlphaF(self._border_opacity)

        if self._status == "working":
            # Animated dashed border for working state
            pen = QPen(color, 2)
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setDashOffset(self._dash_offset)
            painter.setPen(pen)
        else:
            # Solid border for completed states
            painter.setPen(QPen(color, 2))

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, self.BORDER_RADIUS, self.BORDER_RADIUS)

    def _draw_content(self, painter: QPainter, rect: QRectF):
        """Draw icon, name, and status text"""
        painter.setPen(QColor(Colors.TEXT_PRIMARY))

        # Icon (large emoji)
        icon_font = QFont("Segoe UI Emoji", 24)
        painter.setFont(icon_font)
        icon_rect = QRectF(rect.left() + 15, rect.top() + 15, 40, 30)
        painter.drawText(icon_rect, Qt.AlignmentFlag.AlignCenter, self.icon)

        # Agent name
        name_font = Typography.heading_small()
        painter.setFont(name_font)
        name_rect = QRectF(rect.left() + 60, rect.top() + 15, rect.width() - 75, 25)
        painter.drawText(name_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.agent_name)

        # Status text - FIXED: Use proper text alignment flags
        painter.setPen(QColor(Colors.TEXT_SECONDARY))
        status_font = Typography.body_small()
        painter.setFont(status_font)
        status_rect = QRectF(rect.left() + 60, rect.top() + 40, rect.width() - 75, 40)

        # Use proper text alignment flags
        painter.drawText(status_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, self._status_text)

    def set_status(self, status: str, status_text: str = ""):
        """Update node status with smooth animation"""
        if status == self._status and status_text == self._status_text:
            return

        old_status = self._status
        self._status = status
        self._status_text = status_text or self._get_default_status_text(status)

        # Animate border opacity
        target_opacity = 1.0 if status != "idle" else 0.3
        self._border_animation.setStartValue(self._border_opacity)
        self._border_animation.setEndValue(target_opacity)
        self._border_animation.start()

        # Handle working state animation
        if status == "working":
            self._dash_timer.start()
        else:
            self._dash_timer.stop()
            self._dash_offset = 0.0

        # Force repaint
        self.update()

    def _get_default_status_text(self, status: str) -> str:
        """Get default status text for a given status"""
        defaults = {
            "idle": "Ready",
            "working": "Processing...",
            "success": "Complete",
            "error": "Failed"
        }
        return defaults.get(status, "Unknown")

    def _update_dash_animation(self):
        """Update dashed border animation"""
        self._dash_offset += 1.0
        if self._dash_offset > 10.0:  # Reset to prevent overflow
            self._dash_offset = 0.0
        self.update()

    # Property for animations
    def get_border_opacity(self):
        return self._border_opacity

    def set_border_opacity(self, opacity):
        self._border_opacity = opacity
        self.update()

    borderOpacity = Property(float, get_border_opacity, set_border_opacity)

    def get_status(self) -> str:
        """Get current status"""
        return self._status

    def get_status_text(self) -> str:
        """Get current status text"""
        return self._status_text