# gui/workflow_monitor_scene.py - Now with ExecutionEngine visualization!

from typing import Dict, List, Optional, Tuple
from PySide6.QtCore import Qt, QPointF, QTimer, QPropertyAnimation, QEasingCurve, Property, QByteArray, QRectF, Signal
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QPainterPath, QLinearGradient
from PySide6.QtWidgets import QGraphicsScene, QGraphicsPathItem, QGraphicsObject

from gui.agent_node import AgentNode
from gui.components import Colors


class ConnectionArrow(QGraphicsObject):
    """
    Animated arrow connecting two agent nodes.
    Features smooth curves and optional flow animation.
    Inherits from QGraphicsObject to support animations.
    """

    def __init__(self, start_node: AgentNode, end_node: AgentNode):
        super().__init__()

        self.start_node = start_node
        self.end_node = end_node
        self._flow_progress = 0.0
        self._is_active = False

        # Store path and style as member variables
        self._path = QPainterPath()
        self._pen = QPen()

        # Setup appearance
        self._setup_appearance()

        # Setup flow animation - 'self' is now a QObject, so this will work.
        self._flow_animation = QPropertyAnimation(self, QByteArray(b"flowProgress"))
        self._flow_animation.setDuration(1500)  # Faster flow
        self._flow_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._flow_animation.setLoopCount(-1)  # Infinite loop

        # Set Z-value to draw behind nodes
        self.setZValue(-1)

        # Update path
        self._update_path()

    def boundingRect(self) -> QRectF:
        """Required for QGraphicsObject: return the bounding rect of the item."""
        pen_width = self._pen.widthF()
        margin = pen_width / 2 + 5  # Add a small buffer
        return self._path.boundingRect().adjusted(-margin, -margin, margin, margin)

    def paint(self, painter: QPainter, option, widget=None):
        """Required for QGraphicsObject: custom painting logic."""
        painter.setPen(self._pen)
        painter.drawPath(self._path)

    def _setup_appearance(self):
        """Setup the visual appearance of the arrow by setting the member pen."""
        self._pen = QPen(QColor(Colors.BORDER_DEFAULT), 2)
        self._pen.setCapStyle(Qt.PenCapStyle.RoundCap)

    def _update_path(self, control_point_offset=80.0):
        """Update the path between nodes with smooth curve"""
        if not (self.start_node and self.end_node):
            return

        self.prepareGeometryChange()

        start_pos = self.mapFromItem(self.start_node, AgentNode.WIDTH / 2, AgentNode.HEIGHT)
        end_pos = self.mapFromItem(self.end_node, AgentNode.WIDTH / 2, 0)

        # A special case for the correction loop arrow going "backwards"
        is_correction_loop = self.start_node.agent_id == "execution_engine" and self.end_node.agent_id == "reviewer"

        path = QPainterPath()
        path.moveTo(start_pos)

        if is_correction_loop:
            # Draw a different kind of curve for the feedback loop
            mid_x = (start_pos.x() + end_pos.x()) / 2
            mid_y = (start_pos.y() + end_pos.y()) / 2
            control1 = QPointF(mid_x + control_point_offset, start_pos.y())
            control2 = QPointF(mid_x + control_point_offset, end_pos.y())
            path.cubicTo(control1, control2, end_pos)
        else:
            # Normal vertical flow
            dy = end_pos.y() - start_pos.y()
            control1 = QPointF(start_pos.x(), start_pos.y() + dy * 0.5)
            control2 = QPointF(end_pos.x(), end_pos.y() - dy * 0.5)
            path.cubicTo(control1, control2, end_pos)

        self._add_arrowhead(path, end_pos, path.pointAtPercent(0.95))
        self._path = path
        self.update()

    def _add_arrowhead(self, path: QPainterPath, tip: QPointF, from_point: QPointF):
        """Add arrowhead to the path"""
        direction = tip - from_point
        length = (direction.x() ** 2 + direction.y() ** 2) ** 0.5
        if length < 0.1: return

        direction /= length
        arrow_size = 10.0
        angle = 25.0

        p1 = tip - direction * arrow_size
        p2 = QPointF(p1.x() + direction.y() * arrow_size * 0.4, p1.y() - direction.x() * arrow_size * 0.4)
        p3 = QPointF(p1.x() - direction.y() * arrow_size * 0.4, p1.y() + direction.x() * arrow_size * 0.4)

        path.moveTo(p2)
        path.lineTo(tip)
        path.lineTo(p3)

    def set_active(self, active: bool, is_error: bool = False):
        """Activate or deactivate flow animation"""
        if active == self._is_active and self._pen.color() == (Colors.ACCENT_RED if is_error else Colors.ACCENT_BLUE):
            return

        self._is_active = active
        color = Colors.ACCENT_RED if is_error else Colors.ACCENT_BLUE

        if active:
            self._pen = QPen(QColor(color), 3)
            self._pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            self._flow_animation.setStartValue(0.0)
            self._flow_animation.setEndValue(1.0)
            self._flow_animation.start()
        else:
            self._flow_animation.stop()
            self._pen = QPen(QColor(Colors.BORDER_DEFAULT), 2)
            self._pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            self._flow_progress = 0.0

        self.update()

    def update_positions(self):
        """Update arrow path when nodes move"""
        self._update_path()

    def get_flow_progress(self):
        return self._flow_progress

    def set_flow_progress(self, progress):
        self._flow_progress = progress
        self.update()

    flowProgress = Property(float, get_flow_progress, set_flow_progress)


class WorkflowMonitorScene(QGraphicsScene):
    """
    Graphics scene that manages the visual layout of agent nodes and connections.
    Provides automatic layout and real-time updates.
    """
    layout_updated = Signal()

    NODE_SPACING_X = 220
    NODE_SPACING_Y = 130
    SCENE_MARGIN = 50

    def __init__(self):
        super().__init__()
        self._agent_nodes: Dict[str, AgentNode] = {}
        self._connections: Dict[Tuple[str, str], ConnectionArrow] = {}
        self._workflow_layout: List[List[str]] = []
        self._setup_scene()

        self._layout_timer = QTimer()
        self._layout_timer.timeout.connect(self._update_layout)
        self._layout_timer.setSingleShot(True)

    def _setup_scene(self):
        self.setBackgroundBrush(QBrush(QColor(Colors.PRIMARY_BG)))
        self.setSceneRect(0, 0, 800, 600)

    def add_agent_node(self, agent_id: str, agent_name: str, icon: str, row: int = 0, col: int = 0) -> AgentNode:
        if agent_id in self._agent_nodes:
            return self._agent_nodes[agent_id]
        node = AgentNode(agent_id, agent_name, icon)
        self.addItem(node)
        self._agent_nodes[agent_id] = node
        self._ensure_layout_size(row, col)
        self._workflow_layout[row][col] = agent_id
        self._schedule_layout_update()
        return node

    def add_connection(self, from_id: str, to_id: str) -> Optional[ConnectionArrow]:
        if from_id not in self._agent_nodes or to_id not in self._agent_nodes:
            return None
        if (from_id, to_id) in self._connections:
            return self._connections[(from_id, to_id)]

        arrow = ConnectionArrow(self._agent_nodes[from_id], self._agent_nodes[to_id])
        self.addItem(arrow)
        self._connections[(from_id, to_id)] = arrow
        return arrow

    def activate_connection(self, from_id: str, to_id: str, active: bool = True, is_error: bool = False):
        conn = self._connections.get((from_id, to_id))
        if conn:
            conn.set_active(active, is_error)

    def deactivate_connection(self, from_id: str, to_id: str):
        self.activate_connection(from_id, to_id, active=False)

    def update_agent_status(self, agent_id: str, status: str, status_text: str = ""):
        if agent_id in self._agent_nodes:
            self._agent_nodes[agent_id].set_status(status, status_text)

    def clear_workflow(self):
        self.clear()
        self._agent_nodes.clear()
        self._connections.clear()
        self._workflow_layout.clear()

    # --- MODIFIED: The new standard workflow! ---
    def setup_standard_workflow(self):
        """Setup the standard AvA workflow with the self-correction loop."""
        self.clear_workflow()

        # Add agent nodes in a logical flow
        # Row 0
        self.add_agent_node("architect", "Architect", "🏛️", 0, 0)

        # Row 1
        self.add_agent_node("coder", "Coder", "⚙️", 1, 0)

        # Row 2
        self.add_agent_node("assembler", "Assembler", "🧩", 2, 0)

        # Row 3 - The correction loop level
        self.add_agent_node("execution_engine", "Execution Engine", "🚀", 3, 0)
        self.add_agent_node("reviewer", "Reviewer", "🧐", 3, 1)

        # Add primary connections
        self.add_connection("architect", "coder")
        self.add_connection("coder", "assembler")
        self.add_connection("assembler", "execution_engine")

        # Add the self-correction loop connections
        self.add_connection("execution_engine", "reviewer")
        # Let's add a visual loop back from reviewer to assembler
        self.add_connection("reviewer", "assembler")

        self._schedule_layout_update()

    def refresh_workflow(self):
        for node in self._agent_nodes.values():
            node.set_status("idle", "Ready")
        for conn in self._connections.values():
            conn.set_active(False)
        self.update()

    def _ensure_layout_size(self, row: int, col: int):
        while len(self._workflow_layout) <= row:
            self._workflow_layout.append([])
        max_cols = max(len(r) for r in self._workflow_layout) if self._workflow_layout else 0
        max_cols = max(max_cols, col + 1)
        for r in self._workflow_layout:
            while len(r) < max_cols:
                r.append(None)

    def _schedule_layout_update(self):
        self._layout_timer.start(50)

    def _update_layout(self):
        if not self._workflow_layout: return

        # Calculate grid dimensions and center offset
        num_rows = len(self._workflow_layout)
        num_cols = max(len(row) for row in self._workflow_layout) if num_rows > 0 else 0
        grid_width = (num_cols - 1) * self.NODE_SPACING_X
        offset_x = (self.width() - grid_width) / 2

        # Position nodes
        for r, row_data in enumerate(self._workflow_layout):
            row_width = (len(row_data) - 1) * self.NODE_SPACING_X
            row_offset_x = (self.width() - row_width) / 2
            for c, agent_id in enumerate(row_data):
                if agent_id and agent_id in self._agent_nodes:
                    node = self._agent_nodes[agent_id]
                    x = row_offset_x + c * self.NODE_SPACING_X
                    y = self.SCENE_MARGIN + r * self.NODE_SPACING_Y
                    node.setPos(x, y)

        # Update all connection paths
        for (from_id, to_id), conn in self._connections.items():
            start_node = self._agent_nodes.get(from_id)
            end_node = self._agent_nodes.get(to_id)
            if start_node and end_node:
                # Add a special offset for the loopback arrow
                offset = 100.0 if from_id == "reviewer" and to_id == "assembler" else 80.0
                conn._update_path(control_point_offset=offset)

        self.setSceneRect(self.itemsBoundingRect().adjusted(-20, -20, 20, 20))