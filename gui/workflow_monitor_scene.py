# gui/workflow_monitor_scene.py - Graphics Scene for Workflow Visualization

from typing import Dict, List, Optional, Tuple
from PySide6.QtCore import Qt, QPointF, QTimer, QPropertyAnimation, QEasingCurve, Property, QByteArray, QRectF
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
        self._flow_animation.setDuration(2000)
        self._flow_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._flow_animation.setLoopCount(-1)  # Infinite loop

        # Set Z-value to draw behind nodes
        self.setZValue(-1)

        # Update path
        self._update_path()

    def boundingRect(self) -> QRectF:
        """Required for QGraphicsObject: return the bounding rect of the item."""
        pen_width = self._pen.widthF()
        margin = pen_width / 2 + 2  # Add a small buffer
        return self._path.boundingRect().adjusted(-margin, -margin, margin, margin)

    def paint(self, painter: QPainter, option, widget=None):
        """Required for QGraphicsObject: custom painting logic."""
        painter.setPen(self._pen)
        painter.drawPath(self._path)

    def _setup_appearance(self):
        """Setup the visual appearance of the arrow by setting the member pen."""
        self._pen = QPen(QColor(Colors.BORDER_DEFAULT), 2)
        self._pen.setCapStyle(Qt.PenCapStyle.RoundCap)

    def _update_path(self):
        """Update the path between nodes with smooth curve"""
        if not (self.start_node and self.end_node):
            return

        # Inform the scene that the geometry is about to change
        self.prepareGeometryChange()

        # Get connection points
        start_pos = self.start_node.pos() + QPointF(AgentNode.WIDTH, AgentNode.HEIGHT // 2)
        end_pos = self.end_node.pos() + QPointF(0, AgentNode.HEIGHT // 2)

        # Create smooth curved path
        path = QPainterPath()
        path.moveTo(start_pos)

        # Calculate control points for smooth curve
        dx = end_pos.x() - start_pos.x()
        control1 = QPointF(start_pos.x() + dx * 0.5, start_pos.y())
        control2 = QPointF(end_pos.x() - dx * 0.5, end_pos.y())

        path.cubicTo(control1, control2, end_pos)

        # Add arrowhead
        self._add_arrowhead(path, end_pos, start_pos)

        self._path = path
        self.update()  # Schedule a repaint

    def _add_arrowhead(self, path: QPainterPath, tip: QPointF, from_point: QPointF):
        """Add arrowhead to the path"""
        # Calculate arrow direction
        direction = tip - from_point
        length = (direction.x() ** 2 + direction.y() ** 2) ** 0.5
        if length == 0:
            return

        direction = QPointF(direction.x() / length, direction.y() / length)

        # Arrowhead geometry
        arrow_length = 8
        arrow_width = 4

        # Calculate arrowhead points
        base = tip - QPointF(direction.x() * arrow_length, direction.y() * arrow_length)
        perpendicular = QPointF(-direction.y(), direction.x())

        arrow_point1 = base + QPointF(perpendicular.x() * arrow_width, perpendicular.y() * arrow_width)
        arrow_point2 = base - QPointF(perpendicular.x() * arrow_width, perpendicular.y() * arrow_width)

        # Add arrowhead to path
        path.moveTo(arrow_point1)
        path.lineTo(tip)
        path.lineTo(arrow_point2)

    def set_active(self, active: bool):
        """Activate or deactivate flow animation"""
        if active == self._is_active:
            return

        self._is_active = active

        if active:
            # Animate to show data flow
            self._pen = QPen(QColor(Colors.ACCENT_BLUE), 3)
            self._pen.setCapStyle(Qt.PenCapStyle.RoundCap)

            # Start flow animation
            self._flow_animation.setStartValue(0.0)
            self._flow_animation.setEndValue(1.0)
            self._flow_animation.start()
        else:
            # Return to default appearance
            self._flow_animation.stop()
            self._pen = QPen(QColor(Colors.BORDER_DEFAULT), 2)
            self._pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            self._flow_progress = 0.0

        self.update()  # Schedule repaint

    def update_positions(self):
        """Update arrow path when nodes move"""
        self._update_path()

    # Property for flow animation
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

    # Layout constants
    NODE_SPACING_X = 220
    NODE_SPACING_Y = 130
    SCENE_MARGIN = 50

    def __init__(self):
        super().__init__()

        # Storage for nodes and connections
        self._agent_nodes: Dict[str, AgentNode] = {}
        self._connections: List[ConnectionArrow] = []
        self._workflow_layout: List[List[str]] = []  # 2D grid of agent IDs

        # Setup scene
        self._setup_scene()

        # Layout timer for smooth repositioning
        self._layout_timer = QTimer()
        self._layout_timer.timeout.connect(self._update_layout)
        self._layout_timer.setSingleShot(True)

    def _setup_scene(self):
        """Setup the scene with dark background"""
        self.setBackgroundBrush(QBrush(QColor(Colors.PRIMARY_BG)))

        # Set initial scene size
        self.setSceneRect(0, 0, 800, 600)

    def add_agent_node(self, agent_id: str, agent_name: str, icon: str, row: int = 0, col: int = 0) -> AgentNode:
        """Add a new agent node to the scene"""
        if agent_id in self._agent_nodes:
            return self._agent_nodes[agent_id]

        # Create node
        node = AgentNode(agent_id, agent_name, icon)
        self.addItem(node)
        self._agent_nodes[agent_id] = node

        # Update layout grid
        self._ensure_layout_size(row, col)
        self._workflow_layout[row][col] = agent_id

        # Schedule layout update
        self._schedule_layout_update()

        return node

    def remove_agent_node(self, agent_id: str):
        """Remove an agent node from the scene"""
        if agent_id not in self._agent_nodes:
            return

        node = self._agent_nodes[agent_id]

        # Remove connections involving this node
        self._remove_connections_for_node(node)

        # Remove from scene and storage
        self.removeItem(node)
        del self._agent_nodes[agent_id]

        # Update layout grid
        self._remove_from_layout(agent_id)

        # Schedule layout update
        self._schedule_layout_update()

    def add_connection(self, from_agent_id: str, to_agent_id: str) -> Optional[ConnectionArrow]:
        """Add a connection arrow between two agent nodes"""
        if from_agent_id not in self._agent_nodes or to_agent_id not in self._agent_nodes:
            return None

        start_node = self._agent_nodes[from_agent_id]
        end_node = self._agent_nodes[to_agent_id]

        # Check if connection already exists
        for connection in self._connections:
            if connection.start_node == start_node and connection.end_node == end_node:
                return connection

        # Create new connection
        arrow = ConnectionArrow(start_node, end_node)
        self.addItem(arrow)
        self._connections.append(arrow)

        return arrow

    def remove_connection(self, from_agent_id: str, to_agent_id: str):
        """Remove a connection between two agents"""
        if from_agent_id not in self._agent_nodes or to_agent_id not in self._agent_nodes:
            return

        start_node = self._agent_nodes[from_agent_id]
        end_node = self._agent_nodes[to_agent_id]

        # Find and remove connection
        for connection in self._connections[:]:  # Copy list for safe iteration
            if connection.start_node == start_node and connection.end_node == end_node:
                self.removeItem(connection)
                self._connections.remove(connection)
                break

    def activate_connection(self, from_agent_id: str, to_agent_id: str, active: bool = True):
        """Activate or deactivate connection animation"""
        if from_agent_id not in self._agent_nodes or to_agent_id not in self._agent_nodes:
            return

        start_node = self._agent_nodes[from_agent_id]
        end_node = self._agent_nodes[to_agent_id]

        # Find connection and set active state
        for connection in self._connections:
            if connection.start_node == start_node and connection.end_node == end_node:
                connection.set_active(active)
                break

    def update_agent_status(self, agent_id: str, status: str, status_text: str = ""):
        """Update the status of an agent node"""
        if agent_id in self._agent_nodes:
            self._agent_nodes[agent_id].set_status(status, status_text)

    def get_agent_node(self, agent_id: str) -> Optional[AgentNode]:
        """Get an agent node by ID"""
        return self._agent_nodes.get(agent_id)

    def clear_workflow(self):
        """Clear all nodes and connections"""
        # Remove all connections
        for connection in self._connections[:]:
            self.removeItem(connection)
        self._connections.clear()

        # Remove all nodes
        for node in self._agent_nodes.values():
            self.removeItem(node)
        self._agent_nodes.clear()

        # Clear layout
        self._workflow_layout.clear()

    def setup_standard_workflow(self):
        """Setup the standard AvA workflow with Architect -> Coders -> Reviewer"""
        self.clear_workflow()

        # Add agent nodes in workflow order
        architect = self.add_agent_node("architect", "Architect", "🏛️", 0, 1)
        coder1 = self.add_agent_node("coder", "Coder", "⚙️", 1, 0)
        reviewer = self.add_agent_node("reviewer", "Reviewer", "🧐", 2, 1)

        # Add connections
        self.add_connection("architect", "coder")
        self.add_connection("coder", "reviewer")

        # Update layout
        self._update_layout()

    def _ensure_layout_size(self, row: int, col: int):
        """Ensure the layout grid is large enough for the given position"""
        # Expand rows if needed
        while len(self._workflow_layout) <= row:
            self._workflow_layout.append([])

        # Expand columns if needed
        for i in range(len(self._workflow_layout)):
            while len(self._workflow_layout[i]) <= col:
                self._workflow_layout[i].append("")

    def _remove_from_layout(self, agent_id: str):
        """Remove agent ID from layout grid"""
        for row in self._workflow_layout:
            for i, cell in enumerate(row):
                if cell == agent_id:
                    row[i] = ""

    def _remove_connections_for_node(self, node: AgentNode):
        """Remove all connections involving the given node"""
        for connection in self._connections[:]:  # Copy list for safe iteration
            if connection.start_node == node or connection.end_node == node:
                self.removeItem(connection)
                self._connections.remove(connection)

    def _schedule_layout_update(self):
        """Schedule a layout update with small delay for batching"""
        self._layout_timer.start(100)  # 100ms delay

    def _update_layout(self):
        """Update the visual layout of all nodes and connections"""
        if not self._workflow_layout:
            return

        # Calculate grid dimensions
        max_cols = max(len(row) for row in self._workflow_layout) if self._workflow_layout else 0
        max_rows = len(self._workflow_layout)

        # Calculate total scene size needed
        total_width = max_cols * self.NODE_SPACING_X + 2 * self.SCENE_MARGIN
        total_height = max_rows * self.NODE_SPACING_Y + 2 * self.SCENE_MARGIN

        # Update scene size
        self.setSceneRect(0, 0, max(total_width, 800), max(total_height, 600))

        # Position nodes according to grid
        for row_idx, row in enumerate(self._workflow_layout):
            for col_idx, agent_id in enumerate(row):
                if agent_id and agent_id in self._agent_nodes:
                    node = self._agent_nodes[agent_id]

                    # Calculate position
                    x = self.SCENE_MARGIN + col_idx * self.NODE_SPACING_X
                    y = self.SCENE_MARGIN + row_idx * self.NODE_SPACING_Y

                    node.setPos(x, y)

        # Update all connection paths
        for connection in self._connections:
            connection.update_positions()

    def get_workflow_status_summary(self) -> Dict[str, str]:
        """Get a summary of all agent statuses"""
        return {
            agent_id: node.get_status()
            for agent_id, node in self._agent_nodes.items()
        }