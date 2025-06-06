import json
from PyQt5 import QtWidgets, QtGui
import pyqtgraph as pg
import sys

def load_units(unit_path="debate-analyzer/debate_unit_output/debate_units.json", idea_path="debate-analyzer/idea_unit_output/idea_units.json"):
    with open(unit_path, "r", encoding="utf-8") as f1:
        units = json.load(f1)
    with open(idea_path, "r", encoding="utf-8") as f2:
        ideas = json.load(f2)
    return units, ideas

import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg


class GraphViewer(QtWidgets.QWidget):
    def __init__(self, nodes, edges):
        super().__init__()

        self.setWindowTitle("Debate Mindmap Viewer")
        self.resize(800, 600)

        # Setup layout
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        # Graphics layout widget from pyqtgraph
        self.graph_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graph_widget)

        # Create a view box for panning and zooming
        self.view_box = self.graph_widget.addViewBox()
        self.view_box.setAspectLocked()
        self.view_box.invertY(False)

        # Enable mouse interactions: zoom with scroll/pinch, pan with two-finger drag
        self.view_box.setMouseEnabled(x=True, y=True)

        # Create a graphics item group to hold nodes and edges
        self.graph_item = pg.GraphItem()
        self.view_box.addItem(self.graph_item)

        # Legend setup
        self.legend = pg.LegendItem(offset=(10, 10))
        self.legend.setParentItem(self.view_box)

        self.nodes = nodes
        self.edges = edges
        self._draw_graph()

    def _draw_graph(self):
        # Prepare data arrays
        pos = []
        symbols = []
        sizes = []
        brushes = []
        pens = []
        texts = []
        text_items = []

        # Map node id to index for edges
        id_to_index = {node['id']: i for i, node in enumerate(self.nodes)}

        for node in self.nodes:
            x, y = node['pos']
            pos.append([x, y])
            if node['type'] == 'debate':
                symbols.append('s')  # square for debate
                sizes.append(40)
                brushes.append(pg.mkBrush(100, 150, 255))
                pens.append(pg.mkPen('k'))
            else:
                symbols.append('o')  # circle for idea
                sizes.append(30)
                brushes.append(pg.mkBrush(150, 255, 150))
                pens.append(pg.mkPen('k'))

        pos = np.array(pos)

        # Set graph data
        edges = np.array([(id_to_index[e[0]], id_to_index[e[1]]) for e in self.edges], dtype=int)
        self.graph_item.setData(pos=pos, adj=edges, size=sizes, symbol=symbols,
                                symbolBrush=brushes, symbolPen=pens, pxMode=False)

        # Add labels that scale with zoom
        for i, node in enumerate(self.nodes):
            text = pg.TextItem(node['label'], anchor=(0.5, -0.3))
            text.setParentItem(self.graph_item)
            text.setPos(pos[i][0], pos[i][1])
            text.setFont(QtGui.QFont('Arial', 10))
            text_items.append(text)

        self.text_items = text_items

        # Add legend entries
        self.legend.clear()
        # Create sample legend symbols
        debate_symbol = pg.ScatterPlotItem(size=20, brush=pg.mkBrush(100, 150, 255), pen=pg.mkPen('k'), symbol='s')
        idea_symbol = pg.ScatterPlotItem(size=15, brush=pg.mkBrush(150, 255, 150), pen=pg.mkPen('k'), symbol='o')
        self.legend.addItem(debate_symbol, 'Debate')
        self.legend.addItem(idea_symbol, 'Idea')

        # Placeholder for integration with debate/idea data
        # TODO: Replace sample data with actual debate and idea nodes and edges

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.legend.setPos(10, 10)

    def wheelEvent(self, event):
        # Override wheel event for zooming
        delta = event.angleDelta().y()
        scale_factor = 1.2 if delta > 0 else 1/1.2
        self.view_box.scaleBy((1/scale_factor, 1/scale_factor))

        # Scale text items inversely to keep them readable
        for text in self.text_items:
            current_font = text.font()
            new_size = max(6, current_font.pointSizeF() * scale_factor)
            current_font.setPointSizeF(new_size)
            text.setFont(current_font)


def make_short_text(text, max_words=15):
    return ' '.join(text.strip().split()[:max_words])

def build_graph_data(units, ideas):
    nodes = []
    edges = []
    unit_map = {unit["id"]: unit for unit in units}
    unit_to_idea = {}

    for idea in ideas:
        idea_id = idea.get("idea_id", f"idea_{ideas.index(idea)}")
        label = idea.get("summary", f"Idea {idea_id}")
        nodes.append({
            "id": idea_id,
            "label": label,
            "type": "idea",
            "fulltext": idea.get("summary", ""),
        })
        for uid in idea.get("unit_ids", []):
            unit_to_idea[uid] = idea_id
            edges.append((idea_id, uid))

    for unit in units:
        node_id = unit["id"]
        label = make_short_text(unit.get("short_text", unit["text"]))
        nodes.append({
            "id": node_id,
            "label": label,
            "type": "unit",
            "subtype": unit.get("type", "unknown"),
            "fulltext": unit["text"],
        })
        if node_id in unit_to_idea:
            edges.append((unit_to_idea[node_id], node_id))
        if unit.get("parent_id"):
            edges.append((unit["parent_id"], node_id))

    return nodes, edges

class MindMapGraph(pg.GraphicsLayoutWidget):
    def __init__(self, nodes, edges, parent=None):
        super().__init__(parent)
        self.view = self.addViewBox()
        self.view.setAspectLocked(False)
        self.graph = pg.GraphItem()
        self.view.addItem(self.graph)
        self.nodes = nodes
        self.edges = edges
        self.node_id_to_index = {node["id"]: idx for idx, node in enumerate(nodes)}
        self.text_items = []
        self.view_scale = 1.0  # Track overall zoom level
        self.base_font_sizes = []
        self.tooltips = []
        self._init_graph()

    def _init_graph(self):
        import numpy as np
        # Improved edge rendering to mimic the old graphviz_layout behavior
        from networkx.drawing.nx_agraph import graphviz_layout
        import networkx as nx

        type_colors = {
            "claim": "skyblue",
            "rebuttal": "lightcoral",
            "question": "lightyellow",
            "answer": "lightgreen",
            "evidence": "plum",
            "uncertain": "gray",
            "response": "lightgray",
            "concession": "khaki",
            "statement": "white",
            "unknown": "black"
        }

        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(node['id'])
        for src, dst in self.edges:
            G.add_edge(src, dst)
        layout_pos = graphviz_layout(G, prog='twopi')
        scale = 2.0
        pos = np.array([[x * scale, y * scale] for x, y in layout_pos.values()])

        # Ensure we iterate in the same order as layout_pos
        layout_keys = list(layout_pos.keys())
        self.nodes = [next(node for node in self.nodes if node['id'] == node_id) for node_id in layout_keys]

        symbols = []
        sizes = []
        brushes = []
        labels = []

        for node in self.nodes:
            if node["type"] == "idea":
                symbols.append('o')
                brushes.append(pg.mkBrush('lightblue'))
            else:
                symbols.append('s')
                brushes.append(pg.mkBrush(type_colors.get(node.get("subtype", ""), 'gray')))
            sizes.append(80)
            labels.append(node["label"])

        # Debug print to check lengths
        print(f"Positions: {len(pos)}, Symbols: {len(symbols)}, Brushes: {len(brushes)}, Sizes: {len(sizes)}")

        edges = [(layout_keys.index(src), layout_keys.index(dst)) for src, dst in self.edges]

        self.min_edge_lengths = [float('inf')] * len(pos)
        for i1, i2 in edges:
            d = np.linalg.norm(pos[i1] - pos[i2])
            self.min_edge_lengths[i1] = min(self.min_edge_lengths[i1], d)
            self.min_edge_lengths[i2] = min(self.min_edge_lengths[i2], d)
        # Replace inf values with a default to avoid scaling issues on isolated nodes
        self.min_edge_lengths = [
        d if d != float('inf') else 100.0
        for d in self.min_edge_lengths
        ]
        self.graph.setData(pos=pos, adj=np.array(edges), size=sizes, symbol=symbols,
                           symbolBrush=brushes, pxMode=True)
        self.graph.setPen(pg.mkPen('w'))
        self.graph.size = sizes
        self.base_sizes = sizes
        self.graph_data = {
            'pos': pos,
            'adj': np.array(edges),
            'symbol': symbols,
            'symbolBrush': brushes
        }

        for i, (x, y) in enumerate(pos):
            text = pg.TextItem(self.nodes[i]["label"], anchor=(0.5, 0.5))
            text.setColor('w')
            text.setPos(x, y)
            text.setFont(QtGui.QFont("Arial", 16))
            text.setToolTip(self.nodes[i]["fulltext"])
            self.view.addItem(text)
            self.text_items.append(text)
            self.base_font_sizes.append(16)

        # self.graph.sigClicked.connect(self.on_node_clicked)

    def on_node_clicked(self, graph_item, points):
        for p in points:
            idx = int(p.index())
            node = self.nodes[idx]
            print(f"Clicked: {node['id']} — {node['fulltext']}")

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        scale_factor = 1.2 if delta > 0 else 1 / 1.2

        new_view_scale = self.view_scale * scale_factor
        new_view_scale = min(max(new_view_scale, 0.05), 20.0)
        scale_factor = new_view_scale / self.view_scale
        self.view.scaleBy((scale_factor, scale_factor))
        self.view_scale = new_view_scale

        new_sizes = []
        for i, base in enumerate(self.base_sizes):
            max_size = 0.3 * self.min_edge_lengths[i]
            new_sizes.append(max(5, min(base / self.view_scale, max_size)))

        self.graph.setData(
            pos=self.graph_data['pos'],
            adj=self.graph_data['adj'],
            symbol=self.graph_data['symbol'],
            symbolBrush=self.graph_data['symbolBrush'],
            size=new_sizes,
            pxMode=True
        )

        for i, text in enumerate(self.text_items):
            base_size = self.base_font_sizes[i]
            max_font = 0.1 * self.min_edge_lengths[i]
            adjusted_size = max(6, min(base_size / self.view_scale, max_font))
            font = QtGui.QFont("Arial")
            font.setPointSizeF(adjusted_size)
            text.setFont(font)
            text.update()

if __name__ == '__main__':
    import numpy as np
    app = QtWidgets.QApplication([])
    units, ideas = load_units()
    nodes, edges = build_graph_data(units, ideas)
    win = MindMapGraph(nodes, edges)
    win.setWindowTitle('Debate Mindmap Viewer (PyQtGraph)')
    win.resize(1000, 800)
    win.show()
    sys.exit(app.exec_())