import json
import sys
from PyQt5 import QtWidgets, QtGui
import pyqtgraph as pg
import numpy as np

class SemanticGraphViewer(QtWidgets.QMainWindow):
    def __init__(self, node_file="debate-analyzer/semantic_graph_output/semantic_nodes.json", edge_file="debate-analyzer/semantic_graph_output/semantic_edges.json"):
        super().__init__()
        self.setWindowTitle("Semantic Graph Viewer")
        self.resize(1200, 900)

        # Main container widget with horizontal layout
        main_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Left side: Graph view
        self.view = pg.GraphicsLayoutWidget()
        self.plot = self.view.addViewBox()
        self.plot.setAspectLocked()
        self.graph_item = pg.GraphItem()
        self.plot.addItem(self.graph_item)
        main_layout.addWidget(self.view, stretch=3)

        # Right side: Legend panel
        legend_panel = QtWidgets.QWidget()
        legend_layout = QtWidgets.QVBoxLayout()
        legend_panel.setLayout(legend_layout)
        legend_panel.setFixedWidth(200)  # adjust width as needed
        legend_label = QtWidgets.QLabel("<b>Legend:</b>")
        legend_layout.addWidget(legend_label)

        legend_colors = {
            'claim': 'yellow',
            'rebuttal': 'red',
            'question': 'blue',
            'answer': 'green',
            'evidence': 'magenta',
            'statement': 'gray',
            'response': 'orange',
            'concession': 'cyan',
            'uncertain': 'lightgray',
            'unknown': 'darkgray',
            'idea': 'cyan'
        }

        for label, color in legend_colors.items():
            entry = QtWidgets.QLabel(f"<font color='{color}'>●</font> {label}")
            legend_layout.addWidget(entry)

        legend_layout.addStretch()
        main_layout.addWidget(legend_panel, stretch=1)

        self.text_items = []
        self.load_graph(node_file, edge_file)

    def load_graph(self, node_path, edge_path):
        with open(node_path, 'r') as nf:
            nodes = json.load(nf)
        with open(edge_path, 'r') as ef:
            edges = json.load(ef)

        # Ensure all nodes have a "type" field before using it
        for node in nodes:
            if "type" not in node:
                node["type"] = "unit" if "unit_" in node["id"] else "idea"

        from sklearn.decomposition import PCA

        if "pos" not in nodes[0]:
            if "embedding" in nodes[0]:
                print("No 'pos' found in nodes. Computing positions via PCA on embeddings.")
                embeddings = np.array([node["embedding"] for node in nodes])
                pos = PCA(n_components=2).fit_transform(embeddings)
                for i, node in enumerate(nodes):
                    node["pos"] = pos[i].tolist()
            else:
                raise KeyError("'pos' and 'embedding' not found in node data. Cannot determine layout.")

        node_ids = [node["id"] for node in nodes]
        id_to_index = {nid: i for i, nid in enumerate(node_ids)}

        pos = np.array([node["pos"] for node in nodes], dtype=float)
        adj = np.array([[id_to_index[edge["source"]], id_to_index[edge["target"]]] for edge in edges])

        pos = self.apply_node_repulsion(pos)

        symbols = ['o' if node['type'] == 'unit' else 't' for node in nodes]
        sizes = [15 if node['type'] == 'unit' else 25 for node in nodes]

        def type_to_color(t):
            colors = {
                'claim': 'yellow',
                'rebuttal': 'red',
                'question': 'blue',
                'answer': 'green',
                'evidence': 'magenta',
                'statement': 'gray',
                'response': 'orange',
                'concession': 'cyan',
                'uncertain': 'lightgray',
                'unknown': 'darkGray',  # update from 'white' to 'darkGray'
                'unit': 'darkGray'      # ensure unit type is mapped if ever passed
            }
            return pg.mkBrush(colors.get(t, 'darkGray'))  # default to darkGray

        brushes = []
        for node in nodes:
            if node['type'] == 'unit':
                node_type = node.get('label_type', None)
                brushes.append(type_to_color(node_type))
            else:
                brushes.append(pg.mkBrush('cyan'))  # explicitly set for idea

        self.graph_item.setData(pos=pos, adj=adj, symbol=symbols, size=sizes,
                                symbolBrush=brushes, pxMode=True)
        # Add text labels for each node (short label above node, tooltip with full label, styled)
        self.text_items = []
        for i, node in enumerate(nodes):
            short_label = node.get("label", node["id"])
            full_label = node.get("full_label", short_label)
            text_item = pg.TextItem(
                html=f"<div style='background-color:black;color:white;padding:1px 3px;border-radius:3px;font-size:10px'>{short_label}</div>",
                anchor=(0.5, 0),
                angle=0
            )
            text_item.setPos(pos[i][0], pos[i][1] + 0.02)
            text_item.setToolTip(full_label)
            self.plot.addItem(text_item)
            self.text_items.append(text_item)

    def apply_node_repulsion(self, pos, min_dist=0.05, iterations=50, step_size=0.005):
        for _ in range(iterations):
            for i in range(len(pos)):
                for j in range(i + 1, len(pos)):
                    delta = pos[i] - pos[j]
                    dist = np.linalg.norm(delta)
                    if dist < min_dist and dist > 1e-5:
                        move = step_size * (min_dist - dist) * (delta / dist)
                        pos[i] += move
                        pos[j] -= move
        return pos

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    viewer = SemanticGraphViewer()
    viewer.show()
    sys.exit(app.exec_())