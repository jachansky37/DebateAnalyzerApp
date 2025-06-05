from transformers import pipeline
import json
import networkx as nx
import matplotlib.pyplot as plt
import mplcursors


def load_units(path="output/debate_units.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_graph(units):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
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

    for unit in units:
        summary = summarizer(unit['text'], max_length=10, min_length=3, truncation=True)[0]['summary_text']
        node_id = unit["id"]
        info_density = unit.get("information_density", 0.5)
        fontsize = 10 + 15 * info_density
        color = type_colors.get(unit["type"], "gray")
        label = f"{node_id}: {summary}"
        G.add_node(node_id, label=label, color=color, fontsize=fontsize, summary=summary)

    for unit in units:
        parent_id = unit.get("parent_id")
        if parent_id is not None:
            relation = unit.get("relation_type", "")
            G.add_edge(parent_id, unit["id"], label=relation)

    return G

if __name__ == "__main__":
    units = load_units()
    G = build_graph(units)

    pos = nx.spring_layout(G, k=0.5, iterations=100)

    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    node_sizes = [G.nodes[n]['fontsize']*20 for n in G.nodes()]
    labels = {n: G.nodes[n]['label'] for n in G.nodes()}
    edge_labels = {(u, v): G.edges[u, v].get('label', '') for u, v in G.edges()}

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors='black')
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>')
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_family='Helvetica')

    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=7)

    # Add interactive hover to show summary text using mplcursors
    scatter = plt.scatter([], [])  # Dummy scatter for mplcursors
    node_positions = [pos[n] for n in G.nodes()]
    xs, ys = zip(*node_positions)
    scatter = plt.scatter(xs, ys, s=node_sizes, c=node_colors, alpha=0)

    cursor = mplcursors.cursor(scatter, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        # Find nearest node to the cursor position
        x, y = sel.target
        closest_node = None
        min_dist = float('inf')
        for n, (nx_pos, ny_pos) in zip(G.nodes(), node_positions):
            dist = (nx_pos - x)**2 + (ny_pos - y)**2
            if dist < min_dist:
                min_dist = dist
                closest_node = n
        if closest_node is not None:
            sel.annotation.set_text(G.nodes[closest_node]['summary'])
            sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

    plt.axis('off')
    plt.title("Debate Mindmap")
    plt.tight_layout()
    plt.show()