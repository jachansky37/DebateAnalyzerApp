from transformers import pipeline
import json
import networkx as nx
import matplotlib.pyplot as plt
import mplcursors


def load_units(path="debate-analyzer/debate_unit_output/debate_units.json"):
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

    # Group units by idea_id
    idea_groups = {}
    for unit in units:
        idea_id = unit.get("idea_id", "unknown_idea")
        if idea_id not in idea_groups:
            idea_groups[idea_id] = []
        idea_groups[idea_id].append(unit)

    # Create idea nodes summarizing each idea group
    for idea_id, idea_units in idea_groups.items():
        combined_text = " ".join([u["text"] for u in idea_units])
        summary = summarizer(combined_text, max_length=15, min_length=5, truncation=True)[0]['summary_text']
        info_density = max((u.get("information_density", 0.5) for u in idea_units), default=0.5)
        fontsize = 10 + 15 * info_density
        G.add_node(idea_id, label=f"Idea {idea_id}: {summary}", color="lightblue", fontsize=fontsize, summary=summary)

        for u in idea_units:
            G.add_edge(idea_id, u["id"], label="has_unit")

    for unit in units:
        node_id = unit["id"]
        color = type_colors.get(unit["type"], "gray")
        G.add_node(node_id,
                   label=f"{node_id}: {unit['text'][:30]}...",
                   color=color,
                   fontsize=8,
                   summary=unit['text'])

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