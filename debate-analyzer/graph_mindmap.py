from transformers import pipeline
import json
import networkx as nx
import matplotlib.pyplot as plt
import mplcursors
from networkx.drawing.nx_agraph import graphviz_layout
import textwrap


def load_units(unit_path="debate-analyzer/debate_unit_output/debate_units.json", idea_path="debate-analyzer/idea_unit_output/idea_units.json"):
    with open(unit_path, "r", encoding="utf-8") as f1:
        units = json.load(f1)
    with open(idea_path, "r", encoding="utf-8") as f2:
        ideas = json.load(f2)
    return units, ideas

def build_graph(units, ideas):
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

    for idea in ideas:
        idea_id = idea.get("idea_id", f"idea_{ideas.index(idea)}")
        summary = idea.get("summary", f"Idea {idea_id}")
        fontsize = 25
        G.add_node(idea_id, label=summary, color="lightblue", fontsize=fontsize, summary=summary, shape='o')  # circle for ideas

    unit_to_idea = {}
    for idea in ideas:
        for uid in idea["unit_ids"]:
            unit_to_idea[uid] = idea["idea_id"]

    for unit in units:
        node_id = unit["id"]
        color = type_colors.get(unit["type"], "gray")
        # Wrap unit text for label
        short = unit.get("short_text", unit['text'])
        wrapped = "\n".join(textwrap.wrap(short, width=30))
        label = f"{node_id}: {wrapped}"
        G.add_node(
            node_id,
            label=label,
            color=color,
            fontsize=8,
            summary=unit['text'],
            shape='s'  # square for debate units
        )
        if node_id in unit_to_idea:
            G.add_edge(unit_to_idea[node_id], node_id, label="has_unit")

    for unit in units:
        parent_id = unit.get("parent_id")
        if parent_id is not None:
            relation = unit.get("relation_type", "")
            G.add_edge(parent_id, unit["id"], label=relation)

    return G

if __name__ == "__main__":
    units, ideas = load_units()
    G = build_graph(units, ideas)

    base_pos = graphviz_layout(G, prog='twopi')
    # Adjust vertical spread
    for node in base_pos:
        x, y = base_pos[node]
        base_pos[node] = (x, y * 1.15)
    pos = {n: (x * 1.0, y * 1.0) for n, (x, y) in base_pos.items()}

    # Cache nodes by shape for redraw
    shapes = {'o', 's'}
    shape_nodes = {shape: [n for n in G.nodes() if G.nodes[n].get('shape') == shape] for shape in shapes}
    shape_pos = {shape: {n: pos[n] for n in nodes} for shape, nodes in shape_nodes.items()}
    node_colors = {n: G.nodes[n]['color'] for n in G.nodes()}
    node_fontsizes = {n: G.nodes[n]['fontsize'] for n in G.nodes()}
    labels = {n: G.nodes[n]['label'] for n in G.nodes()}
    edge_labels = {(u, v): G.edges[u, v].get('label', '') for u, v in G.edges()}

    # view_scale = 1.0  # Global scale factor for zooming (removed)

    scatter = None  # Will hold reference to mplcursors scatter plot

    fig, ax = plt.subplots(figsize=(16, 12), dpi=150)

    def draw_graph():
        # global pos
        # pos = {n: (x * view_scale, y * view_scale) for n, (x, y) in base_pos.items()}  # removed scaling
        ax.clear()
        # Draw edges
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', ax=ax)

        # Draw nodes by shape with scaled sizes
        for shape in shapes:
            nodes = shape_nodes[shape]
            colors = [node_colors[n] for n in nodes]
            sizes = [G.nodes[n]['fontsize'] * 60 for n in nodes]  # updated to use current fontsize
            positions = {n: pos[n] for n in nodes}
            nx.draw_networkx_nodes(
                G, positions, nodelist=nodes, node_color=colors,
                node_size=sizes, node_shape=shape, edgecolors='black', ax=ax
            )

        # Draw labels with scaled font sizes
        for n in G.nodes():
            x, y = pos[n]
            ax.text(
                x, y, labels[n],
                fontsize=G.nodes[n]['fontsize'],  # updated to use current fontsize
                ha='center', va='center',
                fontfamily='DejaVu Sans',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5)
            )

        # Draw edge labels
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8, ax=ax)  # removed * view_scale

        ax.axis('off')
        ax.set_title("Debate Mindmap")
        plt.tight_layout()

    draw_graph()

    # Create mplcursors scatter plot for interactive hover
    node_positions = [pos[n] for n in G.nodes()]
    xs, ys = zip(*node_positions)
    node_sizes = [G.nodes[n]['fontsize'] * 60 for n in G.nodes()]  # updated to use current fontsize
    node_colors_list = [node_colors[n] for n in G.nodes()]
    scatter = ax.scatter(xs, ys, s=node_sizes, c=node_colors_list, alpha=0)

    cursor = mplcursors.cursor(scatter, hover=True)
    @cursor.connect("add")
    def on_add(sel):
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

    # Variables to track panning
    pan_start = {'x': None, 'y': None}
    orig_xlim = None
    orig_ylim = None

    def on_scroll(event):
        # global view_scale, scatter  # removed
        if event.inaxes != ax:
            return

        base_scale = 1.1
        # Detect if shift is pressed for pan simulation
        if event.key == 'shift':
            # Two-finger scroll to pan simulation: shift + scroll pans horizontally and vertically
            dx = -event.step * 20  # Pan amount scaled arbitrarily
            dy = -event.step * 20
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            ax.set_xlim([x + dx for x in cur_xlim])
            ax.set_ylim([y + dy for y in cur_ylim])
            ax.figure.canvas.draw_idle()
            return

        # Otherwise, treat as zoom (pinch to zoom simulation)
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata
        scale_factor = base_scale if event.step > 0 else 1 / base_scale
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
        ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])

        # Apply scaling to node font sizes
        for n in G.nodes():
            G.nodes[n]['fontsize'] *= scale_factor

        # Redraw graph with updated font sizes and node sizes
        draw_graph()

        ax.figure.canvas.draw_idle()

    def on_press(event):
        if event.inaxes != ax:
            return
        if event.button == 1:
            pan_start['x'] = event.x
            pan_start['y'] = event.y
            global orig_xlim, orig_ylim
            orig_xlim = ax.get_xlim()
            orig_ylim = ax.get_ylim()

    def on_drag(event):
        if event.inaxes != ax:
            return
        if event.button != 1:
            return
        if pan_start['x'] is None or pan_start['y'] is None:
            return

        dx = event.x - pan_start['x']
        dy = event.y - pan_start['y']

        # Convert pixel movement to data coordinates
        inv = ax.transData.inverted()
        start_data = inv.transform((pan_start['x'], pan_start['y']))
        current_data = inv.transform((event.x, event.y))
        delta_data = start_data - current_data
        new_xlim = (orig_xlim[0] + delta_data[0], orig_xlim[1] + delta_data[0])
        new_ylim = (orig_ylim[0] + delta_data[1], orig_ylim[1] + delta_data[1])

        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        ax.figure.canvas.draw_idle()

    def on_release(event):
        pan_start['x'] = None
        pan_start['y'] = None

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_drag)
    fig.canvas.mpl_connect('button_release_event', on_release)

    # Remove aspect ratio lock
    # plt.gca().set_aspect('auto', adjustable='datalim')  # Removed as requested

    # Add legend for node types
    import matplotlib.patches as mpatches

    idea_patch = mpatches.Patch(color='lightblue', label='Idea Node')
    claim_patch = mpatches.Patch(color='skyblue', label='Claim')
    rebuttal_patch = mpatches.Patch(color='lightcoral', label='Rebuttal')
    question_patch = mpatches.Patch(color='lightyellow', label='Question')
    answer_patch = mpatches.Patch(color='lightgreen', label='Answer')
    evidence_patch = mpatches.Patch(color='plum', label='Evidence')
    uncertain_patch = mpatches.Patch(color='gray', label='Uncertain')
    response_patch = mpatches.Patch(color='lightgray', label='Response')
    concession_patch = mpatches.Patch(color='khaki', label='Concession')
    statement_patch = mpatches.Patch(color='white', label='Statement')
    unknown_patch = mpatches.Patch(color='black', label='Unknown')

    plt.subplots_adjust(right=0.75)
    plt.legend(
        handles=[
            idea_patch, claim_patch, rebuttal_patch, question_patch, answer_patch,
            evidence_patch, uncertain_patch, response_patch, concession_patch,
            statement_patch, unknown_patch
        ],
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.,
        fontsize='small'
    )

    plt.show(block=True)