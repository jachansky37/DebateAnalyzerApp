import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_units(path="debate-analyzer/debate_unit_output/debate_units.json"):
    with open(path, "r") as f:
        return json.load(f)

def load_ideas(path="debate-analyzer/idea_unit_output/idea_units.json"):
    with open(path, "r") as f:
        return json.load(f)

def compute_similarity_matrix(embeddings):
    return cosine_similarity(embeddings)

def select_top_edges(similarity_matrix, nodes, top_k=2, similarity_threshold=0.55):
    edges = []
    for i, row in enumerate(similarity_matrix):
        sorted_indices = np.argsort(row)[::-1]
        selected = 0
        for j in sorted_indices:
            if i == j:
                continue  # skip self-loop
            if row[j] < similarity_threshold or selected >= top_k:
                break
            edges.append((nodes[i]['id'], nodes[j]['id']))
            selected += 1
    return edges

def best_sentence_summary(text, model, max_words=12):
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences:
        return text
    emb_text = model.encode([text])[0]
    emb_sents = model.encode(sentences)
    sims = cosine_similarity([emb_text], emb_sents)[0]
    best = max(zip(sentences, sims), key=lambda x: x[1])[0]
    words = best.split()
    return best if len(words) <= max_words else ' '.join(words[:max_words]) + '...'

def build_graph_data():
    units = load_units()
    ideas = load_ideas()

    # Merge both node types into a single list
    all_nodes = []

    model = SentenceTransformer('all-MiniLM-L6-v2')

    for unit in units:
        summary = best_sentence_summary(unit["text"], model)
        all_nodes.append({
            "id": f"unit_{unit['id']}",
            "label": summary,
            "full_text": unit["text"],
            "type": "unit",
            "label_type": unit.get("type", "unknown"),
            "embedding": unit.get("embedding"),
        })

    for idea in ideas:
        embedding = model.encode([idea["summary"]])[0].tolist()
        all_nodes.append({
            "id": f"idea_{idea['idea_id']}",
            "label": idea["summary"],
            "full_text": idea["summary"],
            "type": "idea",
            "embedding": embedding
        })

    # Get embeddings for units only
    unit_embeddings = [node["embedding"] for node in all_nodes if node["embedding"] is not None]
    unit_indices = [i for i, node in enumerate(all_nodes) if node["embedding"] is not None]

    if not unit_embeddings:
        raise ValueError("No embeddings found in unit data.")

    similarity_matrix = compute_similarity_matrix(unit_embeddings)

    # Build edges between units based on similarity
    unit_edges = select_top_edges(similarity_matrix, [all_nodes[i] for i in unit_indices])

    unit_nodes = [node for node in all_nodes if node["type"] == "unit"]
    idea_nodes = [node for node in all_nodes if node["type"] == "idea"]
    unit_embeddings = np.array([node["embedding"] for node in unit_nodes])
    idea_embeddings = np.array([node["embedding"] for node in idea_nodes])
    similarity_matrix = cosine_similarity(unit_embeddings, idea_embeddings)

    idea_edges = []
    for i, unit_node in enumerate(unit_nodes):
        sim_row = similarity_matrix[i]
        top_idea_indices = sim_row.argsort()[::-1][:1]  # top-1 idea
        for idx in top_idea_indices:
            idea_edges.append({
                "source": unit_node["id"],
                "target": idea_nodes[idx]["id"]
            })

    # Ensure all nodes have a 'type' field (if not already)
    for node in all_nodes:
        if "type" not in node:
            node["type"] = "unit" if "unit_" in node["id"] else "idea"
    graph_nodes = all_nodes

    graph_edges = [{"source": src, "target": tgt} for src, tgt in unit_edges]
    graph_edges.extend(idea_edges)

    with open("debate-analyzer/semantic_graph_output/semantic_nodes.json", "w") as f:
        json.dump(graph_nodes, f, indent=2)

    with open("debate-analyzer/semantic_graph_output/semantic_edges.json", "w") as f:
        json.dump(graph_edges, f, indent=2)

if __name__ == "__main__":
    build_graph_data()