import json
from sentence_transformers import SentenceTransformer
import numpy as np
import hdbscan  # Replaces AgglomerativeClustering

# Load sentence transformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def load_units(path="debate-analyzer/debate_unit_output/debate_units.json"):
    with open(path, "r") as f:
        return json.load(f)

def filter_units(units, min_info_density=0.1):
    return [u for u in units if u.get("embedding") and u.get("information_density", 0) >= min_info_density]

def embed_sentences(units):
    return embedder.encode([unit['text'] for unit in units], convert_to_tensor=True)

def cluster_sentences(embeddings, min_cluster_size=2):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(embeddings.cpu().numpy())
    return labels

def group_by_cluster(units, labels):
    clusters = {}
    for unit, label in zip(units, labels):
        if label == -1:
            continue  # Skip noise
        clusters.setdefault(label, []).append(unit)
    return clusters

def summarize_idea(units):
    # Use most central or dense unit's short_text, truncated
    sorted_units = sorted(units, key=lambda u: -u.get("info_density", 0))
    for unit in sorted_units:
        text = unit.get("short_text") or unit.get("text", "")
        if text:
            words = text.split()
            return " ".join(words[:10]) + ("..." if len(words) > 10 else "")
    return "No summary available"

def extract_ideas(units, labels):
    clusters = group_by_cluster(units, labels)
    ideas = []
    for idea_id, (label, cluster_units) in enumerate(clusters.items()):
        summary = summarize_idea(cluster_units)
        info_density = np.mean([unit.get("info_density", 0.5) for unit in cluster_units])
        speaker_ids = list({unit["speaker"] for unit in cluster_units})
        unit_ids = [unit["id"] for unit in cluster_units]
        ideas.append({
            "idea_id": idea_id,
            "summary": summary,
            "info_density": info_density,
            "speaker_ids": speaker_ids,
            "unit_ids": unit_ids,
            "type": "idea"
        })
    return ideas

def save_ideas(ideas, path="debate-analyzer/idea_unit_output/idea_units.json"):
    with open(path, "w") as f:
        json.dump(ideas, f, indent=2)

if __name__ == "__main__":
    units = load_units()
    for u in units:
        print(f"Unit {u['id']} - Info Density: {u.get('information_density', 0)}")
    units = filter_units(units, min_info_density=0.01)

    if not units:
        print("No debate units passed the filter. Skipping idea extraction.")
        exit()

    embeddings = embed_sentences(units)

    # embeddings is a torch.Tensor, check if it's empty by shape[0]
    if embeddings.shape[0] == 0:
        print("No embeddings to cluster. Skipping idea extraction.")
        exit()

    labels = cluster_sentences(embeddings)
    ideas = extract_ideas(units, labels)
    save_ideas(ideas)
