import json
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np

# Load sentence transformer and summarizer
embedder = SentenceTransformer('all-MiniLM-L6-v2')
summarizer = pipeline("summarization", model="google/pegasus-xsum")

def load_units(path="debate-analyzer/debate_unit_output/debate_units.json"):
    with open(path, "r") as f:
        return json.load(f)

def embed_sentences(units):
    return embedder.encode([unit['text'] for unit in units], convert_to_tensor=True)

def cluster_sentences(embeddings, distance_threshold=1.0):
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
    labels = clustering_model.fit_predict(embeddings.cpu().numpy())
    return labels

def group_by_cluster(units, labels):
    clusters = {}
    for unit, label in zip(units, labels):
        clusters.setdefault(label, []).append(unit)
    return clusters

def summarize_idea(units):
    text = " ".join(unit["text"] for unit in units)
    try:
        summary = summarizer(text, max_length=40, min_length=5, do_sample=False)[0]['summary_text']
    except:
        summary = text[:120] + "..."
    return summary

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
            "unit_ids": unit_ids
        })
    return ideas

def save_ideas(ideas, path="debate-analyzer/idea_unit_output/output_ideas.json"):
    with open(path, "w") as f:
        json.dump(ideas, f, indent=2)

if __name__ == "__main__":
    units = load_units()
    embeddings = embed_sentences(units)
    labels = cluster_sentences(embeddings)
    ideas = extract_ideas(units, labels)
    save_ideas(ideas)
