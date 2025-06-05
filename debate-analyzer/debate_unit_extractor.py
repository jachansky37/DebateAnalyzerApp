from dataclasses import dataclass, field
from typing import List, Optional
import json
import re
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import hdbscan

# -----------------------------
# 1. Define the DebateUnit model
# -----------------------------
@dataclass
class DebateUnit:
    id: int
    speaker: str
    start_time: Optional[str]
    end_time: Optional[str]
    text: str
    type: str  # e.g., 'claim', 'rebuttal', 'question', etc.
    good_faith: Optional[bool] = None
    topic: Optional[str] = None
    links_to: List[int] = field(default_factory=list)

    # new fields for logical flow/mindmap support
    parent_id: Optional[int] = None
    relation_type: Optional[str] = None  # 'supports', 'rebuts', 'answers', etc.
    thread_root_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    topic_cluster: Optional[int] = None

    embedding: Optional[List[float]] = None
    information_density: Optional[float] = None


# -----------------------------
# 2. Load and preprocess transcript
# -----------------------------
def load_transcript(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def simple_sentence_split(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s) > 10]  # skip very short chunks


# -----------------------------
# 3. Classify sentence type
# -----------------------------
def classify_argument_type(sentences: List[str]) -> List[str]:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    candidate_labels = ["claim", "rebuttal", "question", "response", "evidence", "concession", "statement"]
    labels = []

    for sent in sentences:
        cleaned_sent = re.sub(r'^SPEAKER [A-Z]+:\s*', '', sent.strip())

        try:
            result = classifier(cleaned_sent, candidate_labels=candidate_labels)
            top_label = result["labels"][0]
            top_score = result["scores"][0]

            print(f"\nText: {cleaned_sent}")
            print(f"Predicted: {top_label} with score {top_score:.2f}")

            if top_score > 0.3:
                labels.append(top_label)
            else:
                print(f"Low confidence ({top_score:.2f}) — marking as 'uncertain'")
                labels.append("uncertain")
        except Exception as e:
            print(f"Error: '{sent}' -> {e}")
            labels.append("unknown")

    return labels


# -----------------------------
# 4. Construct DebateUnits
# -----------------------------
def extract_speaker(text: str) -> str:
    match = re.match(r'^SPEAKER ([A-Z]+):', text.strip())
    return match.group(1) if match else "Unknown"

def construct_debate_units(sentences: List[str], labels: List[str]) -> List[DebateUnit]:
    from numpy.linalg import norm
    import numpy as np
    model = SentenceTransformer('all-MiniLM-L6-v2')

    units = []
    for idx, (text, label) in enumerate(zip(sentences, labels)):
        speaker = extract_speaker(text)
        cleaned_text = re.sub(r'^SPEAKER [A-Z]+:\s*', '', text.strip())

        unit = DebateUnit(
            id=idx,
            speaker=speaker,
            start_time=None,
            end_time=None,
            text=cleaned_text,
            type=label
        )
        unit.embedding = model.encode([cleaned_text])[0].tolist()
        unit.information_density = float(np.mean(np.abs(unit.embedding)))  # or use norm(unit.embedding)
        units.append(unit)
    return units


# -----------------------------
# 5. Save to JSON
# -----------------------------
def save_units(units: List[DebateUnit], output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([unit.__dict__ for unit in units], f, indent=2)


def assign_topic_clusters(units: List[DebateUnit]) -> None:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    filtered_units = [unit for unit in units if unit.embedding is not None]
    embeddings = [unit.embedding for unit in filtered_units]

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    labels = clusterer.fit_predict(embeddings)

    for unit, label in zip(filtered_units, labels):
        unit.topic_cluster = int(label) if label != -1 else None


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    raw_text = load_transcript("data/sample_debate_0.txt")
    sentences = simple_sentence_split(raw_text)
    print("Split into sentences:")
    for s in sentences:
        print(f"- {s}")
    labels = classify_argument_type(sentences)
    units = construct_debate_units(sentences, labels)
    def link_debate_units(units: List[DebateUnit]) -> None:
        last_claim_by_speaker = {}
        last_claim_global = None

        for idx, unit in enumerate(units):
            # Set thread root id for claims
            if unit.type == "claim":
                unit.thread_root_id = unit.id
                last_claim_by_speaker[unit.speaker] = unit.id
                last_claim_global = unit.id

            elif unit.type == "evidence":
                parent_id = last_claim_by_speaker.get(unit.speaker)
                if parent_id is not None:
                    unit.parent_id = parent_id
                    unit.relation_type = "supports"
                    units[parent_id].children_ids.append(unit.id)
                    unit.thread_root_id = units[parent_id].thread_root_id

            elif unit.type == "rebuttal":
                if last_claim_global is not None:
                    unit.parent_id = last_claim_global
                    unit.relation_type = "rebuts"
                    units[last_claim_global].children_ids.append(unit.id)
                    unit.thread_root_id = units[last_claim_global].thread_root_id

            elif unit.type == "answer":
                # naive strategy: attach to last question
                for prev in reversed(units[:idx]):
                    if prev.type == "question":
                        unit.parent_id = prev.id
                        unit.relation_type = "answers"
                        units[prev.id].children_ids.append(unit.id)
                        unit.thread_root_id = prev.thread_root_id or prev.id
                        break

            elif unit.type == "question" and unit.parent_id is None:
                # fallback: relate question to nearest previous claim or statement
                for prev in reversed(units[:idx]):
                    if prev.type in {"claim", "statement"}:
                        unit.parent_id = prev.id
                        unit.relation_type = "relates_to"
                        units[prev.id].children_ids.append(unit.id)
                        unit.thread_root_id = prev.thread_root_id or prev.id
                        break

    link_debate_units(units)
    assign_topic_clusters(units)
    save_units(units, "debate_unit_output/debate_units.json")