import json
import random
from pathlib import Path

SOURCE = Path("data/arxiv-metadata-oai-snapshot.json")
OUTPUT = Path("semantic_search/data/pilot_10.jsonl")

SAMPLE_SIZE = 10
SEED = 483
MIN_ABSTRACT_CHARS = 100
MAX_INPUT_CHARS = 6000

rng = random.Random(SEED)
sample = []
eligible = 0
scanned = 0

with SOURCE.open("r", encoding="utf-8") as f:
    for line in f:
        scanned += 1
        paper = json.loads(line)

        paper_id = str(paper["id"]).strip()
        title = " ".join(paper["title"].split())
        abstract = " ".join(paper["abstract"].split())
        categories = paper["categories"].split()

        input_text = f"{title}\n\n{abstract}"

        if not (
            paper_id
            and title
            and MIN_ABSTRACT_CHARS <= len(abstract)
            and len(input_text) <= MAX_INPUT_CHARS
            and categories
        ):
            continue

        eligible += 1

        record = {
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "categories": categories,
            "input_text": input_text,
        }

        if len(sample) < SAMPLE_SIZE:
            sample.append(record)
        else:
            j = rng.randrange(eligible)
            if j < SAMPLE_SIZE:
                sample[j] = record

assert len(sample) == SAMPLE_SIZE
assert len({p["paper_id"] for p in sample}) == SAMPLE_SIZE

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8") as f:
    for paper in sample:
        f.write(json.dumps(paper, ensure_ascii=False) + "\n")

print("Total records scanned:", scanned)
print("Eligible papers:", eligible)
print("Selected papers:", len(sample))
print("Unique paper IDs:", len({p["paper_id"] for p in sample}))
print("Maximum input characters:", max(len(p["input_text"]) for p in sample))
print("Random seed:", SEED)
print("Output:", OUTPUT)
print("\nSelected papers:")

for paper in sample:
    print(f"  {paper['paper_id']} | {paper['title'][:75]}")

print("\nNo AWS API calls were made.")
