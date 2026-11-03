import hashlib
import json
import random
from collections import Counter
from pathlib import Path

SOURCE = Path("data/arxiv-metadata-oai-snapshot.json")
OUTPUT = Path("semantic_search/data/pilot_50.jsonl")
MANIFEST = Path("semantic_search/data/pilot_50_manifest.json")

QUOTAS = {
    "cs.LG": 20,
    "cs.AI": 10,
    "cs.CV": 5,
    "cs.CL": 5,
    "stat.ML": 5,
    "cs.RO": 5,
}

SEED = 48350
MIN_ABSTRACT_CHARS = 100
MAX_INPUT_CHARS = 6000

rng = random.Random(SEED)

reservoirs = {category: [] for category in QUOTAS}
eligible_counts = Counter()
seen_ids = set()
scanned = 0

with SOURCE.open(encoding="utf-8") as f:
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
        ):
            continue

        # Assign exactly one sampling group using fixed priority.
        group = next(
            (category for category in QUOTAS
             if category in categories),
            None,
        )

        if group is None:
            continue

        eligible_counts[group] += 1
        count = eligible_counts[group]
        quota = QUOTAS[group]

        record = {
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "categories": categories,
            "sampling_group": group,
            "input_text": input_text,
        }

        sample = reservoirs[group]

        if len(sample) < quota:
            sample.append(record)
        else:
            j = rng.randrange(count)
            if j < quota:
                sample[j] = record

papers = [
    paper
    for category in QUOTAS
    for paper in reservoirs[category]
]

assert len(papers) == 50
assert len({p["paper_id"] for p in papers}) == 50
assert all(
    len(reservoirs[c]) == quota
    for c, quota in QUOTAS.items()
)

for paper in papers:
    assert paper["sampling_group"] in paper["categories"]
    assert paper["input_text"] == (
        paper["title"] + "\n\n" + paper["abstract"]
    )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8") as f:
    for paper in papers:
        f.write(json.dumps(paper, ensure_ascii=False) + "\n")

sha256 = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()

manifest = {
    "source": str(SOURCE),
    "dataset": str(OUTPUT),
    "sha256": sha256,
    "seed": SEED,
    "sample_size": len(papers),
    "quotas": QUOTAS,
    "eligible_assigned_counts": dict(eligible_counts),
    "assignment_rule": "first matching category in quota order",
    "min_abstract_chars": MIN_ABSTRACT_CHARS,
    "max_input_chars": MAX_INPUT_CHARS,
    "embedding_status": "NOT_STARTED",
}

MANIFEST.write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)

print("50-PAPER SAMPLING PASSED")
print("Records scanned:", scanned)
print("Selected papers:", len(papers))
print("Unique paper IDs:", len({p["paper_id"] for p in papers}))
print("Random seed:", SEED)

print("\nGROUP ASSIGNMENTS")
for category, quota in QUOTAS.items():
    print(
        f"{category}: {len(reservoirs[category])} selected "
        f"from {eligible_counts[category]:,} assigned eligible"
    )

print("\nDATASET")
print("Output:", OUTPUT)
print("File bytes:", OUTPUT.stat().st_size)
print("SHA-256:", sha256)
print("Manifest:", MANIFEST)
print("AWS requests made: 0")
