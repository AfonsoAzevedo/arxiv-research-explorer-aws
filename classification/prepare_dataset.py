"""
Prepare reproducible arXiv classification datasets.

Dataset:
    Cornell University arXiv metadata snapshot (JSONL).

Methodology:
    - Eight selected primary arXiv categories.
    - Primary category = first category in metadata.
    - Minimum abstract length: 30 words.
    - Normalize whitespace for abstract comparison.
    - Exclude exact-duplicate groups with conflicting labels.
    - Retain the earliest submitted record for same-label duplicates.
    - Training: submitted before 2024.
    - Validation: submitted during 2024.
    - Testing: submitted during 2025 or later.
    - Sample 5,000 training papers per category.
    - Keep validation and test distributions unchanged.
"""

import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import UTC
from email.utils import parsedate_to_datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Point into the metadata dataset
SOURCE_PATH = PROJECT_ROOT / "data" / "arxiv-metadata-oai-snapshot.json"

OUTPUT_DIR = Path(__file__).resolve().parent / "data"

# My project focus on a defined subset
CATEGORIES = (
    "cs.AI",
    "cs.CL",
    "cs.CR",
    "cs.CV",
    "cs.DB",
    "cs.LG",
    "cs.NI",
    "cs.RO",
)

# Config constants
MIN_ABSTRACT_WORDS = 30
TRAIN_SAMPLES_PER_CATEGORY = 5000  # 5K papers per category
RANDOM_SEED = 42


# Normalize white lines. Every sentence "word word word ..."
def normalize_abstract(text: str) -> str:
    """Normalize whitespace while preserving scientific terminology"""
    return " ".join(text.split())


# Normalize the abstract. Convert to lower case and create a deterministic identifier
def abstract_hash(text: str) -> str:
    """Create a deterministic hash for exact-duplicate detection"""
    normalized = normalize_abstract(text).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# Have the date of the paper. v1 is the original submission
def original_submission_date(paper: dict):
    """Extract the original v1 submission timestamp in UTC"""
    versions = paper["versions"]

    if not versions or versions[0]["version"] != "v1":
        raise ValueError(f"Invalid original version for paper {paper['id']}")

    return parsedate_to_datetime(versions[0]["created"]).astimezone(UTC)


def main():
    print("DS483 Classification Dataset Preparation")
    print("----------------------------------------")
    print("Source:", SOURCE_PATH)

    # Check that source exists
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(SOURCE_PATH)

    # Initialize the data structures
    groups = defaultdict(list)  # Groups papers with identical normalized abstracts
    short_removed = (
        Counter()
    )  # 	Counts short abstracts excluded from each selected category
    scanned = 0  # Counts all metadata records read

    # Read the dataset one record at a time
    with SOURCE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            paper = json.loads(line)
            scanned += 1

            # Select the primary category
            primary = paper["categories"].split()[0]
            if primary not in CATEGORIES:
                continue

            # Normalize abstract
            abstract = normalize_abstract(paper["abstract"])

            if len(abstract.split()) < MIN_ABSTRACT_WORDS:
                short_removed[primary] += 1
                continue

            # Determine the date and abstract hash
            created = original_submission_date(paper)
            digest = abstract_hash(abstract)

            # Construct a standardized record
            record = {
                "id": paper["id"],
                "title": normalize_abstract(paper["title"]),
                "abstract": abstract,
                "label": primary,
                "categories": paper["categories"].split(),
                "submitted": created.isoformat(),
                "abstract_sha256": digest,
            }

            # Group identical abstracts
            groups[digest].append(record)

    print("Total records scanned:", scanned)
    print("Unique abstract groups:", len(groups))

    # Initialize the counters
    eligible = []
    conflicting_groups = 0
    conflicting_records = 0
    duplicates_removed = 0

    # Examine each duplicate group
    for records in groups.values():
        labels = {r["label"] for r in records}

        # Remove groups with conflicting labels
        if len(labels) > 1:
            conflicting_groups += 1
            conflicting_records += len(records)
            continue

        # Sort same-label duplicates by date
        records.sort(key=lambda r: (r["submitted"], r["id"]))
        eligible.append(records[0])  # Keep the earliest record
        duplicates_removed += len(records) - 1  # Count the removed duplicates

    # Create the three partitions
    partitions = {
        "train": defaultdict(list),
        "validation": defaultdict(list),
        "test": defaultdict(list),
    }

    # Process each eligible paper
    for record in eligible:
        year = int(record["submitted"][:4])

        # Assign the chronological split
        if year < 2024:
            split = "train"
        elif year == 2024:
            split = "validation"
        else:
            split = "test"

        partitions[split][record["label"]].append(record)

    rng = random.Random(RANDOM_SEED)  # Initialize the random generator

    # Process each category
    training = []
    for category in CATEGORIES:
        candidates = partitions["train"][category]

        # Check that enough papers exist
        if len(candidates) < TRAIN_SAMPLES_PER_CATEGORY:
            raise ValueError(
                f"Insufficient training papers for {category}: {len(candidates)}"
            )

        candidates = sorted(
            candidates, key=lambda r: r["id"]
        )  # Sort candidates before sampling
        # Randomly select 5,000 papers
        training.extend(rng.sample(candidates, TRAIN_SAMPLES_PER_CATEGORY))

    # Construct validation and test datasets
    validation = [
        record
        for category in CATEGORIES
        for record in partitions["validation"][category]
    ]

    test = [
        record for category in CATEGORIES for record in partitions["test"][category]
    ]

    # Create the final datasets
    datasets = {
        "train": sorted(training, key=lambda r: r["id"]),
        "validation": sorted(validation, key=lambda r: r["id"]),
        "test": sorted(test, key=lambda r: r["id"]),
    }

    # Collect the abstract hashes for each split
    all_hashes = {
        split: {r["abstract_sha256"] for r in records}
        for split, records in datasets.items()
    }

    # Check for overlap between datasets
    assert all_hashes["train"].isdisjoint(all_hashes["validation"])
    assert all_hashes["train"].isdisjoint(all_hashes["test"])
    assert all_hashes["validation"].isdisjoint(all_hashes["test"])

    # Count papers by category
    output_counts = {
        split: dict(sorted(Counter(r["label"] for r in records).items()))
        for split, records in datasets.items()
    }

    manifest = {
        "source": str(SOURCE_PATH),
        "total_records_scanned": scanned,
        "categories": list(CATEGORIES),
        "minimum_abstract_words": MIN_ABSTRACT_WORDS,
        "training_samples_per_category": TRAIN_SAMPLES_PER_CATEGORY,
        "random_seed": RANDOM_SEED,
        "short_abstracts_removed": sum(short_removed.values()),
        "conflicting_groups_removed": conflicting_groups,
        "conflicting_records_removed": conflicting_records,
        "same_label_duplicates_removed": duplicates_removed,
        "eligible_records": len(eligible),
        "split_policy": {
            "train": "original submission before 2024; sampled",
            "validation": "original submission during 2024; all eligible",
            "test": "original submission during 2025+; all eligible",
        },
        "split_counts": {split: len(records) for split, records in datasets.items()},
        "category_counts": output_counts,
    }

    # Create the output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split, records in datasets.items():
        destination = OUTPUT_DIR / f"{split}.jsonl"

        # Save the prepared records in JSONL format (each line contains one paper)
        with destination.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"{split}: {len(records):,} records -> {destination}")

    # Save the manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nAccounting check:")
    print(
        "Eligible + exclusions:",
        len(eligible)
        + sum(short_removed.values())
        + conflicting_records
        + duplicates_removed,
    )
    print("Manifest:", manifest_path)
    print("Dataset preparation complete.")

if __name__ == "__main__":
    main()
