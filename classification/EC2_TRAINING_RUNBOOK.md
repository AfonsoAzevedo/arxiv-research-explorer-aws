# DS483 Classification: EC2 Training Runbook

## Project constraints

- AWS Region: `us-east-1`
- Project budget: $40-50 total
- Working mode: individual project
- Production dataset: 40,000 labeled arXiv papers
- Validation dataset: 77,187 papers from 2024
- Protected test dataset: do not access during development
- Full production training: **NOT AUTHORIZED**

Do not launch EC2, enable full training, or upload production artifacts
without separate approval through the project overview workflow.

## 1. Verified local implementation

The classification pipeline uses:

- TF-IDF with unigrams and bigrams
- Logistic Regression
- Eight primary arXiv categories
- Deterministic training selection
- Training diagnostics and reproducibility metadata
- SHA-256 verification
- Separate 2024 validation evaluation

A bounded 80-paper pilot was trained and evaluated locally.

These pilot results are not production-model results.

## 2. Verified source data

Training:

- File: `classification/data/train.jsonl`
- Records: 40,000
- Size: 58,052,946 bytes
- SHA-256:
  `d1aac2d0b524edda20cd8acf971a71d771c863523396d4b8c8db59ac6a985d29`

Validation:

- File: `classification/data/validation.jsonl`
- Records: 77,187
- Size: 122,569,012 bytes
- SHA-256:
  `33ea85f638606a4780c149bd348dd9256056c7d18feb48ff7b6e978cf8f9a138`

The S3 objects have matching reported sizes, but their contents
have not yet been independently verified after EC2 download.

## 3. EC2 environment

Previously verified infrastructure:

- Instance type: `m7i-flex.large`
- OS: Ubuntu 24.04
- Python: 3.12.3
- Memory: 8 GiB
- Connection: AWS Systems Manager Session Manager
- IAM role: `DS483-Arxiv-EC2-Role`

The previous test instance was terminated.

Before launching another instance:

1. Obtain explicit approval.
2. Review AWS cost and remaining budget.
3. Confirm the intended IAM instance profile.
4. Confirm SSM connectivity and instance health.
5. Record the new instance ID and launch time.

Do not assume the previous instance remains available.

## 4. Code and dependencies

Use the GitHub repository:

`AfonsoAzevedo/arxiv-research-explorer-aws`

Install dependencies into a dedicated Python virtual environment using:

`classification/requirements-ec2.txt`

The pinned wheel dependencies resolved for Linux x86_64 / Python 3.12
during a local download-only compatibility check.

Actual Linux installation and imports remain unverified.

## 5. Download and verify datasets

The downloader CLI was verified locally.

On an approved EC2 instance, after installing dependencies and
confirming the IAM role, run from the repository root:

    python3 -m classification.download_ec2_data --data-dir classification/data

Use a fresh destination directory. The downloader refuses to
overwrite existing training or validation files.

The overwrite rejection was observed directly on the Mac.
A successful live S3 download on EC2 remains unverified.

Required source objects:

- `s3://ds483-arxiv-test/classification/data/train.jsonl`
- `s3://ds483-arxiv-test/classification/data/validation.jsonl`

The downloader is designed to:

- Restrict downloads to approved objects
- Refuse existing destination paths
- Check downloaded file sizes
- Check SHA-256 and record counts
- Remove files published by a failed download run

Local mocked success, failure, and overwrite-protection tests passed.

Actual EC2 downloads remain unverified.

## 6. Linux smoke test

After environment and dataset verification:

1. Confirm Python and package versions.
2. Run the training CLI in dry-run mode.
3. Train a bounded pilot of 80 or 800 records.
4. Inspect convergence warnings, fit time, and peak RSS.
5. Confirm all three model artifacts are present.
6. Independently verify model SHA-256.
7. Reload the trusted model and perform inference.
8. Evaluate against the 2024 validation dataset.
9. Verify the evaluation JSON and record its metrics.
10. Confirm that full training remains disabled.

Do not access the protected 2025-2026 test dataset.

Do not proceed if the bounded pilot fails.

## 7. Production training: blocked

The current training CLI rejects `--full-train`.

Production preflight checks have passed locally:

- Exactly 40,000 training records
- Exactly 5,000 papers in each category
- Unique paper IDs
- Expected source-file SHA-256
- Exact match between provided inputs and verified source records

An authorization helper exists but is not connected to model fitting.

Production training must remain blocked until:

1. Linux smoke testing succeeds.
2. Runtime memory and costs are reviewed.
3. The project overview separately approves execution.
4. The production execution path is implemented and tested.

No production accuracy or runtime may be reported before measurement.

## 8. Artifact preservation

The uploader expects:

- `classifier.joblib`
- `training_metrics.json`
- `reproducibility.json`
- `validation_2024.json`

The intended S3 destination is:

`s3://ds483-arxiv-test/classification/models/<run-id>/`

Before upload, it checks local artifact consistency.

For each artifact, it performs:

1. Conditional upload without replacing an existing object
2. Independent download
3. SHA-256 verification

After all four files pass verification, it publishes and verifies:

`COMPLETE.json`

Mocked tests passed for successful publication, overwrite protection,
and an interrupted upload.

Actual S3 IAM permissions and live upload behavior remain unverified.

A completion marker is evidence of this uploader's successful checks,
not an immutable guarantee against subsequent object modification.

## 9. Cleanup and cost control

Before terminating an EC2 instance:

1. Confirm all required artifacts exist locally.
2. Verify every required object in S3.
3. Check `COMPLETE.json`.
4. Record S3 object keys and checksums.
5. Preserve logs and experiment metrics.
6. Only then terminate the EC2 instance.
7. Confirm the instance is terminated and no unexpected EBS volumes remain.
8. Review AWS billing and budget alerts.

Never report an estimated cloud cost as an actual measured charge.

## 10. Remaining verification

- [ ] Review and commit classification EC2 preparation code
- [ ] Obtain approval for a bounded EC2 smoke test
- [ ] Install pinned packages on Ubuntu 24.04 / Python 3.12
- [ ] Download training and validation data from S3
- [ ] Verify downloaded checksums and record counts
- [ ] Run bounded Linux pilot
- [ ] Check model reload, inference, diagnostics, and evaluation
- [ ] Test live artifact upload and completion marker
- [ ] Confirm cleanup and AWS charges
- [ ] Obtain separate approval for full production training
- [ ] Train and evaluate full 40,000-paper model
- [ ] Preserve verified production artifacts and metrics
