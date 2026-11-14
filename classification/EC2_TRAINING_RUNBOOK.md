# DS483 EC2 Classification Training and Reproducibility

I used Amazon EC2 to train my production TF-IDF and Logistic Regression classifier on 40,000 arXiv abstracts. This document explains the training workflow, the files I preserved, and the checks I use to protect my results.

- **AWS region:** `us-east-1`
- **Project budget:** $40 to $50 usage target, not a measured final charge
- **Status:** Production training, validation, and the single held-out test evaluation are complete

This replaces my earlier planning checklist. I do not need to repeat the EC2 training run to demonstrate or evaluate the saved classifier.

## 1. Training data

I prepared the datasets with `classification/prepare_dataset.py` from the full arXiv metadata snapshot.

| Dataset | Papers | Submission period | Use |
|---|---:|---|---|
| Training | 40,000 | Before 2024 | Fit TF-IDF and Logistic Regression |
| Validation | 77,187 | 2024 | Measure performance and review the model |
| Held-out test | 207,257 | 2025 to 2026 | One final evaluation after freezing the model |

I sampled 5,000 training abstracts from each of eight categories using random seed 42. I left validation and test class frequencies unchanged. I removed exact duplicate abstract groups before splitting the data.

My committed `classification/ec2-data-manifest.json` records the expected training and validation file sizes, record counts, and SHA-256 checksums.

## 2. AWS workflow

I used the following workflow:

1. I prepared and validated the training and validation datasets locally.
2. I stored the datasets in Amazon S3.
3. I prepared S3 download and integrity-check scripts for the approved files. The repository does not contain the full EC2 execution logs for these steps.
4. I trained the TF-IDF and Logistic Regression pipeline on EC2.
5. I evaluated the saved model on the separate 2024 validation data.
6. I preserved the model, training diagnostics, reproducibility metadata, validation results, and S3 completion marker.
7. I froze the model before the single 2025 to 2026 held-out test evaluation.

The project used EC2 infrastructure in `us-east-1`. My earlier verified environment was an `m7i-flex.large` instance with Ubuntu 24.04, Python 3.12.3, 8 GiB RAM, and Systems Manager Session Manager access. I do not assume that instance or its exact configuration is still running.

## 3. Source code and roles

| File | Purpose |
|---|---|
| `prepare_dataset.py` | Prepare chronological splits and remove exact duplicates |
| `verify_ec2_data.py` | Verify downloaded dataset sizes, checksums, and record counts |
| `download_ec2_data.py` | Download approved files from S3 without overwriting local data |
| `train.py` | Build the TF-IDF and Logistic Regression pipeline and fit an authorized run |
| `diagnostics.py` | Record fitting time, memory, warnings, and reproducibility metadata |
| `evaluate.py` | Evaluate the fitted model on 2024 validation data |
| `upload_ec2_artifacts.py` | Upload artifacts to S3 and independently verify their checksums |
| `evaluate_test.py` | Run the protected held-out evaluation once with a frozen model |

The training pipeline uses word unigrams and bigrams, up to 100,000 TF-IDF features, and Logistic Regression. I fit the vectorizer only on the training data.

## 4. Reproducibility checks

My dataset manifest records these exact inputs:

| File | Records | Size in bytes | SHA-256 |
|---|---:|---:|---|
| `train.jsonl` | 40,000 | 58,052,946 | `d1aac2d0b524edda20cd8acf971a71d771c863523396d4b8c8db59ac6a985d29` |
| `validation.jsonl` | 77,187 | 122,569,012 | `33ea85f638606a4780c149bd348dd9256056c7d18feb48ff7b6e978cf8f9a138` |

I used `classification/verify_ec2_data.py` to check downloaded files against these values. The downloader uses temporary files and refuses to overwrite existing files.

After training, I preserved the classifier as a joblib pipeline together with `training_metrics.json`, `reproducibility.json`, `validation_2024.json`, and a completion record. The model freeze and final-test result files record this classifier checksum:

```text
60f84cbc1d99a9cff44bf089cec25e0ba4343575cab51ca3a3218777f83533e1
```

My training script creates a run directory named `full_40000_seed42`. The local Streamlit application uses a separately restored copy under `full-40000-seed42`. I need the correct directory name when restoring the app artifacts.

## 5. Verified evaluation results

| Metric | 2024 validation | 2025 to 2026 held-out test |
|---|---:|---:|
| Evaluated papers | 77,187 | 207,257 |
| Accuracy | 80.59% | 74.93% |
| Macro-F1 | 75.64% | 72.63% |
| Balanced accuracy | 79.94% | 76.59% |

I recorded the final model configuration and SHA-256 before evaluating the held-out test set. I did not tune or replace the model after viewing those results.

My committed evidence is in:

- `classification/results/model_freeze.json`
- `classification/results/final_test_2025_2026.json`

The final evaluation is designed to run once. **Do not rerun `evaluate_test.py --run-final-test` as a routine verification command.** It rejects an existing final result or attempt file to protect the evaluation record.

## 6. Safe local checks

These commands do not create AWS resources:

```bash
python -m classification.verify_ec2_data --data-dir classification/data
python -m classification.train --dry-run
```

They require the prepared local files. I should run them only after confirming the paths and environment, and I should not overwrite or regenerate existing datasets during this documentation review.

The application service tests can be run separately:

```bash
python -m unittest discover -s tests -p 'test_app_services.py' -v
```

Those tests require the restored local model and FAISS artifacts. They do not retrain the classifier or contact AWS.

## 7. AWS cost and permissions

The S3 download and upload scripts make real AWS calls when executed. Full training on an EC2 instance also incurs cloud usage. I do not run these operations as part of documentation work.

My code requires explicit confirmation for a full training run, but the confirmation string in the source is an execution safeguard, **not an AWS security credential**. IAM permissions still determine which AWS operations an instance can perform.

Before any new AWS execution, I would check the project budget, instance configuration, IAM permissions, data checksums, and existing artifacts. I would also plan resource cleanup and verify costs afterward.

## 8. Limitations

- My labels come from the first listed arXiv category.
- My duplicate checks do not identify paraphrased abstracts.
- I trained on balanced categories but evaluated on naturally imbalanced recent papers.
- Chronological shifts in the research literature may affect performance.
- Model binaries and the large datasets are intentionally excluded from GitHub.
- This runbook documents the completed workflow, but a fresh clone still requires access to the saved data and model artifacts.

I kept the source, integrity checks, and measured results so I can explain and verify the work without spending money rerunning the cloud experiment.
