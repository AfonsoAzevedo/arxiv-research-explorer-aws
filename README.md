# AI Research Paper Discovery and Classification on AWS

An end-to-end cloud computing and machine learning project for classifying scientific research abstracts and discovering related arXiv papers through semantic similarity.

**Course:** DS483 Cloud Computing  
**Institution:** Embry-Riddle Aeronautical University  
**Developer:** Afonso Azevedo  
**AWS Region:** us-east-1
**Status:** Working local application; additional evaluation and data engineering in progress

## Project Overview

This project combines supervised machine learning, vector similarity search, and Amazon Web Services to help users explore scientific literature.

The application provides two capabilities:

1. **Research Paper Classification:** Predict one of eight computer science research categories from a user-provided abstract using TF-IDF and Logistic Regression.
2. **Semantic Paper Discovery:** Retrieve related arXiv papers using Amazon Titan Text Embeddings V2 and FAISS.

A local Streamlit application integrates both capabilities. The current semantic search demonstration uses previously cached query embeddings and does not make new Amazon Bedrock requests.

## Application Features

The working Streamlit application supports:

- Abstract classification with the three highest-ranked predicted categories
- Semantic retrieval of five related papers from a 368-paper FAISS index
- Paper titles, abstracts, categories, similarity scores, and arXiv links
- Input validation and error handling
- Cached-only semantic queries to avoid unapproved AWS usage

The classification model produces probability estimates, but these probabilities have not been independently calibrated.

**Important:** The semantic search interface currently supports only previously cached queries. Unrestricted live semantic search is not enabled in the application.

## System Architecture

The implemented components follow this workflow:

### Classification Pipeline

```text
arXiv Metadata Dataset
        |
        v
Data Validation and Preparation
        |
        v
40,000 Labeled Training Abstracts
        |
        v
Amazon EC2
TF-IDF + Logistic Regression
        |
        v
Amazon S3 Model Artifacts
        |
        v
Local Production Classifier
        |
        v
Streamlit Category Predictions
```

### Semantic Search Pipeline

```text
Selected arXiv Papers
        |
        v
Amazon Bedrock
Titan Text Embeddings V2
        |
        v
512-Dimensional Embeddings
        |
        v
Local FAISS Index
368 Papers
        |
        v
Cached Query Embedding
        |
        v
Similarity Search
        |
        v
Streamlit Related Papers
```

Amazon S3 stores verified project artifacts. Amazon Athena and AWS Glue Data Catalog were also tested with a small sample, but the complete 3.23-million-paper data engineering pipeline has not yet been implemented.

The application does not execute Athena queries or invoke Bedrock when users submit cached semantic search queries.

## Technology Stack

| Component | Technology |
|---|---|
| Cloud infrastructure | Amazon S3, EC2, IAM |
| Cloud data engineering | Amazon Athena, AWS Glue Data Catalog |
| Embedding generation | Amazon Bedrock Titan Text Embeddings V2 |
| Programming | Python, SQL, Boto3 |
| Classification | scikit-learn, TF-IDF, Logistic Regression |
| Vector retrieval | FAISS IndexFlatIP |
| Application | Streamlit |
| Verified local Python version | 3.14.4 |

FAISS is a local vector-search library, not an AWS managed service.

## Dataset

**Source:** [Cornell University arXiv Metadata on Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)

The local metadata snapshot contains:

- 3,233,620 validated paper records
- Approximately 5.28 GiB of JSONL metadata
- 176 research category labels

The complete metadata dataset is not included in this repository.

### Classification Categories

The production classifier predicts one of eight research categories:

| Category | Description |
|---|---|
| `cs.AI` | Artificial Intelligence |
| `cs.CL` | Computation and Language |
| `cs.CR` | Cryptography and Security |
| `cs.CV` | Computer Vision |
| `cs.DB` | Databases |
| `cs.LG` | Machine Learning |
| `cs.NI` | Networking and Internet Architecture |
| `cs.RO` | Robotics |

### Classification Dataset Splits

The classifier uses a selected subset of the arXiv metadata dataset.

| Dataset | Samples | Purpose |
|---|---:|---|
| Training | 40,000 | Fit the classifier |
| 2024 validation | 77,187 | Evaluate model performance |
| 2025–2026 held-out test | Not yet evaluated | Final independent evaluation |

The training sample contains 5,000 examples per category. The model was not trained on all 3.23 million records.

## Classification Model

The production model was trained on Amazon EC2 using:

- TF-IDF text features
- Unigram and bigram representations
- Up to 100,000 vocabulary features
- Logistic Regression
- Random seed 42 for reproducible training sampling

The saved model and its evaluation artifacts were stored in Amazon S3. The locally restored production artifacts were verified against their recorded checksums.

### Verified Validation Results

Evaluation on 77,187 abstracts from the 2024 validation partition:

| Metric | Result |
|---|---:|
| Accuracy | 80.59% |
| Macro-F1 | 75.64% |
| Training examples | 40,000 |
| Validation examples | 77,187 |
| Categories | 8 |

**These are 2024 validation results, not final held-out test results.**

The 2025–2026 test partition has not yet been evaluated. These metrics should not be interpreted as performance on unseen future research papers without that final evaluation.

## Semantic Search

Semantic retrieval uses Amazon Titan Text Embeddings V2 to represent abstracts as 512-dimensional vectors.

The current application uses:

- Normalized embedding vectors
- FAISS IndexFlatIP for exact inner-product similarity search
- A local search index containing 368 real arXiv papers
- Previously cached query embeddings
- The top five retrieved papers for a supported query

The 368-paper index is a demonstration corpus, not an index of the full arXiv dataset.

### Exploratory TF-IDF Comparison

An earlier retrieval pilot compared semantic search with TF-IDF retrieval using a 312-paper corpus.

| Metric | Semantic Search | TF-IDF |
|---|---:|---:|
| Broad Precision@3 | 44.44% | 22.22% |
| Strict Precision@3 | 0.00% | 0.00% |

This exploratory evaluation used only three selected queries. Relevance judgments were proposed by an assistant and were not independently adjudicated.

**These results do not establish that semantic retrieval is generally superior to TF-IDF.** They also do not represent a comprehensive evaluation of the current 368-paper application index.

## Local Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AfonsoAzevedo/arxiv-research-explorer-aws.git
cd arxiv-research-explorer-aws
```

### 2. Create a Python Environment

The verified development environment uses Python 3.14.4.

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Application Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: -r requirements-app.txt
```

The dependency versions match the environment used for the successful local application tests.

Installation from a completely fresh environment has not yet been independently verified.

### 4. Restore Required Artifacts

The application expects these paths to exist:

```text
classification/models/full-40000-seed42/classifier.joblib

semantic_search/data/faiss_368/index.faiss
semantic_search/data/faiss_368/paper_ids.json
semantic_search/data/retrieval_25k.jsonl
semantic_search/data/application_query_cache/
```

The production classifier, FAISS index, paper metadata, and cached query embeddings are excluded from Git version control.

A fresh clone therefore does **not** currently contain all the files needed for the full demonstration.

These artifacts must be restored from the project's previously generated and verified local or AWS artifacts. Public artifact distribution and complete regeneration instructions are not yet available.

### 5. Run the Application

From the repository root:

```bash
python -m streamlit run app/main.py --server.address 127.0.0.1 --server.port 8501
```

Open:

http://127.0.0.1:8501

The classification interface accepts new abstracts for local inference.

For the current cached Semantic Search demonstration, use this exact query:

```text
Deep learning methods for detecting cybersecurity threats and network intrusions
```

A different query without a cached embedding will be rejected without initiating an AWS Bedrock request.

## Automated Tests

The project includes five automated service tests for:

- Classification inference
- Empty classification input
- Cached semantic retrieval
- Uncached-query rejection
- Empty semantic search input

Run the tests from the repository root:

```bash
python -m unittest tests/test_app_services.py -v
```

**Verified on November 4, 2026:** All five tests passed.

These tests require the locally restored model and retrieval artifacts. They are service-level tests and do not constitute a complete automated browser-testing suite.

## Current Limitations

- The full 3.23-million-paper S3/Athena/Parquet pipeline is not yet complete.
- The current FAISS index contains 368 papers.
- The local application supports cached semantic search queries only.
- Retrieval evaluation remains preliminary and limited in scope.
- Final classification performance on the 2025–2026 held-out test data has not been measured.
- A new GitHub clone requires additional model and retrieval artifacts.
- The application is verified locally but has not been publicly deployed.
- Comprehensive AWS cost and application latency measurements are not yet documented.

## Next Steps

1. Complete the S3/Athena data engineering workflow.
2. Expand and evaluate semantic search using a more representative retrieval corpus.
3. Perform the final classification evaluation on the untouched held-out test dataset.
4. Improve application reproducibility and artifact restoration instructions.
5. Prepare the final project report, presentation, and demonstration.

## Academic Context

This project is being developed individually for DS483 Cloud Computing at Embry-Riddle Aeronautical University.

It demonstrates the integration of cloud services, supervised text classification, semantic vector retrieval, and a Python web application, while distinguishing verified experimental results from planned improvements.
