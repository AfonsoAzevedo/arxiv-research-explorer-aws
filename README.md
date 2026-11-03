# AI Research Paper Discovery and Classification on AWS

An end-to-end cloud computing and machine learning project for classifying scientific research papers and retrieving semantically related research.

**Course:** DS483 Cloud Computing  
**Institution:** Embry-Riddle Aeronautical University  
**Developer:** Afonso Azevedo  
**Status:** In development

## Project Overview

This project uses the arXiv metadata dataset containing 3,233,620 research papers to develop two main capabilities:

1. **Research Paper Classification:** Train a TF-IDF + Logistic Regression model to predict research categories from paper abstracts.
2. **Semantic Search:** Use Amazon Bedrock Titan embeddings and FAISS to retrieve conceptually related papers.

The system will also compare traditional TF-IDF-based retrieval against embedding-based semantic retrieval.

## Technology Stack

- **Cloud:** Amazon S3, Athena, AWS Glue Data Catalog, EC2, IAM, Amazon Bedrock
- **Programming:** Python, SQL, Boto3
- **Machine Learning:** scikit-learn, TF-IDF, Logistic Regression
- **Semantic Search:** Amazon Titan Text Embeddings V2, FAISS
- **Application:** Streamlit

## Dataset

Source: [Cornell University arXiv Metadata on Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)

- 3,233,620 validated paper records
- Approximately 5.28 GiB of JSONL metadata
- 176 research category labels

The full dataset is not included in this repository.

## Architecture

The planned data flow is:

1. Store raw arXiv metadata in Amazon S3.
2. Use Athena and SQL to process data into Parquet.
3. Train a research category classifier using Python on EC2.
4. Generate semantic embeddings using Amazon Bedrock.
5. Build a FAISS similarity search index.
6. Integrate classification and retrieval into a Streamlit application.

## Current Status

The project proposal is complete. Initial feasibility tests for S3, Athena, EC2, Bedrock, and FAISS have succeeded.

The complete application and experimental evaluations are still under development.

## Future Documentation

As the project progresses, this repository will include:

- Architecture diagrams
- Reproducible Python scripts and SQL queries
- Classification and retrieval evaluation results
- Application screenshots
- Setup instructions
- AWS cost and performance measurements
