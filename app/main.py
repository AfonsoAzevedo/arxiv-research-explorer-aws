"""
Streamlit application for the DS483 AI Research Paper Explorer.

The application classifies research abstracts and finds related
arXiv papers using a saved classifier and semantic search.
"""

import streamlit as st

from app.classifier_service import load_classifier, classify_abstract
from app.search_service import load_search_service, find_related_papers
from semantic_search.query_service import QueryNotCached


# Configure the application page.
st.set_page_config(
    page_title="AI Research Paper Explorer",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource
def get_classifier():
    """Load the trained classification model once."""
    return load_classifier()


@st.cache_resource
def get_search_service():
    """Load the existing FAISS index once."""
    return load_search_service()


# Display the application title and description.
st.title("AI Research Paper Explorer")

st.caption(
    "DS483 Cloud Computing | AWS-powered research discovery"
)

st.write(
    "Classify research abstracts using machine learning "
    "or discover related scientific papers using semantic search."
)

# Create a tab for each application feature.
tab_classify, tab_search = st.tabs(
    ["Classify Abstract", "Discover Papers"]
)

with tab_classify:
    st.subheader("Research Paper Classification")

    # Accept a research abstract from the user.
    abstract = st.text_area(
        "Paste a research abstract",
        height=200,
        placeholder="Enter a scientific abstract...",
        key="classification_abstract",
    )

    if st.button("Classify Abstract", type="primary"):
        if not abstract.strip():
            st.warning("Please enter a research abstract.")
        else:
            try:
                # Run the saved classifier on the submitted abstract.
                model = get_classifier()
                result = classify_abstract(model, abstract)

                st.success(
                    f'Predicted category: {result["category_name"]} '
                    f'({result["category_code"]})'
                )

                # Display the three highest-scoring categories.
                st.subheader("Top Category Predictions")

                for category in result["top_categories"]:
                    st.write(
                        f'{category["name"]} ({category["code"]}): '
                        f'{category["probability"]:.2%}'
                    )

                st.caption(
                    "These are uncalibrated model probabilities, "
                    "not guaranteed prediction confidence."
                )

            except (FileNotFoundError, ValueError) as exc:
                st.error(f"Classification unavailable: {exc}")

            except Exception:
                st.error(
                    "An unexpected classification error occurred. "
                    "Check the application terminal for details."
                )

with tab_search:
    st.subheader("Semantic Paper Discovery")

    # Accept a research topic from the user.
    query = st.text_area(
        "Describe your research topic",
        height=140,
        placeholder=(
            "Example: Deep learning methods for detecting "
            "cybersecurity threats and network intrusions"
        ),
        key="search_query",
    )

    st.caption(
        "Offline demonstration mode: only previously cached "
        "queries are supported. No new Bedrock requests are made."
    )

    # Search only when the user submits a nonempty query.
    if st.button("Find Related Papers"):
        if not query.strip():
            st.warning("Please enter a research topic.")
        else:
            try:
                # Search the 368-paper FAISS index using a cached embedding.
                service = get_search_service()
                results = find_related_papers(
                    service,
                    query,
                    top_k=5,
                )

                st.success(
                    f"Found {len(results)} related papers "
                    "in the 368-paper search index."
                )

                # Display each paper with its metadata and arXiv link.
                for number, paper in enumerate(results, start=1):
                    st.subheader(f"{number}. {paper['title']}")

                    st.write(
                        f"Category: {paper['category']} | "
                        f"Cosine similarity: {paper['score']:.4f}"
                    )

                    st.write(
                        paper["abstract"][:400] + (
                            "..."
                            if len(paper["abstract"]) > 400
                            else ""
                        )
                    )

                    # Construct the arXiv link from the validated paper ID.
                    arxiv_url = (
                        f"https://arxiv.org/abs/{paper['paper_id']}"
                    )

                    st.link_button(
                        "View on arXiv",
                        arxiv_url,
                        key=f"paper_{number}_{paper['paper_id']}",
                    )

                    st.divider()

            except QueryNotCached:
                st.warning(
                    "This query does not have a cached embedding. "
                    "New Bedrock requests are disabled in the "
                    "offline demonstration."
                )

            except (FileNotFoundError, ValueError) as exc:
                st.error(f"Semantic search unavailable: {exc}")

            except Exception:
                st.error(
                    "An unexpected search error occurred. "
                    "Check the application terminal for details."
                )

# Identify the technologies used by the application.
st.divider()

st.caption(
    "Classification: TF-IDF + Logistic Regression | "
    "Retrieval: Amazon Titan V2 embeddings + FAISS"
)
