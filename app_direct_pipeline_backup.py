import os
from pathlib import Path
import streamlit as st

from src.pipeline.root_cause_pipeline import RootCausePipeline


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="RootCause AI",
    page_icon="🔍",
    layout="wide"
)


# =====================================================
# PIPELINE CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"
VECTORSTORE_DIRECTORY = PROJECT_ROOT / "vectorstore" / "streamlit_app"

# Streamlit Cloud: read the key from Secrets.
# Local execution: RootCauseAnalyzer can load it from the local .env file.
try:
    secret_api_key = st.secrets.get("OPENAI_API_KEY")
    if secret_api_key:
        os.environ["OPENAI_API_KEY"] = secret_api_key
except Exception:
    pass


@st.cache_resource(show_spinner="Initializing RootCause AI pipeline...")
def get_pipeline():
    """Initialize the Hybrid RAG pipeline once per Streamlit process."""
    return RootCausePipeline(
        data_directory=str(DATA_DIRECTORY),
        persist_directory=str(VECTORSTORE_DIRECTORY),
        model_name="gpt-4o-mini"
    )


# =====================================================
# CUSTOM STYLING
# =====================================================

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 700;
    color: #4F8BF9;
}

.subtitle {
    font-size: 17px;
    color: gray;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="main-title">🔍 RootCause AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Hybrid RAG-Based Software Incident Investigation System'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("⚙️ Investigation Settings")

    top_k = st.slider(
        "Final Results (Top-K)",
        1, 10, 3
    )

    candidate_k = st.slider(
        "Candidate Retrieval Count",
        1, 20, 5
    )

    st.divider()

    st.subheader("System Architecture")

    st.write("**Retrieval:** BM25 + Vector Search")
    st.write("**Fusion:** Reciprocal Rank Fusion")
    st.write("**LLM:** GPT-4o-mini")
    st.write("**Execution:** In-process pipeline")

    # Pipeline status
    st.success("Streamlit pipeline mode")
    st.caption("Hybrid RAG runs directly inside this application.")

# =====================================================
# INCIDENT INVESTIGATION
# =====================================================

st.header("🚨 Incident Investigation")

st.write(
    "Describe the software incident you want to investigate."
)

incident_query = st.text_area(
    "Incident Description",
    placeholder=(
        "Example: Payment Service started returning HTTP 500 "
        "errors after the latest deployment. Investigate "
        "possible database connection failures."
    ),
    height=150
)


# =====================================================
# INVESTIGATION BUTTON
# =====================================================

if st.button(
    "🔍 Investigate Incident",
    type="primary",
    use_container_width=True
):

    if not incident_query.strip():

        st.warning("Please enter an incident description.")

    elif candidate_k < top_k:

        st.warning(
            "Candidate Retrieval Count must be greater "
            "than or equal to Final Results."
        )

    else:

        try:
            with st.spinner(
                "Investigating incident using Hybrid RAG..."
            ):
                pipeline = get_pipeline()
                result = pipeline.run(
                    query=incident_query,
                    top_k=top_k,
                    candidate_k=candidate_k
                )

            st.session_state["investigation_result"] = result

            if result.get("status") == "completed":
                st.success("Investigation completed successfully!")
            elif result.get("status") == "abstained":
                st.warning(
                    "Investigation abstained. "
                    "The system did not proceed with root-cause analysis."
                )
            else:
                st.info("Investigation response received.")

        except ValueError as error:
            st.error(f"Configuration or input error: {error}")
        except Exception as error:
            st.error(
                "Investigation failed. Check the application logs "
                f"for details. Error: {error}"
            )


# =====================================================
# INVESTIGATION RESULTS
# =====================================================

if "investigation_result" in st.session_state:

    result = st.session_state["investigation_result"]

    st.divider()

    st.header("📊 Investigation Results")

    # Retrieve response fields safely

    status = result.get("status", "unknown")

    retrieved_count = result.get("retrieved_count", 0)

    evidence_list = result.get("retrieved_evidence", [])

    abstention = result.get("abstention", {})

    # -----------------------------------------------
    # STATUS DISPLAY
    # -----------------------------------------------

    if status == "completed":

        status_label = "🟢 Completed"

    elif status == "abstained":

        status_label = "🟠 Abstained"

    else:

        status_label = "⚪ Unknown"

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Retrieved Evidence",
            retrieved_count
        )

    with col2:

        st.metric(
            "Investigation Status",
            status_label
        )

    # -----------------------------------------------
    # ABSTENTION INFORMATION
    # -----------------------------------------------

    if status == "abstained":

        st.warning("### ⚠️ Investigation Abstained")

        st.write(
            "**Reason:**",
            abstention.get("reason", "Not specified")
        )

        st.write(
            "**Category:**",
            abstention.get("category", "Not specified")
        )

        st.write(
            "**Explanation:**",
            abstention.get(
                "message",
                "The system could not proceed with the investigation."
            )
        )

        max_score = abstention.get("max_vector_score")

        if max_score is not None:

            st.write(
                f"**Maximum Vector Similarity Score:** "
                f"{max_score:.4f}"
            )

        threshold = abstention.get("threshold")

        if threshold is not None:

            st.write(
                f"**Configured Threshold:** {threshold:.2f}"
            )

    # -----------------------------------------------
    # ROOT CAUSE ANALYSIS
    # -----------------------------------------------

    st.subheader("🧠 Root Cause Analysis")

    st.markdown(
        result.get(
            "analysis",
            "No analysis was returned."
        )
    )

    # -----------------------------------------------
    # RETRIEVED EVIDENCE
    # -----------------------------------------------

    st.divider()

    st.subheader("📚 Retrieved Historical Evidence")

    if evidence_list:

        for index, evidence in enumerate(
            evidence_list,
            start=1
        ):

            incident_id = evidence.get(
                "incident_id",
                "Unknown"
            )

            with st.expander(
                f"Evidence {index} — {incident_id}",
                expanded=(index == 1)
            ):

                st.write(
                    "**Incident ID:**",
                    incident_id
                )

                st.write(
                    "**Vector Score:**",
                    evidence.get("vector_score", "N/A")
                )

                st.write(
                    "**BM25 Score:**",
                    evidence.get("bm25_score", "N/A")
                )

                rrf_score = evidence.get("rrf_score")

                if rrf_score is not None:

                    st.write(
                        f"**RRF Score:** {rrf_score:.6f}"
                    )

                st.write("**Retrieval Rankings:**")

                st.json(
                    evidence.get("retrieval_ranks", {})
                )

                st.write("**Historical Incident Details:**")

                st.text(
                    evidence.get("content", "")
                )

    else:

        st.info(
            "No historical incident evidence was retrieved."
        )

    # -----------------------------------------------
    # FOOTNOTE
    # -----------------------------------------------

    st.caption(
        "Findings are evidence-based hypotheses. "
        "Verify them against current application logs and metrics."
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "RootCause AI | Hybrid RAG Software Incident Investigation System"
)