import os
import requests
import streamlit as st


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="RootCause AI",
    page_icon="🔍",
    layout="wide"
)


# =====================================================
# BACKEND CONFIGURATION
# =====================================================

# Local default: http://127.0.0.1:8001
# Streamlit Community Cloud: set API_URL in App Settings > Secrets.
try:
    configured_api_url = st.secrets.get("API_URL", "")
except Exception:
    configured_api_url = ""

API_URL = (
    configured_api_url
    or os.getenv("API_URL")
    or "http://127.0.0.1:8001"
).rstrip("/")

HEALTH_URL = f"{API_URL}/health"
INVESTIGATE_URL = f"{API_URL}/investigate"

CONNECT_TIMEOUT = 10
INVESTIGATION_TIMEOUT = 240


def check_backend():
    """Return (is_healthy, message) for the configured FastAPI backend."""
    try:
        response = requests.get(
            HEALTH_URL,
            timeout=CONNECT_TIMEOUT
        )
        response.raise_for_status()
        health_data = response.json()

        if health_data.get("status") == "healthy":
            return True, health_data

        return False, health_data

    except requests.exceptions.Timeout:
        return False, "The backend health check timed out."
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to the FastAPI backend."
    except requests.exceptions.RequestException as error:
        return False, f"Backend health check failed: {error}"
    except ValueError:
        return False, "The backend returned an invalid health response."


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
    st.write("**Execution:** FastAPI backend")

    st.caption(f"Backend URL: `{API_URL}`")

    backend_healthy, backend_info = check_backend()

    if backend_healthy:
        st.success("FastAPI backend connected")
        st.caption(
            "Pipeline initialized: "
            f"{backend_info.get('pipeline_initialized', False)}"
        )
    else:
        st.error("FastAPI backend unavailable")
        st.caption(str(backend_info))


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

    elif len(incident_query.strip()) < 10:
        st.warning("Please enter at least 10 characters.")

    elif candidate_k < top_k:
        st.warning(
            "Candidate Retrieval Count must be greater "
            "than or equal to Final Results."
        )

    else:
        payload = {
            "query": incident_query.strip(),
            "top_k": top_k,
            "candidate_k": candidate_k
        }

        try:
            with st.spinner(
                "Sending incident to FastAPI and running Hybrid RAG..."
            ):
                response = requests.post(
                    INVESTIGATE_URL,
                    json=payload,
                    timeout=INVESTIGATION_TIMEOUT
                )

                response.raise_for_status()
                api_result = response.json()

            # FastAPI wraps the pipeline result inside "investigation".
            result = api_result.get("investigation")

            if not isinstance(result, dict):
                raise ValueError(
                    "The backend response does not contain a valid "
                    "'investigation' object."
                )

            st.session_state["investigation_result"] = result

            if result.get("status") == "completed":
                st.success("Investigation completed successfully!")
            elif result.get("status") == "abstained":
                st.warning(
                    "Investigation abstained. The system did not "
                    "proceed with root-cause analysis."
                )
            else:
                st.info("Investigation response received.")

        except requests.exceptions.Timeout:
            st.error(
                "The investigation timed out. The backend may still "
                "be processing the request. Check the FastAPI logs."
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to the FastAPI backend. Confirm that "
                "the backend is running and API_URL is correct."
            )

        except requests.exceptions.HTTPError as error:
            detail = ""
            try:
                detail = response.json().get("detail", "")
            except Exception:
                pass

            st.error(
                f"FastAPI returned an HTTP error "
                f"({response.status_code}): {detail or str(error)}"
            )

        except (ValueError, requests.exceptions.RequestException) as error:
            st.error(f"Investigation request failed: {error}")

        except Exception:
            st.error(
                "An unexpected error occurred. Check the Streamlit "
                "and FastAPI logs for details."
            )


# =====================================================
# INVESTIGATION RESULTS
# =====================================================

if "investigation_result" in st.session_state:
    result = st.session_state["investigation_result"]

    st.divider()
    st.header("📊 Investigation Results")

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
        st.metric("Retrieved Evidence", retrieved_count)

    with col2:
        st.metric("Investigation Status", status_label)

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
                f"**Maximum Vector Similarity Score:** {max_score:.4f}"
            )

        threshold = abstention.get("threshold")
        if threshold is not None:
            st.write(f"**Configured Threshold:** {threshold:.2f}")

    # -----------------------------------------------
    # ROOT CAUSE ANALYSIS
    # -----------------------------------------------

    st.subheader("🧠 Root Cause Analysis")

    st.markdown(
        result.get("analysis", "No analysis was returned.")
    )

    # -----------------------------------------------
    # RETRIEVED EVIDENCE
    # -----------------------------------------------

    st.divider()
    st.subheader("📚 Retrieved Historical Evidence")

    if evidence_list:
        for index, evidence in enumerate(evidence_list, start=1):
            incident_id = evidence.get("incident_id", "Unknown")

            with st.expander(
                f"Evidence {index} — {incident_id}",
                expanded=(index == 1)
            ):
                st.write("**Incident ID:**", incident_id)
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
                    st.write(f"**RRF Score:** {rrf_score:.6f}")

                st.write("**Retrieval Rankings:**")
                st.json(evidence.get("retrieval_ranks", {}))

                st.write("**Historical Incident Details:**")
                st.text(evidence.get("content", ""))

    else:
        st.info("No historical incident evidence was retrieved.")

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