
import streamlit as st
from pathlib import Path
import re
import json
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="MediRAG Healthcare Assistant",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 MediRAG")
st.subheader("Smart Rural Healthcare Assistant")

st.write(
    "Search relevant information from the local healthcare knowledge document."
)

st.info(
    "Educational prototype only. This tool does not diagnose illnesses "
    "or replace a qualified healthcare professional."
)

DATA_FILE = Path(__file__).parent / "healthcare_guide.txt"
ALERTS_FILE = Path(__file__).parent / "alerts.json"

def load_alerts():
    if not ALERTS_FILE.exists():
        return {"alerts": []}

    try:
        with open(ALERTS_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return {"alerts": []}


def save_alerts(data):
    with open(ALERTS_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def load_document():
    if DATA_FILE.exists():
        return DATA_FILE.read_text(encoding="utf-8")
    return ""


def split_into_passages(document):
    # Split the document at topic headings such as FEVER: and HEADACHE:
    parts = re.split(r"(?m)^(?=[A-Z][A-Z ]+:\s*$)", document)
    return [part.strip() for part in parts if part.strip()]


def retrieve_information(question, document):
    passages = split_into_passages(document)

    if not passages:
        return []

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    try:
        passage_vectors = vectorizer.fit_transform(passages)
        question_vector = vectorizer.transform([question])
    except ValueError:
        return []

    scores = cosine_similarity(
        question_vector, passage_vectors
    ).flatten()

    ranked_indices = scores.argsort()[::-1]

    results = []
    for index in ranked_indices[:3]:
        if scores[index] > 0:
            results.append((passages[index], float(scores[index])))

    return results


name = st.text_input("Enter your name", key="patient_name")

question = st.text_area(
    "Describe your health question",
    placeholder="Example: What information is available about headaches?",
    key="health_question"
)

if st.button("Search Healthcare Information"):
    if not name.strip() or not question.strip():
        st.warning("Please enter your name and question.")
    else:
        document = load_document()

        if not document:
            st.error(
                "Healthcare document not found. Check "
                "data/healthcare_guide.txt."
            )
        else:
            results = retrieve_information(question, document)

            st.subheader("Retrieved Healthcare Information")

            if results:
                for passage, score in results:
                    st.markdown(passage)
                    st.caption(f"Search relevance score: {score:.2f}")
                    st.divider()

                st.caption(
                    "Passages are ranked by text similarity. "
                    "Scores are not medical confidence levels."
                )
            else:
                st.warning(
                    "No relevant passage was found in the current document. "
                    "Please consult a qualified healthcare professional."
                )
st.divider()

st.header("🔔 Health Worker Notification & Alert System")

st.info(
    "Demo alert system for health-worker follow-up. "
    "It does not send real notifications."
)

with st.form("alert_form"):
    patient_name = st.text_input("Demo patient name or ID")
    symptom = st.text_area("Health concern")
    priority = st.selectbox(
        "Follow-up priority",
        ["Routine follow-up", "Urgent review"]
    )

    submit_alert = st.form_submit_button("Create Alert")

    if submit_alert:
        if not patient_name.strip() or not symptom.strip():
            st.warning("Please enter the patient ID and health concern.")
        else:
            data = load_alerts()

            new_alert = {
                "patient": patient_name.strip(),
                "concern": symptom.strip(),
                "priority": priority,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "Pending"
            }

            data["alerts"].append(new_alert)
            save_alerts(data)

            st.success("Demo alert saved successfully!")


st.divider()

st.subheader("📋 Saved Health Alerts")

saved_data = load_alerts()
saved_alerts = saved_data.get("alerts", [])

if saved_alerts:
    for alert in reversed(saved_alerts):
        st.markdown(f"**Patient ID:** {alert['patient']}")
        st.write(f"**Health concern:** {alert['concern']}")
        st.write(f"**Priority:** {alert['priority']}")
        st.write(f"**Status:** {alert['status']}")
        st.caption(f"Created at: {alert['created_at']}")
        st.divider()
else:
    st.info("No alerts have been saved yet.")

st.divider()

st.divider()

st.subheader("📊 Alert Dashboard")

dashboard_data = load_alerts()
dashboard_alerts = dashboard_data.get("alerts", [])

pending_count = sum(
    1 for alert in dashboard_alerts
    if alert.get("status", "Pending") == "Pending"
)

completed_count = sum(
    1 for alert in dashboard_alerts
    if alert.get("status", "Pending") == "Completed"
)

col1, col2 = st.columns(2)

with col1:
    st.metric("Pending Alerts", pending_count)

with col2:
    st.metric("Completed Alerts", completed_count)

st.subheader("📋 Saved Health Alerts")

saved_data = load_alerts()
saved_alerts = saved_data.get("alerts", [])

if saved_alerts:
    for index in range(len(saved_alerts) - 1, -1, -1):
        alert = saved_alerts[index]

        st.markdown(f"**Patient ID:** {alert['patient']}")
        st.write(f"**Health concern:** {alert['concern']}")
        st.write(f"**Priority:** {alert['priority']}")
        st.write(f"**Status:** {alert.get('status', 'Pending')}")
        st.caption(f"Created at: {alert.get('created_at', 'Not available')}")

        if alert.get("status", "Pending") == "Pending":
            if st.button("Mark as Completed", key=f"complete_{index}"):
                saved_alerts[index]["status"] = "Completed"
                save_alerts({"alerts": saved_alerts})
                st.success("Alert marked as completed!")
                st.rerun()

        st.divider()
else:
    st.info("No alerts have been saved yet.")
