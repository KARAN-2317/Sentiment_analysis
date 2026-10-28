import html
import json
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "evaluation.json"
TRAIN_DATA_PATH = BASE_DIR / "data" / "raw" / "train_df.csv"

st.set_page_config(
    page_title="SentimentX AI",
    page_icon="💬",
    layout="wide",
)

SENTIMENT_COLORS = {
    "Positive": "#16A34A",
    "Neutral": "#64748B",
    "Negative": "#DC2626",
}

SENTIMENT_ORDER = ["Negative", "Neutral", "Positive"]


# =========================================================
# 2. LOAD MODEL AND METRICS
# =========================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run: python src/train_model.py"
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return {}

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


@st.cache_data
def load_training_distribution():
    if not TRAIN_DATA_PATH.exists():
        return pd.DataFrame({
            "Sentiment": SENTIMENT_ORDER,
            "Examples": [0, 0, 0],
        })

    df = pd.read_csv(TRAIN_DATA_PATH, usecols=["sentiment"])
    counts = (
        df["sentiment"].astype(str).str.lower().value_counts()
    )

    return pd.DataFrame({
        "Sentiment": SENTIMENT_ORDER,
        "Examples": [
            int(counts.get("negative", 0)),
            int(counts.get("neutral", 0)),
            int(counts.get("positive", 0)),
        ],
    })


# =========================================================
# 3. TEXT ANALYSIS
# =========================================================

def analyze_texts(texts, model):
    cleaned = [
        re.sub(r"\s+", " ", str(text)).strip()
        for text in texts
    ]

    results = [
        {
            "Text": str(original),
            "Sentiment": "Not analyzed",
            "Confidence (%)": None,
        }
        for original in texts
    ]

    valid_indices = [
        index for index, text in enumerate(cleaned) if text
    ]

    if not valid_indices:
        return pd.DataFrame(results)

    valid_texts = [cleaned[index] for index in valid_indices]

    predictions = model.predict(valid_texts)
    probabilities = model.predict_proba(valid_texts)

    for index, prediction, probability in zip(
        valid_indices, predictions, probabilities
    ):
        results[index]["Sentiment"] = str(prediction).capitalize()
        results[index]["Confidence (%)"] = round(
            float(max(probability)) * 100, 2
        )

    return pd.DataFrame(results)


def save_to_history(results, source="Text Analyzer"):
    """Save analyzed rows to the current Streamlit session."""
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for _, row in results.iterrows():
        if row["Sentiment"] == "Not analyzed":
            continue

        st.session_state.analysis_history.append({
            "Time": timestamp,
            "Source": source,
            "Text": str(row["Text"]),
            "Sentiment": row["Sentiment"],
            "Confidence (%)": row["Confidence (%)"],
        })


# =========================================================
# 4. PDF REPORT GENERATOR
# =========================================================

def create_pdf_report(
    results,
    title="SentimentX AI Analysis Report",
    source="SentimentX AI",
):
    """Create a downloadable PDF report from analysis results."""
    buffer = BytesIO()

    # Landscape layout gives the text column more room.
    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=30,
        leftMargin=30,
        topMargin=35,
        bottomMargin=35,
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(html.escape(title), styles["Title"]))
    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "Generated: "
            + datetime.now().strftime("%d %b %Y, %I:%M %p"),
            styles["Normal"],
        )
    )
    story.append(Paragraph(f"Report type: {html.escape(source)}", styles["Normal"]))
    story.append(Spacer(1, 14))

    total = len(results)
    valid = results[results["Sentiment"] != "Not analyzed"]

    story.append(
        Paragraph(
            f"Total entries: {total} | Successfully analyzed: {len(valid)}",
            styles["Heading2"],
        )
    )
    story.append(Spacer(1, 8))

    if not valid.empty:
        counts = valid["Sentiment"].value_counts()

        summary_data = [["Sentiment", "Number of entries"]]

        for sentiment in SENTIMENT_ORDER:
            summary_data.append([
                sentiment,
                str(int(counts.get(sentiment, 0))),
            ])

        summary_table = Table(
            summary_data,
            colWidths=[160, 150],
            repeatRows=1,
        )

        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))

        story.append(summary_table)
        story.append(Spacer(1, 18))

    story.append(Paragraph("Analysis Details", styles["Heading2"]))
    story.append(Spacer(1, 8))

    body_style = getSampleStyleSheet()["BodyText"]
    body_style.fontSize = 8
    body_style.leading = 10

    table_data = [[
        "Text",
        "Predicted sentiment",
        "Confidence",
    ]]

    for _, row in results.iterrows():
        text = re.sub(r"\s+", " ", str(row["Text"])).strip()

        confidence = row["Confidence (%)"]
        confidence_text = (
            f"{float(confidence):.2f}%"
            if pd.notna(confidence)
            else "N/A"
        )

        table_data.append([
            Paragraph(html.escape(text), body_style),
            Paragraph(html.escape(str(row["Sentiment"])), body_style),
            confidence_text,
        ])

    details_table = Table(
        table_data,
        colWidths=[480, 140, 100],
        repeatRows=1,
        hAlign="LEFT",
    )

    details_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))

    story.append(details_table)
    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "Important: confidence is the model's estimated probability "
            "for its predicted class. It is not a guarantee of correctness. "
            "Results may vary on different text sources.",
            styles["Italic"],
        )
    )

    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# =========================================================
# 5. INITIALIZE SESSION STATE AND LOAD MODEL
# =========================================================

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

try:
    model = load_model()
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(f"Unable to load the trained model: {error}")
    st.info(
        "Verify that models/sentiment_model.joblib exists. "
        "If it does not, run python src/train_model.py."
    )
    st.stop()

metrics = load_metrics()
test_metrics = metrics.get("test", {})
training_records = metrics.get("training_records", 0)
test_accuracy = test_metrics.get("accuracy")
macro_f1 = test_metrics.get("macro_f1")


# =========================================================
# 6. HEADER AND NAVIGATION — STYLISH SIDEBAR
# =========================================================

st.markdown(
    """
    <style>
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #0f172a 0%,
            #172554 55%,
            #111827 100%
        );
        border-right: 1px solid rgba(148, 163, 184, 0.20);
    }

    /* Sidebar content spacing */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #e2e8f0;
    }

    /* Sidebar brand card */
    .sx-brand {
        padding: 18px 15px;
        margin-bottom: 22px;
        border-radius: 16px;
        background: linear-gradient(
            135deg,
            rgba(59, 130, 246, 0.24),
            rgba(99, 102, 241, 0.10)
        );
        border: 1px solid rgba(147, 197, 253, 0.24);
    }

    .sx-brand-title {
        font-size: 25px;
        font-weight: 800;
        letter-spacing: -0.7px;
        color: #ffffff;
        margin: 0;
    }

    .sx-brand-accent {
        color: #60a5fa;
    }

    .sx-brand-subtitle {
        font-size: 12px;
        line-height: 1.6;
        color: #cbd5e1;
        margin-top: 6px;
    }

    .sx-section-label {
        font-size: 10px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #93a4bd;
        font-weight: 700;
        margin: 12px 0 8px 3px;
    }

    .sx-footer {
        margin-top: 24px;
        padding: 14px;
        border-radius: 13px;
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid rgba(148, 163, 184, 0.15);
    }

    .sx-footer-name {
        color: #f8fafc;
        font-size: 13px;
        font-weight: 700;
    }

    .sx-footer-role {
        color: #94a3b8;
        font-size: 11px;
        margin-top: 4px;
    }

    /* Navigation radio options */
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 7px;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: 10px 11px;
        border-radius: 10px;
        border: 1px solid transparent;
        transition: background 0.2s ease, border 0.2s ease;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(96, 165, 250, 0.12);
        border-color: rgba(96, 165, 250, 0.24);
    }

    /* Hide the radio circles for a cleaner menu */
    [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {
        display: none;
    }

    /* Keep text readable */
    [data-testid="stSidebar"] [role="radiogroup"] p {
        font-size: 13px;
        font-weight: 500;
        margin: 0;
    }

    /* Sidebar divider */
    [data-testid="stSidebar"] hr {
        border-color: rgba(148, 163, 184, 0.20);
    }

    /* Main page heading */
    .block-container {
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main page title
st.title("💬 SentimentX AI")
st.caption(
    "Sentiment Intelligence Platform | NLP • Machine Learning • Analytics"
)

st.success(
    "Trained TF-IDF + Logistic Regression model loaded successfully."
)

# Sidebar brand
with st.sidebar:
    st.markdown(
        """
        <div class="sx-brand">
            <p class="sx-brand-title">
                Sentiment<span class="sx-brand-accent">X</span> AI
            </p>
            <div class="sx-brand-subtitle">
                ✦ Talent in text. Insights in seconds.<br>
                Sentiment Intelligence Workspace
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sx-section-label">WORKSPACE</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Text Analyzer",
            "Bulk Analyzer",
            "Analysis History",
            "About Project",
            "About Me",
        ],
        label_visibility="collapsed",
        key="main_navigation",
    )

    st.divider()

    st.markdown(
        """
        <div class="sx-section-label">DEVELOPER</div>
        <div class="sx-footer">
            <div class="sx-footer-name">Karan Dev Singh</div>
            <div class="sx-footer-role">
                Student Developer<br>
                SentimentX AI Project
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="
            text-align:center;
            color:#94a3b8;
            font-size:10px;
            padding:14px 0 4px 0;
        ">
            NLP · ML · ANALYTICS<br>
            Built for smarter text insights
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# 7. DASHBOARD
# =========================================================

# =========================================================
# 7. DASHBOARD — PREMIUM UI
# =========================================================

if page == "Dashboard":

    st.markdown("""
    <style>
    .hero-card{
        background:linear-gradient(135deg,#0f172a,#1e3a8a);
        padding:28px;
        border-radius:20px;
        color:white;
        border:1px solid rgba(255,255,255,.12);
        margin-bottom:20px;
    }

    .hero-title{
        font-size:34px;
        font-weight:800;
        margin-bottom:6px;
    }

    .hero-sub{
        color:#cbd5e1;
        font-size:15px;
    }

    .info-card{
        background:linear-gradient(145deg,#111827,#1f2937);
        border:1px solid rgba(255,255,255,.08);
        border-radius:18px;
        padding:18px;
        color:white;
        text-align:center;
        transition:.25s;
    }

    .info-card:hover{
        transform:translateY(-4px);
        border-color:#3b82f6;
    }

    .info-label{
        color:#94a3b8;
        font-size:13px;
        text-transform:uppercase;
        letter-spacing:1px;
    }

    .info-value{
        font-size:30px;
        font-weight:800;
        margin-top:6px;
        color:#ffffff;
    }

    .feature-card{
        background:rgba(255,255,255,.03);
        border:1px solid rgba(255,255,255,.08);
        border-radius:16px;
        padding:18px;
        height:140px;
    }

    .feature-title{
        font-weight:700;
        color:#60a5fa;
        margin-bottom:8px;
        font-size:16px;
    }

    .feature-text{
        color:#cbd5e1;
        font-size:13px;
        line-height:1.5;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="hero-card">
        <div class="hero-title">💬 SentimentX AI Dashboard</div>
        <div class="hero-sub">
            Advanced Sentiment Intelligence Platform powered by
            TF-IDF and Logistic Regression
        </div>
        <br>
        <b>Model Accuracy:</b> {test_accuracy*100:.2f}% &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Macro F1:</b> {macro_f1*100:.2f}% &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Dataset:</b> {training_records:,} Samples
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-label">Training Samples</div>
            <div class="info-value">{training_records:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-label">Test Accuracy</div>
            <div class="info-value">{test_accuracy*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-label">Macro F1</div>
            <div class="info-value">{macro_f1*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    left, right = st.columns([1,1])

    with left:

        st.subheader("Training Dataset Distribution")

        distribution = load_training_distribution()

        fig = px.pie(
            distribution,
            names="Sentiment",
            values="Examples",
            hole=.62,
            color="Sentiment",
            color_discrete_map=SENTIMENT_COLORS,
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
        )

        fig.update_layout(
            showlegend=True,
            height=430,
            margin=dict(t=20,b=20,l=20,r=20),
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        st.subheader("Class-wise Performance")

        report = test_metrics.get("classification_report", {})
        rows=[]

        for sentiment in ["negative","neutral","positive"]:
            values=report.get(sentiment,{})
            if values:
                rows.append({
                    "Sentiment": sentiment.capitalize(),
                    "Precision": values["precision"]*100,
                    "Recall": values["recall"]*100,
                    "F1": values["f1-score"]*100,
                })

        if rows:

            performance=pd.DataFrame(rows)

            chart=performance.melt(
                id_vars="Sentiment",
                value_vars=["Precision","Recall","F1"],
                var_name="Metric",
                value_name="Score"
            )

            fig=px.bar(
                chart,
                x="Sentiment",
                y="Score",
                color="Metric",
                barmode="group",
                height=430,
            )

            fig.update_layout(
                yaxis_range=[0,100],
                margin=dict(t=20,b=20,l=20,r=20),
            )

            st.plotly_chart(fig,use_container_width=True)

            st.dataframe(
                performance.round(2),
                hide_index=True,
                use_container_width=True,
            )

    st.divider()

    st.subheader("Model Architecture")

    m1,m2,m3=st.columns(3)

    with m1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🧠 Machine Learning</div>
            <div class="feature-text">
            Logistic Regression classifier trained on
            three sentiment categories.
            </div>
        </div>
        """,unsafe_allow_html=True)

    with m2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📝 NLP Features</div>
            <div class="feature-text">
            TF-IDF vectorization with
            uni-grams and bi-grams.
            </div>
        </div>
        """,unsafe_allow_html=True)

    with m3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📊 Analytics</div>
            <div class="feature-text">
            Interactive charts,
            confidence scores,
            PDF reports and history tracking.
            </div>
        </div>
        """,unsafe_allow_html=True)

    st.divider()

    st.success(
        "Model Status: Successfully loaded and ready for real-time sentiment prediction."
    )


# =========================================================
# 8. TEXT ANALYZER — PREMIUM UI
# =========================================================

elif page == "Text Analyzer":

    st.markdown("""
    <style>
    .analyzer-hero {
        padding: 24px;
        border-radius: 18px;
        background: linear-gradient(120deg, #0f172a, #1e3a8a);
        border: 1px solid rgba(96,165,250,.25);
        margin-bottom: 20px;
    }
    .analyzer-title {
        font-size: 29px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 8px;
    }
    .analyzer-subtitle {
        color: #cbd5e1;
        font-size: 14px;
        line-height: 1.7;
    }
    .result-card {
        padding: 22px;
        border-radius: 17px;
        border: 1px solid rgba(148,163,184,.25);
        background: rgba(100,116,139,.08);
        margin: 12px 0 18px 0;
    }
    .result-label {
        color: #94a3b8;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .result-value {
        font-size: 32px;
        font-weight: 800;
        margin-top: 8px;
    }
    .result-note {
        color: #94a3b8;
        font-size: 13px;
        margin-top: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="analyzer-hero">
        <div class="analyzer-title">💬 Text Sentiment Analyzer</div>
        <div class="analyzer-subtitle">
            Understand the emotional tone of reviews, feedback and comments.
            Enter your text below to analyze it using the trained
            TF-IDF + Logistic Regression model.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("#### ✍️ Enter your text")

        text = st.text_area(
            "Review or comment",
            placeholder=(
                "Example: The camera is excellent "
                "but the battery is poor."
            ),
            height=160,
            key="single_text_input",
            label_visibility="collapsed",
        )

        st.caption(
            f"Characters entered: {len(text)}"
        )

        analyze_clicked = st.button(
            "🔍 Analyze Sentiment",
            type="primary",
            use_container_width=True,
        )

    if analyze_clicked:

        if not text.strip():
            st.warning("Please enter some text before analyzing.")

        else:
            result = analyze_texts([text], model)
            save_to_history(result, source="Text Analyzer")

            sentiment = result.iloc[0]["Sentiment"]
            confidence = float(result.iloc[0]["Confidence (%)"])

            sentiment_styles = {
                "Positive": {
                    "emoji": "😊",
                    "color": "#16a34a",
                    "message": "The model classified this text as positive.",
                },
                "Negative": {
                    "emoji": "😟",
                    "color": "#dc2626",
                    "message": "The model classified this text as negative.",
                },
                "Neutral": {
                    "emoji": "😐",
                    "color": "#64748b",
                    "message": "The model classified this text as neutral.",
                },
            }

            style = sentiment_styles.get(
                sentiment,
                {
                    "emoji": "💬",
                    "color": "#2563eb",
                    "message": "The model produced a sentiment prediction.",
                },
            )

            st.divider()
            st.subheader("📊 Analysis Results")

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Predicted sentiment</div>
                    <div class="result-value" style="color:{style['color']}">
                        {style['emoji']} {sentiment}
                    </div>
                    <div class="result-note">{style['message']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            metric1, metric2 = st.columns(2)

            metric1.metric(
                "Model Confidence",
                f"{confidence:.2f}%",
            )

            metric2.metric(
                "Text Length",
                f"{len(text.strip()):,} characters",
            )

            st.caption(
                "Confidence represents the model's estimated probability "
                "for its prediction; it does not guarantee correctness."
            )

            with st.expander("🧠 View all class probabilities"):

                probabilities = model.predict_proba([text.strip()])[0]

                probability_df = pd.DataFrame({
                    "Sentiment": [
                        str(label).capitalize()
                        for label in model.classes_
                    ],
                    "Estimated probability (%)": [
                        round(float(value) * 100, 2)
                        for value in probabilities
                    ],
                })

                probability_df = probability_df.sort_values(
                    "Estimated probability (%)",
                    ascending=False,
                )

                st.dataframe(
                    probability_df,
                    use_container_width=True,
                    hide_index=True,
                )

                probability_chart = px.bar(
                    probability_df,
                    x="Sentiment",
                    y="Estimated probability (%)",
                    color="Sentiment",
                    color_discrete_map=SENTIMENT_COLORS,
                    range_y=[0, 100],
                )

                probability_chart.update_layout(
                    height=320,
                    showlegend=False,
                    margin=dict(t=20, b=20, l=15, r=15),
                )

                st.plotly_chart(
                    probability_chart,
                    use_container_width=True,
                )

            st.markdown("#### 📄 Export your result")

            st.download_button(
                "⬇️ Download PDF Report",
                data=create_pdf_report(
                    result,
                    title="SentimentX AI Text Analysis Report",
                    source="Single Text Analysis",
                ),
                file_name="sentimentx_analysis.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

# =========================================================
# 9. BULK CSV ANALYZER — PREMIUM UI
# =========================================================

elif page == "Bulk Analyzer":

    st.markdown("""
    <style>
    .bulk-hero {
        background: linear-gradient(125deg, #0f172a, #1e3a8a);
        border: 1px solid rgba(96,165,250,.25);
        border-radius: 18px;
        padding: 25px;
        margin-bottom: 20px;
    }
    .bulk-title {
        color: #ffffff;
        font-size: 29px;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .bulk-subtitle {
        color: #cbd5e1;
        font-size: 14px;
        line-height: 1.7;
    }
    .bulk-tip {
        padding: 15px;
        border: 1px solid rgba(148,163,184,.22);
        border-radius: 13px;
        background: rgba(100,116,139,.07);
        margin-bottom: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="bulk-hero">
        <div class="bulk-title">📂 Bulk Sentiment Analyzer</div>
        <div class="bulk-subtitle">
            Analyze multiple reviews in one go. Upload a CSV file to
            generate sentiment predictions, confidence summaries and
            downloadable reports.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="bulk-tip">
        <b>📌 CSV format</b><br>
        Your file should contain a text column named
        <code>text</code>, <code>review</code>, <code>comment</code>
        or <code>content</code>.
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        key="bulk_csv_uploader",
        help="Upload a CSV containing reviews, comments or other text.",
    )

    if uploaded is not None:

        try:
            data = pd.read_csv(uploaded)

            columns = {
                str(column).strip().lower(): column
                for column in data.columns
            }

            source_column = next(
                (
                    columns[name]
                    for name in ("text", "review", "comment", "content")
                    if name in columns
                ),
                None,
            )

            if source_column is None:

                st.error(
                    "No supported text column found. Use a column named "
                    "text, review, comment or content."
                )

            elif data.empty:

                st.warning("The uploaded CSV contains no rows.")

            else:

                st.markdown("### 📊 Analysis Overview")

                results = analyze_texts(
                    data[source_column].fillna("").tolist(),
                    model,
                )

                valid = results[
                    results["Sentiment"] != "Not analyzed"
                ]

                total_rows = len(results)
                analyzed_rows = len(valid)
                skipped_rows = total_rows - analyzed_rows

                positive_count = int(
                    (valid["Sentiment"] == "Positive").sum()
                )
                neutral_count = int(
                    (valid["Sentiment"] == "Neutral").sum()
                )
                negative_count = int(
                    (valid["Sentiment"] == "Negative").sum()
                )

                a, b, c = st.columns(3)

                a.metric("📄 Total Rows", f"{total_rows:,}")
                b.metric("✅ Analyzed", f"{analyzed_rows:,}")
                c.metric("⏭️ Skipped Rows", f"{skipped_rows:,}")

                st.markdown("#### Sentiment Summary")

                s1, s2, s3 = st.columns(3)

                s1.metric("😊 Positive", positive_count)
                s2.metric("😐 Neutral", neutral_count)
                s3.metric("😟 Negative", negative_count)

                if not valid.empty:

                    st.divider()

                    left, right = st.columns(2)

                    count_data = (
                        valid["Sentiment"]
                        .value_counts()
                        .reindex(SENTIMENT_ORDER, fill_value=0)
                        .rename_axis("Sentiment")
                        .reset_index(name="Reviews")
                    )

                    with left:

                        st.subheader("Sentiment Distribution")

                        fig = px.pie(
                            count_data,
                            names="Sentiment",
                            values="Reviews",
                            hole=0.58,
                            color="Sentiment",
                            color_discrete_map=SENTIMENT_COLORS,
                        )

                        fig.update_traces(
                            textposition="inside",
                            textinfo="percent+label",
                        )

                        fig.update_layout(
                            height=360,
                            margin=dict(t=20, b=15, l=10, r=10),
                            legend_title_text="Sentiment",
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                        )

                    with right:

                        st.subheader("Average Model Confidence")

                        confidence_data = (
                            valid.groupby("Sentiment", as_index=False)
                            ["Confidence (%)"]
                            .mean()
                        )

                        confidence_data["Confidence (%)"] = (
                            confidence_data["Confidence (%)"].round(2)
                        )

                        confidence_chart = px.bar(
                            confidence_data,
                            x="Sentiment",
                            y="Confidence (%)",
                            color="Sentiment",
                            color_discrete_map=SENTIMENT_COLORS,
                            range_y=[0, 100],
                        )

                        confidence_chart.update_layout(
                            height=360,
                            showlegend=False,
                            margin=dict(t=20, b=15, l=10, r=10),
                        )

                        st.plotly_chart(
                            confidence_chart,
                            use_container_width=True,
                        )

                        st.dataframe(
                            confidence_data,
                            use_container_width=True,
                            hide_index=True,
                        )

                st.divider()

                st.subheader("🔎 Prediction Results")

                st.caption(
                    "Review the model's predictions below. Blank rows are "
                    "retained in the results and marked as not analyzed."
                )

                st.dataframe(
                    results,
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("### 💾 Save and Export")

                save_col, csv_col, pdf_col = st.columns(3)

                with save_col:

                    if st.button(
                        "💾 Save to History",
                        use_container_width=True,
                    ):

                        save_to_history(
                            results,
                            source="Bulk CSV Analyzer",
                        )

                        st.success(
                            f"Saved {len(valid)} analyzed rows to history."
                        )

                with csv_col:

                    st.download_button(
                        "⬇️ Download CSV",
                        data=results.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="sentimentx_bulk_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                with pdf_col:

                    st.download_button(
                        "📄 Download PDF",
                        data=create_pdf_report(
                            results,
                            title="SentimentX AI Bulk Analysis Report",
                            source="Bulk CSV Analysis",
                        ),
                        file_name="sentimentx_bulk_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )

        except (ValueError, UnicodeDecodeError) as error:

            st.error(f"Unable to read CSV: {error}")


elif page == "Analysis History":
    st.markdown("""
    <div class="hero-card">
        <div class="hero-eyebrow">YOUR WORKSPACE</div>
        <h1>Analysis History</h1>
        <p>Review your saved predictions, track sentiment patterns,
        and export your analysis reports.</p>
    </div>
    """, unsafe_allow_html=True)

    history = st.session_state.get("analysis_history", [])

    if history:
        history_df = pd.DataFrame(history)

        # Sentiment counts
        total_saved = len(history_df)
        positive_count = int(
            (history_df["Sentiment"] == "Positive").sum()
        )
        neutral_count = int(
            (history_df["Sentiment"] == "Neutral").sum()
        )
        negative_count = int(
            (history_df["Sentiment"] == "Negative").sum()
        )

        # Metric cards
        st.markdown("### History Overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Analyses", total_saved)
        col2.metric("Positive", positive_count)
        col3.metric("Neutral", neutral_count)
        col4.metric("Negative", negative_count)

        st.divider()

        # Sentiment distribution
        left, right = st.columns([1, 1.4])

        with left:
            st.markdown("### Sentiment Distribution")

            distribution_df = pd.DataFrame({
                "Sentiment": ["Positive", "Neutral", "Negative"],
                "Count": [
                    positive_count,
                    neutral_count,
                    negative_count,
                ],
            })

            distribution_df = distribution_df[
                distribution_df["Count"] > 0
            ]

            if not distribution_df.empty:
                fig = px.pie(
                    distribution_df,
                    names="Sentiment",
                    values="Count",
                    hole=0.58,
                    color="Sentiment",
                    color_discrete_map={
                        "Positive": "#22C55E",
                        "Neutral": "#F59E0B",
                        "Negative": "#EF4444",
                    },
                )

                fig.update_layout(
                    height=300,
                    margin=dict(t=10, b=10, l=10, r=10),
                    showlegend=True,
                )

                fig.update_traces(
                    textposition="inside",
                    textinfo="percent+label",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        with right:
            st.markdown("### Sentiment Insights")

            if total_saved:
                positive_pct = positive_count / total_saved * 100
                neutral_pct = neutral_count / total_saved * 100
                negative_pct = negative_count / total_saved * 100

                st.markdown(
                    f"""
                    <div class="result-card">
                        <h4 style="color:#22C55E;">Positive</h4>
                        <h2>{positive_pct:.1f}%</h2>
                        <p>{positive_count} saved analyses</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="result-card">
                        <h4 style="color:#F59E0B;">Neutral</h4>
                        <h2>{neutral_pct:.1f}%</h2>
                        <p>{neutral_count} saved analyses</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="result-card">
                        <h4 style="color:#EF4444;">Negative</h4>
                        <h2>{negative_pct:.1f}%</h2>
                        <p>{negative_count} saved analyses</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.divider()

        # Search history
        st.markdown("### Saved Prediction Records")

        search_query = st.text_input(
            "Search history",
            placeholder="Search text or sentiment...",
            key="history_search",
        )

        filtered_df = history_df.copy()

        if search_query.strip():
            query = search_query.strip().lower()

            text_match = (
                filtered_df["Text"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains(query, regex=False)
            )

            sentiment_match = (
                filtered_df["Sentiment"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains(query, regex=False)
            )

            filtered_df = filtered_df[
                text_match | sentiment_match
            ]

        st.caption(
            f"Showing {len(filtered_df)} of {total_saved} saved records"
        )

        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
        )

        # Export options
        st.markdown("### Export Your History")

        export_col1, export_col2 = st.columns(2)

        with export_col1:
            st.download_button(
                "Download History CSV",
                data=history_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="sentimentx_history.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with export_col2:
            pdf_columns = [
                column
                for column in ["Text", "Sentiment", "Confidence (%)"]
                if column in history_df.columns
            ]

            pdf_history = history_df[pdf_columns]

            st.download_button(
                "Download History PDF",
                data=create_pdf_report(
                    pdf_history,
                    title="SentimentX AI Analysis History",
                    source="Current Session History",
                ),
                file_name="sentimentx_history.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        st.caption(
            "History is stored in the current Streamlit session. "
            "It is not permanently saved to a database and may reset "
            "when the session ends."
        )

        st.divider()

        # Clear history
        with st.expander("History Management"):
            st.warning(
                "Clearing history removes all saved analyses "
                "from the current session."
            )

            confirm_clear = st.checkbox(
                "I understand that this action cannot be undone.",
                key="confirm_clear_history",
            )

            if st.button(
                "Clear All History",
                disabled=not confirm_clear,
                type="primary",
            ):
                st.session_state.analysis_history = []
                st.rerun()

    else:
        st.markdown("""
        <div class="result-card" style="text-align:center;">
            <h2>No Analysis History Yet</h2>
            <p>Your saved text predictions and bulk analyses
            will appear here.</p>
            <p>Start with Text Analyzer or Bulk Analyzer
            to build your history.</p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# 11. ABOUT PROJECT
# =========================================================
elif page == "About Project":
    st.header("About SentimentX AI")

    st.write(
        "SentimentX AI is a Natural Language Processing application "
        "that classifies text into positive, negative and neutral sentiment."
    )

    st.subheader("Current methodology")

    st.markdown(
        """
        1. **Input:** Accept a review, feedback message or social-media text.
        2. **Text features:** TF-IDF converts text into numerical features
           using individual words and two-word combinations.
        3. **Classification:** Logistic Regression predicts one of the
           three sentiment classes.
        4. **Estimated probability:** The application displays estimated
           class probabilities.
        5. **Batch analysis:** CSV files can be analyzed and exported.
        6. **PDF reporting:** Export individual or bulk analysis results.
        7. **Session history:** Review analyses saved during the current session.
        8. **Evaluation:** Display test-set accuracy, precision, recall and F1.
        """
    )

    st.subheader("Training and evaluation")

    if training_records:
        st.write(f"Training records: {training_records:,}")

    if test_accuracy is not None:
        st.write(f"Test accuracy: {test_accuracy * 100:.2f}%")

    if macro_f1 is not None:
        st.write(f"Test macro F1-score: {macro_f1 * 100:.2f}%")

    st.subheader("Current limitations")

    st.markdown(
        """
        - Sarcasm, slang, irony and mixed opinions may be misclassified.
        - Neutral text can be difficult to distinguish from other classes.
        - Confidence is not a guarantee of prediction correctness.
        - Evaluation results apply to the provided test dataset and may
          differ on other text sources.
        - Emotion detection and aspect-based sentiment analysis are
          not implemented in this version.
        - Analysis history currently lasts only for the current session.
        """
    )

    st.caption(
        "Model: TF-IDF + Logistic Regression | "
        "Classes: Negative, Neutral and Positive"
    )


# =========================================================
# 12. ABOUT ME / MEET THE DEVELOPER
# =========================================================

elif page == "About Me":
    st.header("Meet the Developer")
    st.caption("The person behind SentimentX AI")

    st.markdown(
        """
        <div style="
            padding: 28px;
            border: 1px solid rgba(128,128,128,0.35);
            border-radius: 18px;
            background: linear-gradient(
                135deg,
                rgba(59,130,246,0.15),
                rgba(100,116,139,0.08)
            );
            margin-bottom: 20px;
        ">
            <h1 style="margin-bottom: 5px;">
                Karan Dev Singh
            </h1>
            <p style="font-size: 17px; margin-bottom: 16px;">
                Student Developer
            </p>
            <hr>
            <p><b>College</b></p>
            <p>Shri Ramswaroop Memorial University</p>
            <p><b>Project</b></p>
            <p>SentimentX AI — Sentiment Intelligence Platform</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("About Me")

    st.write(
        "I am a Student Developer working on SentimentX AI, "
        "a Natural Language Processing and Machine Learning project. "
        "The application analyzes text and predicts whether its "
        "sentiment is positive, neutral or negative. It also provides "
        "bulk CSV analysis, visualizations, PDF reports and session history."
    )

    st.subheader("Project Focus")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### NLP")
        st.write("Text processing and TF-IDF features")

    with col2:
        st.markdown("### Machine Learning")
        st.write("TF-IDF + Logistic Regression")

    with col3:
        st.markdown("### Analytics")
        st.write("Sentiment charts and downloadable reports")

    st.divider()
    st.subheader("Connect with Me")

    github_col, email_col = st.columns(2)

    with github_col:
        st.markdown("#### GitHub")

        st.link_button(
            "Visit GitHub Profile",
            "https://github.com/KARAN-2317",
            use_container_width=True,
        )

    with email_col:
        st.markdown("#### Email")

        st.link_button(
            "Send an Email",
            "mailto:karandev2323@gmail.com",
            use_container_width=True,
        )

    st.divider()

    st.caption(
        "SentimentX AI is a student project. Model predictions "
        "are estimates and may not always be correct."
    )