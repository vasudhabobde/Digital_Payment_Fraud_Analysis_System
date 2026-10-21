"""
Digital Payment Fraud Pattern Analysis - Streamlit dashboard.

Rule-based fraud screening: suspicious behaviour earns risk points, the points are
added into a 0-100 score, and transactions at or above a threshold are flagged for review.
"""
from __future__ import annotations

import json
import hashlib
import io
import joblib
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_CSV = BASE_DIR / "data" / "digital_payment.csv"
DB_PATH = BASE_DIR / "database" / "review_decisions.db"
MODEL_PATH = BASE_DIR / "models" / "fraud_model.joblib"
FEATURE_PATH = BASE_DIR / "data" / "feature_columns.json"
RULE_PATH = BASE_DIR / "data" / "risk_rules.json"
PAGES = ["Overview Dashboard", "Dataset Analysis", "Fraud Detection", "Fraud Review Center", "History & Reports"]
REQUIRED_COLUMNS = ["payment_type", "amount", "oldbalance_sender", "newbalance_sender", "trans_hour"]
DEFAULT_RULE_THRESHOLD = 20   # points needed to flag a transaction
HIGH_RISK_POINTS = 40         # points at which a flagged transaction is "High" risk
DECISIONS = ["Pending Review", "Legitimate", "Confirmed Fraud", "Needs Further Investigation"]

NAVY, BLUE, LIGHT_BLUE, RED, GRID = "#122342", "#1764CE", "#65A9E8", "#D43C52", "#DCE3EF"

CSS = """
<style>
.stApp, [data-testid="stAppViewContainer"] { background:#F4F7FB !important; color:#0B1F44; }
[data-testid="stHeader"] { background:#F4F7FB !important; }
[data-testid="stHeader"] * { color:#0B1F44 !important; }
.block-container { padding-top:2.2rem; padding-bottom:3rem; max-width:1400px; }
h1,h2,h3,h4 { color:#0B1F44 !important; letter-spacing:-0.01em; }
h1 { font-weight:750 !important; font-size:2rem !important; margin-bottom:.1rem !important; }
.page-sub { color:#5B6B86; margin:0 0 1.4rem 0; font-size:.98rem; }

/* Sidebar */
[data-testid="stSidebar"] { background:#FFFFFF !important; border-right:1px solid #E3EAF4; }
[data-testid="stSidebar"] * { color:#0B1F44 !important; }
.brand { font-size:1.15rem; font-weight:750; color:#0B1F44; line-height:1.25; }
.brand small { display:block; font-weight:500; font-size:.8rem; color:#5B6B86 !important; margin-top:4px; }
.status-pill { display:inline-block; padding:4px 12px; border-radius:999px; font-size:.78rem; font-weight:600; }
.status-ok  { background:#E7F0FF; color:#1D4ED8 !important; }
.status-bad { background:#FEE9E9; color:#B91C1C !important; }
[data-testid="stSidebar"] [role="radiogroup"] { gap:.25rem; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    padding:.65rem .85rem; border-radius:10px; width:100%; transition:background .15s; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background:#EEF3FB; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background:#E7F0FF; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) * { color:#1D4ED8 !important; font-weight:650; }
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display:none; }

/* KPI cards */
.kpi { background:#fff; border:1px solid #E3EAF4; border-radius:14px; padding:18px 20px;
       box-shadow:0 1px 2px rgba(11,31,68,.04); height:118px; margin-bottom:1rem;
       display:flex; flex-direction:column; justify-content:center; overflow:hidden; }
.kpi .label { font-size:.74rem; font-weight:650; letter-spacing:.06em; text-transform:uppercase; color:#5B6B86; }
.kpi .value { font-size:1.75rem; font-weight:750; color:#0B1F44; line-height:1.2; margin-top:6px;
              white-space:normal; overflow-wrap:anywhere; }
.kpi .sub { font-size:.78rem; color:#6B7A93; margin-top:2px; }
.kpi.accent { border-left:4px solid #2563EB; }
.kpi.danger { border-left:4px solid #EF4444; }

/* Chart / content cards */
div[class*="st-key-card"] { background:#fff; border:1px solid #E3EAF4; border-radius:14px;
       padding:14px 18px 8px 18px; box-shadow:0 1px 2px rgba(11,31,68,.04); margin-bottom:1rem; }
.card-title { font-weight:700; color:#0B1F44; font-size:1rem; margin-bottom:2px; }

/* Tabs, buttons, tables */
button[data-baseweb="tab"] { font-weight:600; }
.stButton > button, .stDownloadButton > button { border-radius:10px; font-weight:600; }
.stButton > button[kind="primary"] { background:#2563EB; border-color:#2563EB; color:#fff; }
[data-testid="stDataFrame"], [data-testid="stDataEditor"] { border:1px solid #E3EAF4; border-radius:12px; overflow:hidden; }
/* High contrast for controls, tabs, metrics and data displays */
[data-testid="stAppViewContainer"] p, [data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] span, [data-testid="stAppViewContainer"] div[data-testid="stMarkdownContainer"] {
 color: #203451;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span {color:#203451;}
[data-testid="stMetric"] {background:#FFFFFF; border:1px solid #DBE3EF; padding:14px; border-radius:12px;}
[data-testid="stMetricValue"] {color:#122342 !important;}
[data-testid="stMetricLabel"] {color:#405579 !important;}
[data-baseweb="input"] input, [data-baseweb="select"] input, textarea {color:#122342 !important;}
button[data-baseweb="tab"] {color:#334866 !important; opacity:1 !important;}
button[data-baseweb="tab"][aria-selected="true"] {color:#1764CE !important; border-bottom:3px solid #1764CE;}
[data-testid="stAlert"] * {opacity:1 !important;}
footer { visibility:hidden; }
</style>
"""

# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def fmt_inr(value: float) -> str:
    """Compact rupee format that never gets truncated in a card."""
    v = float(value)
    if abs(v) >= 1e7:
        return f"₹{v / 1e7:,.2f} Cr"
    if abs(v) >= 1e5:
        return f"₹{v / 1e5:,.2f} L"
    return f"₹{v:,.2f}"


def kpi(label: str, value: str, sub: str = "", tone: str = "accent") -> None:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="kpi {tone}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(color=NAVY, size=13), legend=dict(orientation="h", y=1.12, x=0, title=None), colorway=[BLUE,"#0D9488",RED,"#805AD5","#E69E21"],
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def chart_card(key: str, title: str, fig: go.Figure, height: int = 340) -> None:
    with st.container(key=f"card_{key}"):
        st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
        st.plotly_chart(style_fig(fig, height), key=f"chart_{key}", config={"displaylogo": False})


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f"# {title}")
    st.markdown(f'<p class="page-sub">{subtitle}</p>', unsafe_allow_html=True)


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


# --------------------------------------------------------------------------- #
# Data validation
# --------------------------------------------------------------------------- #
def validate_dataframe(raw: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str], str | None]:
    """Cleans an uploaded dataframe. Returns (clean_df, warnings, error)."""
    warnings: list[str] = []
    if raw is None or raw.empty:
        return None, warnings, "The file contains no rows."

    df = raw.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return None, warnings, (f"Missing required column(s): {', '.join(missing)}. "
                                f"Required: {', '.join(REQUIRED_COLUMNS)}. Found: {', '.join(df.columns)}.")

    n0 = len(df)
    for col in ["amount", "oldbalance_sender", "newbalance_sender", "trans_hour"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["payment_type"] = df["payment_type"].astype("string").str.strip().str.upper()

    bad = df[REQUIRED_COLUMNS].isna().any(axis=1) | (df["amount"] <= 0) | ~df["trans_hour"].between(0, 23)
    if bad.any():
        warnings.append(f"Removed {int(bad.sum())} row(s) with missing/invalid values "
                        "(non-numeric, amount <= 0, or hour outside 0-23).")
        df = df[~bad].copy()
    if df.empty:
        return None, warnings, "No valid rows remain after cleaning."

    df["trans_hour"] = df["trans_hour"].astype(int)
    df["payment_type"] = df["payment_type"].astype(str)

    if "trans_id" not in df.columns:
        df.insert(0, "trans_id", [f"ROW{i + 1:06d}" for i in range(len(df))])
        warnings.append("No `trans_id` column found, so row IDs were generated.")
    df["trans_id"] = df["trans_id"].astype(str)
    dups = df["trans_id"].duplicated()
    if dups.any():
        warnings.append(f"Removed {int(dups.sum())} duplicate `trans_id` row(s).")
        df = df[~dups].copy()

    if "is_fraud" in df.columns:
        label = pd.to_numeric(df["is_fraud"], errors="coerce")
        if label.isin([0, 1]).all():
            df["is_fraud"] = label.astype(int)
        else:
            df = df.drop(columns="is_fraud")
            warnings.append("`is_fraud` contains values other than 0/1, so it was ignored.")

    if len(df) < n0 and not warnings:
        warnings.append(f"{n0 - len(df)} row(s) were removed during cleaning.")
    return df.reset_index(drop=True), warnings, None


# --------------------------------------------------------------------------- #
# Rule-based risk scoring (points system)
# --------------------------------------------------------------------------- #
RULE_POINTS = {
    "Unusually high amount": 20,
    "Many transactions in a short period": 25,
    "Repeated failed attempts": 15,
    "Unusual location": 20,
    "Duplicate payment pattern": 20,
}
RULE_HELP = {
    "Unusually high amount": ("Amount is in the top 5% of the dataset.", "amount"),
    "Many transactions in a short period":
        ("Same sender made 3 or more transactions within the same hour (the data has hours, not full timestamps).",
         "sender_id"),
    "Repeated failed attempts": ("`failed_attempts` is 3 or more.", "failed_attempts"),
    "Unusual location": ("`location` is rare (used by under 5% of transactions).", "location"),
    "Duplicate payment pattern": ("Same sender, receiver and amount appear more than once.", "sender_id + receiver_id"),
}


def compute_rule_scores(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, dict]:
    """Returns (rule_score 0-100, triggered-rules text, availability per rule).

    A rule is only evaluated when the columns it needs exist; otherwise it
    is reported as unavailable and adds no points.
    """
    cols = set(df.columns)
    hits: dict[str, pd.Series] = {}

    hits["Unusually high amount"] = df["amount"] >= df["amount"].quantile(0.95)

    if "sender_id" in cols:
        n_same_hour = df.groupby(["sender_id", "trans_hour"])["trans_id"].transform("size")
        hits["Many transactions in a short period"] = n_same_hour >= 3

    if "failed_attempts" in cols:
        hits["Repeated failed attempts"] = pd.to_numeric(df["failed_attempts"], errors="coerce").fillna(0) >= 3

    if "location" in cols:
        share = df["location"].map(df["location"].value_counts(normalize=True))
        hits["Unusual location"] = share < 0.05

    if {"sender_id", "receiver_id"} <= cols:
        n_dup = df.groupby(["sender_id", "receiver_id", df["amount"].round(2)])["trans_id"].transform("size")
        hits["Duplicate payment pattern"] = n_dup >= 2

    score = pd.Series(0, index=df.index, dtype=int)
    for name, hit in hits.items():
        score = score + hit.astype(int) * RULE_POINTS[name]
    triggered = pd.Series("", index=df.index, dtype=object)
    for name, hit in hits.items():
        triggered = triggered.where(~hit, triggered + np.where(triggered == "", "", "; ") + name)
    return score, triggered, {name: name in hits for name in RULE_POINTS}


# --------------------------------------------------------------------------- #
# SQLite review storage
# --------------------------------------------------------------------------- #
def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS review_decisions (
               trans_id TEXT PRIMARY KEY,
               risk_score REAL,
               decision TEXT NOT NULL,
               notes TEXT,
               reviewed_at TEXT NOT NULL)"""
    )
    return conn


def save_decisions(rows: list[tuple]) -> int:
    """rows: (trans_id, risk_score, decision, notes). Upserts by trans_id."""
    now = datetime.now().isoformat(timespec="seconds")
    with closing(_connect()) as conn, conn:
        conn.executemany(
            """INSERT INTO review_decisions (trans_id, risk_score, decision, notes, reviewed_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(trans_id) DO UPDATE SET risk_score=excluded.risk_score,
                   decision=excluded.decision, notes=excluded.notes, reviewed_at=excluded.reviewed_at""",
            [(t, float(s), d, n or "", now) for t, s, d, n in rows],
        )
    return len(rows)


def load_decisions() -> pd.DataFrame:
    with closing(_connect()) as conn:
        return pd.read_sql_query(
            "SELECT trans_id, risk_score, decision, notes, reviewed_at "
            "FROM review_decisions ORDER BY reviewed_at DESC", conn)


# --------------------------------------------------------------------------- #
# Active dataset (uploaded > bundled sample)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_sample(mtime: float):
    raw = pd.read_csv(SAMPLE_CSV)
    return validate_dataframe(raw)


def get_active_dataset() -> tuple[pd.DataFrame | None, str]:
    if st.session_state.get("df") is not None:
        return st.session_state["df"], st.session_state.get("source", "Uploaded file")
    if SAMPLE_CSV.exists():
        df, _, err = load_sample(_mtime(SAMPLE_CSV))
        if df is not None:
            return df, "Bundled sample dataset (data/digital_payment.csv)"
    return None, ""


def read_uploaded_csv(uploaded) -> tuple[pd.DataFrame | None, list[str], str | None]:
    try:
        raw = pd.read_csv(uploaded)
    except pd.errors.EmptyDataError:
        return None, [], "The file is empty."
    except pd.errors.ParserError as exc:
        return None, [], f"The file is not a valid CSV ({exc})."
    except UnicodeDecodeError:
        return None, [], "The file could not be decoded. Please upload a UTF-8 encoded CSV."
    except Exception as exc:
        return None, [], f"Could not read the file: {exc}"
    return validate_dataframe(raw)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown('<div class="brand">Digital Payment<br>Fraud Analysis'
                    '<small>Big Data Analytics project · PySpark</small></div>', unsafe_allow_html=True)
        st.write("")
        page = st.radio("Navigation", PAGES, label_visibility="collapsed", key="nav")
        st.divider()
        _, src = get_active_dataset()
        st.caption(f"Active data: {src}" if src else "No active dataset")
    return page



def page_dataset() -> None:
    page_header("Dataset Analysis", "Upload a CSV, explore transaction patterns and review summary statistics.")

    with st.container(key="card_upload"):
        st.markdown('<div class="card-title">Upload dataset</div>', unsafe_allow_html=True)
        st.caption("Required columns: " + ", ".join(REQUIRED_COLUMNS) +
                   ". Optional: trans_id, sender_id, receiver_id, is_fraud.")
        up = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
        if up is not None and st.session_state.get("upload_sig") != (up.name, hashlib.sha256(up.getvalue()).hexdigest()):
            df, warns, err = read_uploaded_csv(up)
            if err:
                st.error(err)
            else:
                st.session_state.pop("active_run", None)
                st.session_state.update(df=df, source=f"Uploaded: {up.name}", upload_warnings=warns,
                                        upload_sig=(up.name, hashlib.sha256(up.getvalue()).hexdigest()))
                st.rerun()
        if st.session_state.get("df") is not None:
            b1, _ = st.columns([1, 4])
            if b1.button("Use sample dataset instead"):
                for k in ("df", "source", "upload_warnings", "upload_sig", "active_run"):
                    st.session_state.pop(k, None)
                st.rerun()
        for w in st.session_state.get("upload_warnings", []):
            st.warning(w)

    df, source = get_active_dataset()
    if df is None:
        st.info("Upload a CSV file to begin.")
        return
    st.caption(f"Active dataset: {source} · {len(df):,} rows")
    has_label = "is_fraud" in df.columns
    has_sender = "sender_id" in df.columns
    has_receiver = "receiver_id" in df.columns

    t1, t2, t3 = st.tabs(["Data preview", "Transaction patterns", "Summary statistics"])

    with t1:
        f1, f2, f3 = st.columns([2, 2, 3])
        types = f1.multiselect("Payment type", sorted(df["payment_type"].unique()))
        label_opt = f2.selectbox("Fraud label", ["All", "Fraud only", "Legitimate only"]) if has_label else "All"
        query = f3.text_input("Search transaction ID", placeholder="e.g. TX10002")
        v = df
        if types:
            v = v[v["payment_type"].isin(types)]
        if label_opt == "Fraud only":
            v = v[v["is_fraud"] == 1]
        elif label_opt == "Legitimate only":
            v = v[v["is_fraud"] == 0]
        if query:
            v = v[v["trans_id"].str.contains(query, case=False, na=False)]
        st.caption(f"Showing {len(v):,} of {len(df):,} rows")
        st.dataframe(v, hide_index=True, height=420)
        st.download_button("Download filtered rows (CSV)", v.to_csv(index=False).encode("utf-8"),
                           "filtered_transactions.csv", "text/csv")

    with t2:
        c1, c2 = st.columns(2)
        with c1:
            if has_label:
                g = df.groupby("payment_type")["is_fraud"].agg(["size", "sum"]).reset_index()
                g["Fraud rate (%)"] = 100 * g["sum"] / g["size"]
                fig = px.bar(g.sort_values("size", ascending=False), x="payment_type", y="size",
                             color="Fraud rate (%)", color_continuous_scale=["#BFD7FF", BLUE, NAVY],
                             labels={"payment_type": "Payment type", "size": "Transactions"})
            else:
                fig = px.bar(df["payment_type"].value_counts().reset_index(), x="payment_type", y="count",
                             color_discrete_sequence=[BLUE], labels={"payment_type": "Payment type", "count": "Transactions"})
            chart_card("da_type", "Volume by payment type" + (" (colour = fraud rate)" if has_label else ""), fig)
        with c2:
            if has_label:
                d = df.assign(Class=np.where(df["is_fraud"] == 1, "Fraud", "Legitimate"))
                fig = px.histogram(d, x="amount", color="Class", nbins=40, log_x=True, barmode="overlay", opacity=.75,
                                   color_discrete_map={"Legitimate": BLUE, "Fraud": RED},
                                   labels={"amount": "Amount (₹, log scale)"})
            else:
                fig = px.histogram(df, x="amount", nbins=40, log_x=True, color_discrete_sequence=[BLUE],
                                   labels={"amount": "Amount (₹, log scale)"})
            chart_card("da_amt", "Amount distribution", fig)

        c1, c2 = st.columns(2)
        with c1:
            h = df.groupby("trans_hour").size().rename("Transactions").reset_index()
            fig = px.bar(h, x="trans_hour", y="Transactions", color_discrete_sequence=[LIGHT_BLUE],
                         labels={"trans_hour": "Hour of day"})
            if has_label:
                fh = df.groupby("trans_hour")["is_fraud"].sum().reset_index()
                fig.add_trace(go.Scatter(x=fh["trans_hour"], y=fh["is_fraud"], mode="lines+markers",
                                         name="Fraud cases", line=dict(color=RED, width=2.5)))
            chart_card("da_hour", "Activity by hour of day", fig)
        with c2:
            s = df.sample(min(len(df), 3000), random_state=1)
            if has_label:
                s = s.assign(Class=np.where(s["is_fraud"] == 1, "Fraud", "Legitimate"))
                fig = px.scatter(s[s.oldbalance_sender > 0], x="oldbalance_sender", y="amount", color="Class", opacity=.65, log_x=True, log_y=True,
                                 color_discrete_map={"Legitimate": BLUE, "Fraud": RED},
                                 labels={"oldbalance_sender": "Sender balance before (₹)", "amount": "Amount (₹)"})
            else:
                fig = px.scatter(s[s.oldbalance_sender > 0], x="oldbalance_sender", y="amount", opacity=.65, log_x=True, log_y=True,
                                 color_discrete_sequence=[BLUE],
                                 labels={"oldbalance_sender": "Sender balance before (₹)", "amount": "Amount (₹)"})
            chart_card("da_scatter", "Amount vs sender balance", fig)

        if has_sender:
            c1, c2 = st.columns(2)
            freq = df.groupby("sender_id").size().rename("transactions").reset_index()
            with c1:
                fig = px.histogram(freq, x="transactions", nbins=25, color_discrete_sequence=[BLUE],
                                   labels={"transactions": "Transactions per sender"})
                chart_card("da_freq", "Sender frequency", fig)
            with c2:
                with st.container(key="card_topsenders"):
                    st.markdown('<div class="card-title">Most active senders</div>', unsafe_allow_html=True)
                    top = freq.sort_values("transactions", ascending=False).head(10)
                    if has_label:
                        fc = df.groupby("sender_id")["is_fraud"].sum().rename("fraud_cases")
                        top = top.merge(fc, on="sender_id")
                    st.dataframe(top, hide_index=True, height=340)
        if has_receiver and has_label:
            top_r = (df[df["is_fraud"] == 1].groupby("receiver_id").size().rename("fraud_cases")
                     .sort_values(ascending=False).head(10).reset_index())
            if not top_r.empty:
                fig = px.bar(top_r, x="receiver_id", y="fraud_cases", color_discrete_sequence=[NAVY],
                             labels={"receiver_id": "Receiver", "fraud_cases": "Fraud cases"})
                chart_card("da_recv", "Receivers with the most fraud cases", fig, 300)
        st.caption("Location views are available only when a location column is provided.")

    with t3:
        num_cols = [c for c in ["amount", "oldbalance_sender", "newbalance_sender", "trans_hour"] + (["is_fraud"] if has_label else [])]
        with st.container(key="card_stats"):
            st.markdown('<div class="card-title">Numeric summary</div>', unsafe_allow_html=True)
            st.dataframe(df[num_cols].describe().T.round(2))
        c1, c2 = st.columns(2)
        with c1:
            with st.container(key="card_quality"):
                st.markdown('<div class="card-title">Data quality</div>', unsafe_allow_html=True)
                q = pd.DataFrame({"column": df.columns, "type": [str(t) for t in df.dtypes],
                                  "missing": df.isna().sum().values, "unique": df.nunique().values})
                st.dataframe(q, hide_index=True)
        with c2:
            corr = df[num_cols].corr().round(2)
            fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, zmin=-1, zmax=1,
                                       colorscale=[[0, "#DCE8FF"], [.5, "#FFFFFF"], [1, BLUE]],
                                       text=corr.values, texttemplate="%{text}"))
            fig.update_layout(yaxis_autorange="reversed")
            chart_card("da_corr", "Correlation matrix", fig, 330)





def db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS analysis_runs(run_id TEXT PRIMARY KEY, dataset_name TEXT, created_at TEXT,
       total_count INTEGER, suspicious_count INTEGER, risk_summary TEXT, verified_fraud INTEGER, settings TEXT);
    CREATE TABLE IF NOT EXISTS cases(run_id TEXT, transaction_id TEXT, risk_score REAL, severity TEXT,
       reasons TEXT, rule_flag INTEGER, ml_prediction INTEGER, verified_label INTEGER,
       status TEXT DEFAULT 'Pending Review', notes TEXT DEFAULT '', updated_at TEXT,
       PRIMARY KEY(run_id, transaction_id));
    CREATE TABLE IF NOT EXISTS alerts(run_id TEXT, transaction_id TEXT, severity TEXT, reasons TEXT,
       created_at TEXT, is_read INTEGER DEFAULT 0, dismissed INTEGER DEFAULT 0,
       PRIMARY KEY(run_id, transaction_id));
    """)
    return conn


def configuration():
    cfg = {"weights": RULE_POINTS.copy(), "high_amount_quantile": .95,
           "frequency_count": 3, "failed_attempts_threshold": 3, "rare_location_share": .05,
           "unusual_hours": [0, 1, 2, 3, 4, 5], "flag_threshold": 20,
           "risk_categories": {"Medium": 20, "High": 40, "Critical": 70}}
    if RULE_PATH.exists():
        try:
            with RULE_PATH.open() as f: cfg.update(json.load(f))
        except (ValueError, OSError) as e: st.warning(f"Unable to load risk rules: {e}")
    cfg.update(st.session_state.get("rules_override", {}))
    return cfg


def active_identity(df):
    # Content fingerprint, not just filename/size; changes invalidate cached results.
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:20]


def screen(df, cfg):
    r = df.copy()
    hits = {}
    w = cfg["weights"]
    hits["Unusually high amount"] = r.amount >= r.amount.quantile(float(cfg["high_amount_quantile"]))
    if "sender_id" in r:
        hits["Many transactions in a short period"] = (r.groupby(["sender_id", "trans_hour"], dropna=False)["trans_id"]
                  .transform("size") >= int(cfg["frequency_count"]))
    if "failed_attempts" in r:
        hits["Repeated failed attempts"] = pd.to_numeric(r.failed_attempts, errors="coerce").fillna(0) >= int(cfg["failed_attempts_threshold"])
    if "location" in r:
        hits["Unusual location"] = r.location.map(r.location.value_counts(normalize=True, dropna=False)).fillna(0) < float(cfg["rare_location_share"])
    if {"sender_id", "receiver_id"}.issubset(r):
        hits["Duplicate payment pattern"] = (r.assign(_rounded_amount=r.amount.round(2))
             .groupby(["sender_id", "receiver_id", "_rounded_amount"], dropna=False)["trans_id"].transform("size") >= 2)
    if "Unusual transaction hour" in w:
        hits["Unusual transaction hour"] = r.trans_hour.isin(cfg.get("unusual_hours", []))
    score = pd.Series(0, index=r.index, dtype=float)
    reason = pd.Series('', index=r.index, dtype=str)
    for name, hit in hits.items():
        hit = hit.fillna(False)
        score += hit.astype(int) * float(w.get(name, 0))
        reason = reason.mask(hit, reason + name + '; ')
    r['rule_score'] = score.clip(0,100).round().astype(int)
    r['rules_triggered'] = reason.str.rstrip('; ')
    bounds = cfg['risk_categories']
    r['risk_level'] = np.select([score >= bounds['Critical'], score >= bounds['High'], score >= bounds['Medium']],
                                 ['Critical', 'High', 'Medium'], default='Low')
    r['rule_flag'] = score >= float(cfg['flag_threshold'])
    r['ml_prediction'] = pd.Series([pd.NA] * len(r), index=r.index, dtype='Int64')
    r['ml_score'] = np.nan
    r['ml_status'] = 'Model unavailable'
    r['suspicious'] = r.rule_flag
    # The category below describes heuristic rule risk only, not model confidence.
    r['risk_score'] = r.rule_score # heuristic points, NOT probability
    return r


def active_results():
    df, source = get_active_dataset()
    if df is None: return None, source, None
    cfg = configuration()
    identity = active_identity(df)
    fingerprint = (identity, json.dumps(cfg, sort_keys=True), 'rules-only')
    if st.session_state.get('_result_key') != fingerprint:
        st.session_state['_results'] = screen(df, cfg)
        st.session_state['_result_key'] = fingerprint
    # Restore matching persisted run after a new browser session/app restart.
    cfg_json = json.dumps(cfg, sort_keys=True)
    matching_run = hashlib.sha256((identity + cfg_json + 'rules-only').encode()).hexdigest()[:24]
    if st.session_state.get('_active_fingerprint') != fingerprint:
        st.session_state.pop('active_run', None)
        st.session_state['_active_fingerprint'] = fingerprint
    if not st.session_state.get('active_run'):
        with closing(db()) as con:
            exists = con.execute('SELECT 1 FROM analysis_runs WHERE run_id=?', (matching_run,)).fetchone()
        if exists:
            st.session_state['active_run'] = matching_run
    return st.session_state['_results'], source, identity


def save_run(res, source, identity):
    cfg = configuration()
    settings = json.dumps(cfg, sort_keys=True)
    run_id = hashlib.sha256((identity + settings + 'rules-only').encode()).hexdigest()[:24]
    counts = {k: int((res.risk_level == k).sum()) for k in ['Low','Medium','High','Critical']}
    now = datetime.now().isoformat(timespec='seconds')
    labelled = 'is_fraud' in res
    with closing(db()) as con, con:
        exists = con.execute('SELECT 1 FROM analysis_runs WHERE run_id=?', (run_id,)).fetchone()
        if not exists:
            con.execute('INSERT INTO analysis_runs VALUES (?,?,?,?,?,?,?,?)',
                (run_id, source, now, len(res), int(res.suspicious.sum()), json.dumps(counts),
                 int(res.is_fraud.sum()) if labelled else None, settings))
            flagged = res[res.suspicious]
            con.executemany("""INSERT INTO cases(run_id,transaction_id,risk_score,severity,reasons,rule_flag,ml_prediction,verified_label,status,notes,updated_at)
                 VALUES(?,?,?,?,?,?,?,?,'Pending Review','',?)""", [
                (run_id, str(x.trans_id), float(x.risk_score), x.risk_level, x.rules_triggered or 'Rule score threshold met',
                 int(x.rule_flag), int(x.ml_prediction) if pd.notna(x.ml_prediction) else None,
                 int(x.is_fraud) if labelled else None, now) for x in flagged.itertuples()])
            con.executemany('INSERT INTO alerts(run_id,transaction_id,severity,reasons,created_at) VALUES(?,?,?,?,?)', [
                (run_id, str(x.trans_id), x.risk_level, x.rules_triggered or 'Rule score threshold met', now)
                for x in flagged.itertuples() if x.risk_level in ('High','Critical') or (pd.notna(x.ml_prediction) and x.ml_prediction == 1)])
    st.session_state['active_run'] = run_id
    return run_id


def query(sql, params=()):
    with closing(db()) as con:
        return pd.read_sql_query(sql, con, params=params)


def page_overview():
    page_header('Overview Dashboard', 'Live results from the currently selected dataset — never the training summary.')
    r, source, identity = active_results()
    if r is None:
        st.info('Upload a CSV on Dataset Analysis to start.'); return
    st.caption(f'Active dataset: {source} · {len(r):,} valid transactions')
    if not st.session_state.get('active_run'):
        st.caption('Open Fraud Detection and save an analysis run to populate persistent alerts and reviews.')
    run = st.session_state.get('active_run')
    cases = query('SELECT status FROM cases WHERE run_id=?', (run,)) if run else pd.DataFrame()
    alerts = query('SELECT * FROM alerts WHERE run_id=? ORDER BY created_at DESC', (run,)) if run else pd.DataFrame()
    unread = int(((alerts.is_read == 0) & (alerts.dismissed == 0)).sum()) if not alerts.empty else 0
    pending = int((cases.status == 'Pending Review').sum()) if not cases.empty else 0
    c=st.columns(4)
    for col, label, val in zip(c, ['Transactions','Suspicious (rules or ML)','Unread alerts','Pending reviews'],
                                [len(r),int(r.suspicious.sum()),unread,pending]):
        with col: kpi(label,f'{val:,}')
    c=st.columns(4)
    for col, level in zip(c,['Low','Medium','High','Critical']):
        with col:kpi(level+' risk',f'{int((r.risk_level==level).sum()):,}')
    if 'is_fraud' in r:
        st.metric('Verified fraud labels in active CSV',int(r.is_fraud.sum()))
        st.caption('Verified labels are dataset ground truth; alerts and ML predictions are not confirmations.')
    else: st.info('No verified fraud statistics: this dataset has no valid is_fraud column.')
    c1,c2=st.columns(2)
    with c1:chart_card('risk_counts','Rule risk categories',px.bar(r.risk_level.value_counts().reindex(['Low','Medium','High','Critical'],fill_value=0).reset_index(),x='risk_level',y='count'))
    with c2:chart_card('type_counts','Transactions by payment type',px.bar(r.payment_type.value_counts().reset_index(),x='payment_type',y='count'))
    st.subheader('Integrated alerts')
    if alerts.empty:st.caption('No stored alerts for this analysis. Save an analysis run on Fraud Detection.')
    else:
        visible=alerts[alerts.dismissed==0]
        st.dataframe(visible[['transaction_id','severity','reasons','created_at','is_read']],hide_index=True)
        if not visible.empty:
            selected=st.selectbox('Manage alert',visible.transaction_id.tolist())
            a=visible.loc[visible.transaction_id==selected].iloc[0]
            c=st.columns(3)
            for col,title,read,dismiss in [(c[0],'Mark read',1,0),(c[1],'Mark unread',0,0),(c[2],'Dismiss',int(a.is_read),1)]:
                if col.button(title,key=f'alert_{title}'):
                    with closing(db()) as con,con:con.execute('UPDATE alerts SET is_read=?,dismissed=? WHERE run_id=? AND transaction_id=?',(read,dismiss,run,selected))
                    st.rerun()


def page_detection():
    page_header('Fraud Detection', 'Identify suspicious transactions using configurable rules and clear risk scores.')
    r, source, identity = active_results()
    if r is None:
        st.info('Upload a CSV on Dataset Analysis to begin.'); return
    cfg = configuration()
    st.caption(f'Active dataset: {source} · {len(r):,} valid transactions')

    with st.expander('Configure detection rules', expanded=False):
        st.caption('Rule points are heuristic indicators, not verified fraud probabilities.')
        threshold = st.slider('Flag when rule points reach', 0, 100, int(cfg['flag_threshold']))
        quant = st.slider('High-amount percentile', 50, 99, int(float(cfg['high_amount_quantile'])*100))
        freq = st.number_input('Minimum transactions per sender in the same hour', 2, 100, int(cfg['frequency_count']))
        failed = st.number_input('Minimum failed attempts', 1, 100, int(cfg['failed_attempts_threshold']))
        rare = st.slider('Rare location share (%)', 1, 30, int(float(cfg['rare_location_share'])*100))
        selected_hours = st.multiselect('Unusual hours', list(range(24)), default=cfg.get('unusual_hours', [0,1,2,3,4,5]))
        weights = {}
        st.markdown('**Turn off a rule by setting its points to zero.**')
        for name, points in cfg['weights'].items():
            weights[name] = st.number_input(f'Points · {name}', 0, 100, int(points), key='weight_'+name)
        if st.button('Apply rule settings', type='primary'):
            st.session_state['rules_override'] = {
                'flag_threshold':threshold, 'high_amount_quantile':quant/100,
                'frequency_count':freq, 'failed_attempts_threshold':failed,
                'rare_location_share':rare/100, 'unusual_hours':selected_hours,
                'weights':weights}
            st.session_state.pop('active_run', None)
            st.rerun()
        if st.button('Reset to exported rule settings'):
            st.session_state.pop('rules_override', None)
            st.session_state.pop('active_run', None)
            st.rerun()

    st.subheader('Explainable rule-based detection')
    c=st.columns(4)
    for col,title,value in zip(c,['Transactions checked','Flagged by rules','High risk','Critical risk'],
        [len(r),int(r.rule_flag.sum()),int((r.risk_level=='High').sum()),int((r.risk_level=='Critical').sum())]):
        with col: kpi(title,f'{value:,}')
    activated=[]
    requirements = {
        'Unusually high amount':'amount',
        'Many transactions in a short period':'sender_id + trans_hour',
        'Repeated failed attempts':'failed_attempts',
        'Unusual location':'location',
        'Duplicate payment pattern':'sender_id + receiver_id',
        'Unusual transaction hour':'trans_hour'}
    for name, pts in cfg['weights'].items():
        required = [x.strip() for x in requirements.get(name,'').split('+')]
        available = all(x in r.columns for x in required)
        activated.append({'Rule':name,'Points':pts,'Requires':requirements.get(name,''),
          'Availability':'Active' if available and pts else ('Disabled' if not pts else 'Column not supplied')})
    st.dataframe(pd.DataFrame(activated),hide_index=True)
    byrisk = r.risk_level.value_counts().reindex(['Low','Medium','High','Critical'],fill_value=0).reset_index()
    fig = px.bar(byrisk,x='risk_level',y='count',color='risk_level',
        color_discrete_map={'Low':'#087F71','Medium':'#A76A00','High':'#D26A11','Critical':'#C7354D'})
    chart_card('rules_severity','Risk categories from rule points',fig)
    cols=['trans_id','payment_type','amount','rule_score','risk_level','rule_flag','rules_triggered']
    if 'is_fraud' in r:cols.append('is_fraud')
    st.dataframe(r[cols].sort_values('rule_score',ascending=False),hide_index=True,height=400)
    st.download_button('Download rule-based results CSV',r[cols].to_csv(index=False).encode(),'rule_detection.csv','text/csv')
    st.divider()
    st.subheader('Suspicious transactions flagged by rules')
    st.caption(f'{int(r.rule_flag.sum()):,} transactions flagged by the active rules. Flags are not confirmed fraud.')
    st.download_button('Export all suspicious transactions',r[r.rule_flag].to_csv(index=False).encode(),'suspicious_transactions.csv','text/csv')
    if st.button('Save analysis run, review queue and alerts',type='primary'):
        rule_results = r.copy()
        rule_results['suspicious'] = rule_results['rule_flag']
        rule_results['ml_prediction'] = pd.Series(pd.NA, index=rule_results.index, dtype='Int64')
        run=save_run(rule_results,source,identity)
        st.success(f'Analysis saved ({run}). Review the flagged transactions in Fraud Review Center.')


def page_reviews():
    page_header('Fraud Review Center','Human review decisions are saved to SQLite and never inferred from flags.')
    run=st.session_state.get('active_run')
    if not run:st.info('Save an analysis run in Fraud Detection first.');return
    cases=query('SELECT * FROM cases WHERE run_id=? ORDER BY risk_score DESC',(run,))
    if cases.empty:st.info('There are no flagged cases in this run.');return
    selected_status=st.multiselect('Filter status',DECISIONS,default=DECISIONS)
    shown=cases[cases.status.isin(selected_status)]
    st.dataframe(shown[['transaction_id','risk_score','severity','reasons','status','notes','updated_at']],hide_index=True)
    if shown.empty:return
    trans=st.selectbox('Transaction to review',shown.transaction_id.tolist())
    row=shown[shown.transaction_id==trans].iloc[0]
    st.caption(f"Severity: {row.severity} | Rules: {row.reasons or 'Rule flag'} | Verified label: {row.verified_label if pd.notna(row.verified_label) else 'Not available'}")
    with st.form('review_case'):
        status=st.selectbox('Review status',DECISIONS,index=DECISIONS.index(row.status))
        notes=st.text_area('Reviewer notes',value=row.notes or '')
        if st.form_submit_button('Save review',type='primary'):
            with closing(db()) as con,con:
                con.execute('UPDATE cases SET status=?,notes=?,updated_at=? WHERE run_id=? AND transaction_id=?',
                    (status,notes,datetime.now().isoformat(timespec='seconds'),run,trans))
            st.success('Review saved.');st.rerun()


def page_history():
    page_header('History & Reports','Saved analysis runs, reviewer decisions and CSV exports.')
    runs=query('SELECT * FROM analysis_runs ORDER BY created_at DESC')
    if runs.empty:st.info('No runs saved yet.');return
    st.dataframe(runs[['run_id','dataset_name','created_at','total_count','suspicious_count','verified_fraud','risk_summary']],hide_index=True)
    run=st.selectbox('Analysis run',runs.run_id.tolist())
    row=runs[runs.run_id==run].iloc[0]
    st.json(json.loads(row.risk_summary))
    cases=query('SELECT * FROM cases WHERE run_id=?',(run,))
    alerts=query('SELECT * FROM alerts WHERE run_id=?',(run,))
    st.subheader('Review decisions and notes')
    st.dataframe(cases,hide_index=True)
    st.download_button('Export reviewed cases CSV',cases[cases.status!='Pending Review'].to_csv(index=False).encode(),f'reviews_{run}.csv','text/csv')
    st.download_button('Export suspicious case summary CSV',cases.to_csv(index=False).encode(),f'suspicious_{run}.csv','text/csv')
    summary=pd.DataFrame([{'run_id':run,'dataset':row.dataset_name,'date':row.created_at,'transactions':row.total_count,
       'suspicious':row.suspicious_count,'verified_fraud_if_labelled':row.verified_fraud,**json.loads(row.risk_summary)}])
    st.download_button('Export analysis summary CSV',summary.to_csv(index=False).encode(),f'analysis_{run}.csv','text/csv')
    st.caption(f'Alerts stored for selected run: {len(alerts):,}')


def main():
    st.set_page_config(page_title='Fraud Analysis',page_icon='🛡️',layout='wide',initial_sidebar_state='expanded')
    st.markdown(CSS,unsafe_allow_html=True)
    db().close()
    page=render_sidebar()
    {'Overview Dashboard':page_overview,'Dataset Analysis':page_dataset,'Fraud Detection':page_detection,
     'Fraud Review Center':page_reviews,'History & Reports':page_history}[page]()

if __name__=='__main__':main()
