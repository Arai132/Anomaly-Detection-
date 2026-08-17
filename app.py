"""Streamlit UI for interactive anomaly detection with dataset upload and method recommendation."""
import io
import json
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from src.recommender import NATURE_TAXONOMY, DETECTOR_MAP, recommend, auto_detect_natures
from src.detectors import (
    IsolationForestDetector, LOFDetector, ZScoreDetector, IQRDetector,
    AutoencoderDetector, VAEDetector, GaussianDetector, EnsembleStackingDetector,
    EllipticEnvelopeDetector, OneClassSVMDetector, PCADetector,
    RollingZScoreDetector, STLDetector,
)

st.set_page_config(
    page_title="AnomalyDetection — Smart Recommender",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  .stAlert { border-radius: 8px; }
  div[data-testid="stMetricValue"] { font-size: 1.6rem; }
  .rec-card {
    background: #1e2130; border: 1px solid #2e3248; border-radius: 10px;
    padding: 1rem 1.2rem; margin-bottom: .8rem;
  }
  .score-bar-wrap { background: #2e3248; border-radius: 6px; height: 8px; margin: .4rem 0; }
  .score-bar { background: #7c6af7; height: 8px; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ─────────────────────────────────────────────────────
for key, val in {
    "df": None, "col_target": None, "selections": {}, "results": {},
    "auto_natures": {}, "run_ids": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_data(uploaded) -> pd.DataFrame | None:
    name = uploaded.name.lower()
    raw = uploaded.read()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(io.BytesIO(raw))
        if name.endswith(".json"):
            obj = json.loads(raw)
            if isinstance(obj, list):
                return pd.DataFrame(obj)
            return pd.DataFrame([obj]) if isinstance(obj, dict) else pd.json_normalize(obj)
        if name.endswith((".txt", ".dat", ".tsv")):
            sep = "\t" if name.endswith(".tsv") else r"\s+"
            return pd.read_csv(io.BytesIO(raw), sep=sep, header=None,
                               names=[f"col_{i}" for i in range(100)]).dropna(axis=1, how="all")
    except Exception as e:
        st.error(f"Could not parse file: {e}")
    return None


def parse_pasted(text: str) -> pd.DataFrame | None:
    text = text.strip()
    if not text:
        return None
    try:
        rows = []
        for line in text.splitlines():
            line = line.strip().replace(",", " ")
            vals = [float(v) for v in line.split()]
            if vals:
                rows.append(vals)
        if not rows:
            return None
        max_cols = max(len(r) for r in rows)
        padded = [r + [np.nan] * (max_cols - len(r)) for r in rows]
        cols = [f"col_{i}" for i in range(max_cols)]
        return pd.DataFrame(padded, columns=cols).dropna(axis=1, how="all")
    except Exception:
        return None


def get_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include="number")


def build_detector(det_id: str, params: dict, contamination: float):
    """Instantiate the correct detector class from id + params."""
    p = {**params}
    if "contamination" in p:
        p["contamination"] = contamination

    mapping = {
        "zscore":           lambda: ZScoreDetector(threshold=p.get("threshold", 3.0)),
        "iqr":              lambda: IQRDetector(multiplier=p.get("multiplier", 1.5)),
        "isolation_forest": lambda: IsolationForestDetector(
                                n_estimators=int(p.get("n_estimators", 100)),
                                contamination=contamination),
        "lof":              lambda: LOFDetector(
                                n_neighbors=int(p.get("n_neighbors", 20)),
                                contamination=contamination),
        "autoencoder":      lambda: AutoencoderDetector(
                                encoding_dim=int(p.get("encoding_dim", 16)),
                                epochs=int(p.get("epochs", 50)),
                                threshold_percentile=p.get("threshold_percentile", 95)),
        "vae":              lambda: VAEDetector(
                                latent_dim=int(p.get("latent_dim", 16)),
                                epochs=int(p.get("epochs", 50)),
                                beta=p.get("beta", 1.0),
                                threshold_percentile=p.get("threshold_percentile", 95)),
        "gaussian":         lambda: GaussianDetector(
                                n_components=int(p.get("n_components", 1)),
                                threshold_percentile=p.get("threshold_percentile", 95)),
        "elliptic_envelope":lambda: EllipticEnvelopeDetector(contamination=contamination),
        "one_class_svm":    lambda: OneClassSVMDetector(
                                nu=contamination,
                                kernel=p.get("kernel", "rbf")),
        "pca":              lambda: PCADetector(
                                n_components=p.get("n_components", 0.95),
                                threshold_percentile=p.get("threshold_percentile", 95)),
        "rolling_zscore":   lambda: RollingZScoreDetector(
                                window=int(p.get("window", 20)),
                                threshold=p.get("threshold", 3.0)),
        "stl":              lambda: STLDetector(
                                period=int(p.get("period", 12)),
                                threshold_percentile=p.get("threshold_percentile", 95)),
        "ensemble_stacking":lambda: EnsembleStackingDetector(
                                detectors=[
                                    AutoencoderDetector(encoding_dim=16, epochs=30),
                                    GaussianDetector(),
                                    IsolationForestDetector(contamination=contamination),
                                ],
                                threshold_percentile=p.get("threshold_percentile", 95)),
    }
    return mapping[det_id]()


def run_detector(det_id: str, params: dict, X: np.ndarray, contamination: float,
                 y_true: np.ndarray | None = None):
    """Fit on 80% train, predict on full dataset. Returns (predictions, scores, metrics)."""
    detector = build_detector(det_id, params, contamination)
    n = len(X)
    split = max(10, int(n * 0.8))
    X_train, X_test = X[:split], X[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_full_s = scaler.transform(X)

    detector.fit(X_train_s)

    preds = detector.predict(X_full_s)
    try:
        scores = detector.score_samples(X_full_s)
    except NotImplementedError:
        scores = preds.astype(float)

    metrics = {}
    if y_true is not None and len(np.unique(y_true)) > 1:
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        metrics["precision"] = round(precision_score(y_true, preds, zero_division=0), 3)
        metrics["recall"]    = round(recall_score(y_true, preds, zero_division=0), 3)
        metrics["f1"]        = round(f1_score(y_true, preds, zero_division=0), 3)
        try:
            metrics["roc_auc"] = round(roc_auc_score(y_true, scores), 3)
        except Exception:
            pass

    metrics["n_anomalies"] = int(preds.sum())
    metrics["anomaly_rate"] = f"{preds.mean()*100:.1f}%"
    return preds, scores, metrics


def score_plot(scores: np.ndarray, preds: np.ndarray, y_true: np.ndarray | None,
               title: str, index=None) -> go.Figure:
    x = index if index is not None else list(range(len(scores)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=scores, mode="lines",
                             line=dict(color="#7c6af7", width=1.2), name="Score", opacity=0.8))
    anom_idx = np.where(preds == 1)[0]
    if len(anom_idx):
        fig.add_trace(go.Scatter(
            x=[x[i] for i in anom_idx], y=scores[anom_idx],
            mode="markers", marker=dict(color="#f97316", size=7, symbol="circle"),
            name="Detected anomaly"))
    if y_true is not None:
        true_idx = np.where(y_true == 1)[0]
        if len(true_idx):
            fig.add_trace(go.Scatter(
                x=[x[i] for i in true_idx], y=scores[true_idx],
                mode="markers", marker=dict(color="#ec4899", size=9, symbol="x"),
                name="True anomaly"))
    fig.update_layout(title=title, height=320,
                      plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                      font=dict(color="#e2e8f0"),
                      xaxis=dict(gridcolor="#2e3248"),
                      yaxis=dict(gridcolor="#2e3248"),
                      legend=dict(bgcolor="#1a1d27"),
                      margin=dict(l=40, r=20, t=40, b=30))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.title("🔍 AnomalyDetection — Smart Recommender")
st.caption("Upload your data, describe its nature, get method recommendations, and run detection interactively.")

tab_upload, tab_describe, tab_results = st.tabs(
    ["① Upload Data", "② Describe & Recommend", "③ Run & Results"]
)


# ══════════════════════════════════ TAB 1: UPLOAD ═════════════════════════════
with tab_upload:
    st.subheader("Load your dataset")
    col_up, col_paste = st.columns([1, 1], gap="large")

    with col_up:
        st.markdown("**Upload a file**")
        uploaded = st.file_uploader(
            "CSV, JSON, TXT, TSV, or DAT",
            type=["csv", "json", "txt", "tsv", "dat"],
            label_visibility="collapsed",
        )
        if uploaded:
            df = load_data(uploaded)
            if df is not None:
                st.session_state.df = df
                st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

    with col_paste:
        st.markdown("**Or paste numbers directly**")
        pasted = st.text_area(
            "One value per line, or space/comma-separated rows",
            height=140, label_visibility="collapsed",
            placeholder="1.2  0.9  1.1\n15.3 0.8  1.0\n..."
        )
        if st.button("Parse pasted data", use_container_width=True):
            df = parse_pasted(pasted)
            if df is not None:
                st.session_state.df = df
                st.success(f"Parsed {len(df):,} rows × {len(df.columns)} columns")
            else:
                st.error("Could not parse the pasted text.")

    # ── Preview ───────────────────────────────────────────────────────────────
    if st.session_state.df is not None:
        df = st.session_state.df
        st.divider()
        st.markdown("### Preview")

        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        pcol1.metric("Rows", f"{len(df):,}")
        pcol2.metric("Columns", len(df.columns))
        pcol3.metric("Numeric cols", len(get_numeric(df).columns))
        pcol4.metric("Missing values", int(df.isna().sum().sum()))

        st.dataframe(df.head(50), use_container_width=True, height=220)

        # Optional: label column selection
        num_cols = get_numeric(df).columns.tolist()
        all_cols = ["(none — no labels)"] + df.columns.tolist()
        label_col = st.selectbox(
            "Optional: select a label/target column if you have anomaly ground-truth (0=normal, 1=anomaly)",
            all_cols, index=0,
        )
        st.session_state.col_target = None if label_col == "(none — no labels)" else label_col

        # Distribution preview — first numeric col
        if num_cols:
            st.markdown("#### Distribution preview")
            preview_col = st.selectbox("Column to preview", num_cols, index=0)
            hist_fig = px.histogram(
                df, x=preview_col, nbins=50,
                color_discrete_sequence=["#7c6af7"],
                template="plotly_dark",
            )
            hist_fig.update_layout(height=260, margin=dict(l=30, r=10, t=30, b=30),
                                   plot_bgcolor="#0f1117", paper_bgcolor="#0f1117")
            st.plotly_chart(hist_fig, use_container_width=True)

        # Auto-detect natures
        if st.button("🔎 Auto-detect data characteristics", use_container_width=True):
            with st.spinner("Analysing…"):
                natures = auto_detect_natures(df)
                st.session_state.auto_natures = natures
            st.success("Auto-detection complete — see ② Describe & Recommend to review and refine.")


# ══════════════════════════════ TAB 2: DESCRIBE ═══════════════════════════════
with tab_describe:
    if st.session_state.df is None:
        st.info("Upload your data in tab ① first.")
        st.stop()

    df = st.session_state.df
    auto = st.session_state.auto_natures

    st.subheader("Describe your data's nature")
    st.caption("Select all characteristics that apply. Auto-detected suggestions are pre-filled where available.")

    contamination = st.slider(
        "Expected anomaly rate (contamination)", 0.01, 0.30, 0.05, 0.01,
        format="%.2f",
        help="Approximate fraction of anomalies you expect. Used to calibrate thresholds.",
    )

    selections: dict[str, list[str] | str] = {}

    # Render each nature category
    for cat_id, cat_info in NATURE_TAXONOMY.items():
        st.markdown(f"**{cat_info['label']}** — {cat_info['help']}")
        options = cat_info["options"]
        auto_val = auto.get(cat_id, [] if not cat_info["single"] else None)

        if cat_info["single"]:
            keys = list(options.keys())
            labels = list(options.values())
            default_idx = keys.index(auto_val) if auto_val in keys else 0
            chosen = st.radio(cat_id, labels, index=default_idx,
                              horizontal=True, label_visibility="collapsed",
                              key=f"radio_{cat_id}")
            selections[cat_id] = keys[labels.index(chosen)]
        else:
            default_checked = set(auto_val) if isinstance(auto_val, list) else set()
            chosen_keys = []
            cols = st.columns(min(len(options), 4))
            for i, (k, v) in enumerate(options.items()):
                with cols[i % len(cols)]:
                    if st.checkbox(v, value=(k in default_checked), key=f"cb_{cat_id}_{k}"):
                        chosen_keys.append(k)
            selections[cat_id] = chosen_keys

        st.markdown("")  # spacing

    st.session_state.selections = selections
    st.session_state["contamination"] = contamination

    # ── Recommendations ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Recommended methods")
    st.caption("Ranked by suitability for your selected data characteristics.")

    ranked = recommend(selections, top_n=6)
    max_score = ranked[0][1] if ranked and ranked[0][1] > 0 else 1

    for det, score in ranked:
        pct = max(0, min(100, int(score / max(max_score, 1) * 100)))
        badge_color = "#22c55e" if pct >= 70 else "#f97316" if pct >= 40 else "#94a3b8"

        with st.expander(f"**{det.name}** — {det.short}", expanded=(pct >= 70)):
            ecol1, ecol2 = st.columns([2, 1])
            with ecol1:
                st.markdown(det.description)
                st.markdown("**Strengths:** " + " · ".join(f"`{s}`" for s in det.strengths))
                st.markdown("**Limitations:** " + " · ".join(f"`{w}`" for w in det.weaknesses))
            with ecol2:
                st.markdown(f"**Suitability score**")
                st.markdown(
                    f'<div class="score-bar-wrap"><div class="score-bar" style="width:{pct}%"></div></div>',
                    unsafe_allow_html=True,
                )
                st.caption(f"{pct}% match for your selections")
                if det.time_series_only:
                    st.info("⏱ Time series only")


# ══════════════════════════════ TAB 3: RESULTS ════════════════════════════════
with tab_results:
    if st.session_state.df is None:
        st.info("Upload your data in tab ① first.")
        st.stop()

    df = st.session_state.df
    num_df = get_numeric(df)
    if num_df.empty:
        st.error("No numeric columns found. Cannot run detectors.")
        st.stop()

    selections = st.session_state.get("selections", {})
    contamination = st.session_state.get("contamination", 0.05)

    # Recommend again (in case user skipped tab 2)
    ranked = recommend(selections, top_n=6) if selections else [(DETECTOR_MAP[d], 0) for d in
                                                                 ["isolation_forest", "zscore", "iqr"]]

    st.subheader("Run detectors")

    # Target column
    target_col = st.session_state.get("col_target")
    y_true = None
    if target_col and target_col in df.columns:
        y_true = df[target_col].values.astype(int)
        st.success(f"Using `{target_col}` as ground-truth labels.")

    # Feature columns
    feature_candidates = [c for c in num_df.columns if c != target_col]
    selected_features = st.multiselect(
        "Feature columns to use",
        feature_candidates,
        default=feature_candidates,
    )
    if not selected_features:
        st.warning("Select at least one feature column.")
        st.stop()

    X = num_df[selected_features].fillna(num_df[selected_features].median()).values

    # Detector selection + param tuning
    st.markdown("**Select detectors to run and tune their parameters:**")

    to_run: list[tuple[str, dict]] = []
    for det, score in ranked:
        with st.expander(f"{det.name}", expanded=False):
            col_check, col_params = st.columns([1, 3])
            with col_check:
                run_it = st.checkbox("Run this detector", value=True, key=f"run_{det.id}")
            with col_params:
                tuned_params = {}
                param_cols = st.columns(min(len(det.params), 3))
                for i, (pname, pdefault) in enumerate(det.params.items()):
                    with param_cols[i % len(param_cols)]:
                        if isinstance(pdefault, float) and pname in ("threshold", "multiplier", "nu"):
                            tuned_params[pname] = st.number_input(pname, 0.01, 10.0, float(pdefault), 0.1, key=f"{det.id}_{pname}")
                        elif isinstance(pdefault, float) and pname == "n_components":
                            tuned_params[pname] = st.slider(pname, 0.5, 1.0, float(pdefault), 0.05, key=f"{det.id}_{pname}")
                        elif isinstance(pdefault, float):
                            tuned_params[pname] = st.number_input(pname, 0.0, 100.0, float(pdefault), 1.0, key=f"{det.id}_{pname}")
                        elif isinstance(pdefault, int):
                            tuned_params[pname] = st.number_input(pname, 1, 1000, int(pdefault), 1, key=f"{det.id}_{pname}")
                        else:
                            tuned_params[pname] = st.text_input(pname, str(pdefault), key=f"{det.id}_{pname}")
            if run_it:
                to_run.append((det.id, tuned_params))

    # ── Run ───────────────────────────────────────────────────────────────────
    if st.button("▶ Run selected detectors", type="primary", use_container_width=True):
        results = {}
        progress = st.progress(0, text="Running detectors…")

        for i, (det_id, params) in enumerate(to_run):
            det_spec = DETECTOR_MAP[det_id]
            progress.progress((i) / len(to_run), text=f"Running {det_spec.name}…")
            try:
                preds, scores, metrics = run_detector(det_id, params, X, contamination, y_true)
                results[det_id] = {"preds": preds, "scores": scores, "metrics": metrics, "name": det_spec.name}
            except Exception as e:
                st.warning(f"{det_spec.name} failed: {e}")

        progress.progress(1.0, text="Done!")
        st.session_state.results = results

    # ── Display results ───────────────────────────────────────────────────────
    results = st.session_state.get("results", {})
    if not results:
        st.info("Configure detectors above and click ▶ Run.")
        st.stop()

    st.divider()
    st.subheader("Results")

    # Metrics comparison table
    if len(results) > 1:
        st.markdown("#### Detector comparison")
        rows = []
        for det_id, res in results.items():
            row = {"Detector": res["name"]} | res["metrics"]
            rows.append(row)
        comp_df = pd.DataFrame(rows).set_index("Detector")
        st.dataframe(comp_df.style.highlight_max(axis=0, subset=[c for c in comp_df.columns
                                                                   if c not in ("anomaly_rate",)],
                                                  color="#7c6af730"),
                     use_container_width=True)

    # Per-detector score plots + anomaly table
    for det_id, res in results.items():
        st.markdown(f"#### {res['name']}")
        mcols = st.columns(len(res["metrics"]))
        for col, (k, v) in zip(mcols, res["metrics"].items()):
            col.metric(k.replace("_", " ").title(), v)

        fig = score_plot(
            res["scores"], res["preds"], y_true,
            title=f"{res['name']} — Anomaly Scores",
        )
        st.plotly_chart(fig, use_container_width=True)

        anom_mask = res["preds"] == 1
        if anom_mask.sum() > 0:
            anom_df = df.iloc[np.where(anom_mask)[0]].copy()
            anom_df.insert(0, "anomaly_score", res["scores"][anom_mask].round(4))
            anom_df = anom_df.sort_values("anomaly_score", ascending=False)
            with st.expander(f"Detected anomaly rows ({anom_mask.sum()})"):
                st.dataframe(anom_df.head(200), use_container_width=True)

        st.divider()

    # Download
    all_preds = pd.DataFrame({"index": range(len(X))})
    for det_id, res in results.items():
        all_preds[f"{res['name']}_score"] = res["scores"].round(4)
        all_preds[f"{res['name']}_pred"] = res["preds"]
    if y_true is not None:
        all_preds["y_true"] = y_true

    st.download_button(
        "⬇ Download predictions CSV",
        all_preds.to_csv(index=False).encode(),
        file_name="anomaly_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
