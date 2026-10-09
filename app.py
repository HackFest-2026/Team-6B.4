from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from medvision.pipeline import prepare_features

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "medvision_pipeline.joblib"

st.set_page_config(page_title="MEDVISION | Readmission Analytics", page_icon="🌿", layout="wide")
st.markdown("""
<style>
:root { color-scheme:light; --green:#08764e; --green-dark:#07563c; --ink:#18372c; --muted:#698075; }
[data-testid="stAppViewContainer"] { background:linear-gradient(180deg,#eff8f2 0,#f8faf8 300px,#f8faf8 100%); color:var(--ink) !important; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stMainBlockContainer"] { max-width:1480px; padding-top:1rem; }
body,.stApp { font-family:'DM Sans','Segoe UI',sans-serif; }
h1,h2,h3,h4 { font-family:'Manrope','Segoe UI',sans-serif !important; color:var(--ink); letter-spacing:-.02em; }
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer, .stDeployButton { display:none !important; }
.topbar { display:flex; align-items:center; justify-content:space-between; padding:13px 21px; margin:0 0 14px; border-radius:17px; color:white; background:linear-gradient(110deg,#07563c,#08764e 62%,#159766); box-shadow:0 10px 28px #08764e20; }
.brand { display:flex; align-items:center; gap:12px; font:800 18px 'Segoe UI',sans-serif; letter-spacing:.04em; }
.brand small { display:block; margin-top:2px; color:#dbf4e6; font:600 10px 'Segoe UI',sans-serif; letter-spacing:.1em; text-transform:uppercase; }
.logo-mark { width:42px; height:42px; display:grid; place-items:center; border:1px solid #ffffff80; border-radius:14px; background:#ffffff1c; font:800 15px 'Segoe UI',sans-serif; animation:logoPulse 3s ease-in-out infinite; }
@keyframes logoPulse { 0%,100% { transform:translateY(0); box-shadow:0 0 0 0 #b7f3d133; } 50% { transform:translateY(-2px); box-shadow:0 0 0 7px #b7f3d11c; } }
.hero { padding:20px 26px; margin-bottom:14px; border:1px solid #d9eee2; border-radius:19px; background:linear-gradient(120deg,#fff,#eef8f2); }
.hero h1 { margin:4px 0; font:800 27px 'Segoe UI',sans-serif; }
.hero p { margin:0; color:var(--muted); font-size:13px; }
.metric-card { min-height:108px; padding:16px; border:1px solid #dce9e1; border-radius:15px; background:#fff; box-shadow:0 4px 14px #18372c08; transition:border-color .18s,box-shadow .18s,transform .18s; }
.metric-card:hover { border-color:#21a66e; box-shadow:0 0 0 3px #16a36a22,0 12px 26px #08764e12; transform:translateY(-2px); }
.metric-label { color:#688074; font-size:10px; font-weight:700; letter-spacing:.09em; text-transform:uppercase; }
.metric-value { margin-top:8px; color:#143b2b; font:800 24px 'Segoe UI',sans-serif; }
.metric-note { margin-top:3px; color:#83958c; font-size:10px; }
[data-testid="stRadio"] [role="radiogroup"] { gap:7px; }
[data-testid="stRadio"] label { padding:7px 13px; border:1px solid #d8e8de; border-radius:999px; background:#fff !important; color:#18372c !important; font-weight:650; opacity:1 !important; }
[data-testid="stRadio"] label *, [data-testid="stRadio"] label:has(input:checked) { color:#07563c !important; }
[data-testid="stRadio"] label:has(input:checked) { border-color:#08764e; background:#e5f4eb !important; }
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *, [data-testid="stWidgetLabel"] p,
div[data-testid="stSelectbox"] label, div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label *, div[data-testid="stNumberInput"] label * { color:#18372c !important; opacity:1 !important; }
[data-baseweb="select"] > div { background:#fff !important; border-color:#bfd5c7 !important; color:#18372c !important; }
[data-baseweb="select"] *, [data-baseweb="input"] input { color:#18372c !important; }
[data-baseweb="input"] { background:#fff !important; border-color:#bfd5c7 !important; }
[data-testid="stNumberInput"] button { background:#f3faf5 !important; color:#08764e !important; }
[data-testid="stSelectbox"] [data-baseweb="popover"], [data-testid="stSelectbox"] [role="listbox"] { background:#fff !important; color:#18372c !important; }
[data-testid="stSelectbox"] [role="option"] { color:#18372c !important; }
[data-testid="stDataFrame"] { border:1px solid #dce9e1; border-radius:13px; overflow:hidden; }
div.stButton > button[kind="primary"], button[kind="primaryFormSubmit"] { background:var(--green); border:0; border-radius:10px; }
div.stButton > button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover { background:var(--green-dark); }
</style>
<div class="topbar"><div class="brand"><div class="logo-mark">MV</div><div>MEDVISION<small>Readmission analytics</small></div></div><div>Patient risk overview</div></div>
<div class="hero"><div style="color:#08764e;font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase">Hospital analytics</div><h1>Readmission risk dashboard</h1><p>Model performance, historical outcomes, and new encounter estimates.</p></div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.info("Train the model to load dashboard results.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
feature_columns = bundle["feature_columns"]
mappings = bundle.get("mappings", {})
baseline = bundle.get("explanation_baseline", {})
threshold = float(bundle.get("threshold", 0.5))
metrics = json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
historical = pd.read_csv(ARTIFACTS / "historical_predictions.csv")

for field, mapping in mappings.items():
    if field in historical:
        historical[field] = historical[field].map(lambda value: mapping.get(str(value), str(value)) if pd.notna(value) else "Unknown")
for column in ("age", "gender", "race", "diabetesmed", "insulin"):
    if column in historical:
        historical[column] = historical[column].fillna("Unknown").astype(str)
overall_actual = float(historical["actual_readmitted_within_30_days"].mean())
overall_prediction = historical["predicted_readmitted_within_30_days"].to_numpy(dtype=int)
overall_truth = historical["actual_readmitted_within_30_days"].to_numpy(dtype=int)


def group_statistics(frame: pd.DataFrame):
    y_true = frame["actual_readmitted_within_30_days"].to_numpy(dtype=int)
    y_pred = frame["predicted_readmitted_within_30_days"].to_numpy(dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    predicted = int((y_pred == 1).sum())
    return {
        "n": len(frame), "actual_rate": positives / len(frame) if len(frame) else 0,
        "recall": tp / positives if positives else 0,
        "false_alarm": fp / negatives if negatives else 0,
        "precision": tp / predicted if predicted else 0,
        "mean_risk": float(frame["risk_probability"].mean()) if len(frame) else 0,
    }


def render_fairness(profile: dict, key_prefix: str):
    st.subheader("Fairness context")
    st.caption("Aggregate holdout outcomes for people in similar demographic groups, compared with the full holdout.")
    overall = group_statistics(historical)
    dimensions = [("age", "Age band"), ("gender", "Gender"), ("race", "Race")]
    cards = st.columns(3)
    for col, (attribute, title) in zip(cards, dimensions):
        value = str(profile.get(attribute, "Unknown"))
        peers = historical[historical[attribute].fillna("Unknown").astype(str) == value]
        stats = group_statistics(peers)
        with col:
            with st.container(border=True):
                st.markdown(f"#### {title}: {value}")
                st.metric("Holdout encounters", f"{stats['n']:,}")
                st.metric("Observed readmission", f"{stats['actual_rate']:.1%}", f"{stats['actual_rate']-overall['actual_rate']:+.1%} vs all")
                st.metric("Model recall", f"{stats['recall']:.1%}", f"{stats['recall']-overall['recall']:+.1%} vs all")
                st.caption(f"False-alarm rate {stats['false_alarm']:.1%} · average model risk {stats['mean_risk']:.1%}")


def format_feature_name(name: str) -> str:
    labels = {
        "age": "Age band", "gender": "Gender", "race": "Race",
        "admission_type_id": "Admission type", "admission_source_id": "Admission source",
        "time_in_hospital": "Hospital days", "number_inpatient": "Prior inpatient visits",
        "number_emergency": "Prior emergency visits", "number_outpatient": "Prior outpatient visits",
        "number_diagnoses": "Number of diagnoses", "num_medications": "Number of medications",
        "a1cresult": "A1C result", "diabetesmed": "Diabetes medication", "insulin": "Insulin treatment",
        "diag_1": "Primary diagnosis (ICD-9)", "diag_2": "Secondary diagnosis (ICD-9)",
        "diag_3": "Additional diagnosis (ICD-9)", "max_glu_serum": "Maximum glucose result",
        "change": "Medication change during visit",
    }
    if name in labels:
        return labels[name]
    if name.startswith("diag_"):
        return f"Diagnosis {name[5:]} (ICD-9)"
    return name.replace("_", " ").capitalize()


def explain_estimate(raw_record, input_name: str):
    raw_frame = pd.DataFrame([raw_record])
    features = prepare_features(raw_frame, feature_columns, mappings)
    risk = float(pipeline.predict_proba(features)[0, 1])
    st.subheader("How this estimate was formed")
    st.caption("One-field comparisons against the training-set typical value. These are learned associations, not causes.")
    changes = []
    for feature in feature_columns:
        typical = baseline.get(feature)
        current = features.iloc[0][feature]
        if typical is None or pd.isna(current) or str(current) == str(typical):
            continue
        counterfactual = features.copy()
        counterfactual.loc[counterfactual.index[0], feature] = typical
        alternative = float(pipeline.predict_proba(counterfactual)[0, 1])
        changes.append((risk - alternative, feature))
    increases = sorted((item for item in changes if item[0] > 0.001), reverse=True)[:3]
    decreases = sorted((item for item in changes if item[0] < -0.001))[:3]
    with st.container(border=True):
        result_col, explain_col = st.columns([.8, 1.7])
        with result_col:
            level = "Higher" if risk >= .7 else "Moderate" if risk >= .3 else "Lower"
            st.metric("Estimated 30-day readmission risk", f"{risk:.1%}", level)
            st.progress(min(max(risk, 0.0), 1.0))
        with explain_col:
            factor_cols = st.columns(2)
            with factor_cols[0]:
                st.markdown("**Signals raising the estimate**")
                st.write("\n".join(f"• {format_feature_name(name)}" for _, name in increases) if increases else "No strong upward signal among entered fields.")
            with factor_cols[1]:
                st.markdown("**Signals lowering the estimate**")
                st.write("\n".join(f"• {format_feature_name(name)}" for _, name in decreases) if decreases else "No strong downward signal among entered fields.")
            st.caption("The compact form leaves some model inputs to training-fitted imputation.")
    return risk, features


page = st.radio("Main menu", ["Home", "Historical records", "New data input"], horizontal=True, label_visibility="collapsed", key="dashboard_page")

if page == "Home":
    st.subheader("Model breakthrough results")
    selected_metrics = metrics["selected_test_metrics"]
    st.caption(f"{metrics['selected_model']} · decision cutoff {threshold:.2f} · untouched patient-level test set ({metrics['test_rows']:,} encounters)")
    cards = st.columns(5)
    for col, (label, name, note) in zip(cards, [
        ("Accuracy", "accuracy", "Correct classifications"), ("Precision", "precision", "Positive predictions that were correct"),
        ("Recall", "recall", "Positive cases found"), ("F1 score", "f1", "Precision/recall balance"), ("AUC", "auc", "Ranking quality across cutoffs")]):
        with col:
            value = selected_metrics[name]
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value:.1%}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)
    target_col, baseline_col, target_note = st.columns([1.15, 1.15, 2])
    target_col.markdown("**90% accuracy goal**")
    target_col.caption("Reached" if selected_metrics["accuracy"] >= .9 else "Not reached")
    baseline_col.markdown("**Always-negative baseline**")
    baseline_col.caption(f"{metrics['test_majority_baseline_accuracy']:.1%}")
    target_note.caption(f"Positive cases are {metrics['test_positive_rate']:.1%} of the test set. Model choice used validation AUC; the displayed cutoff was selected on validation for F1.")

    st.subheader("Model comparison")
    comparison = metrics["validation_metrics_at_0_5"]
    names = list(comparison)
    compare = go.Figure()
    compare.add_trace(go.Bar(name="Accuracy", y=names, x=[comparison[n]["accuracy"] for n in names], orientation="h", marker_color="#83cba6"))
    compare.add_trace(go.Bar(name="AUC", y=names, x=[comparison[n]["auc"] for n in names], orientation="h", marker_color="#08764e"))
    compare.update_layout(template="plotly_white", barmode="group", height=360, margin={"l":12,"r":20,"t":42,"b":24},
                          xaxis={"range":[0,1],"tickformat":".0%","tickfont":{"color":"#18372c"},"gridcolor":"#e2ece5"},
                          yaxis={"autorange":"reversed","tickfont":{"color":"#18372c","size":12},"automargin":True},
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"family":"Segoe UI","color":"#18372c"},
                          legend={"orientation":"h","y":1.12,"font":{"color":"#18372c"}})
    st.plotly_chart(compare, width="stretch", config={"displayModeBar":False})

    st.subheader("Holdout distribution")
    chart_col, detail_col = st.columns([1, 1.15], gap="large")
    with chart_col:
        distribution = st.selectbox("Distribution", ["Risk group", "Diabetes medication", "Insulin treatment"], label_visibility="collapsed")
        if distribution == "Risk group":
            series = historical["risk_category"].replace({"Higher":"Higher risk","Moderate":"Moderate risk","Lower":"Lower risk"})
            order, colors = ["Higher risk","Moderate risk","Lower risk"], ["#d7675c","#e9b84d","#38a678"]
        elif distribution == "Diabetes medication":
            series = historical.get("diabetesmed", pd.Series("Unknown", index=historical.index)).replace({"Yes":"On diabetes medication","No":"No diabetes medication"})
            order, colors = ["On diabetes medication","No diabetes medication","Unknown"], ["#08764e","#7dc9a3","#bdc9c1"]
        else:
            series = historical.get("insulin", pd.Series("Unknown", index=historical.index)).replace({"No":"No insulin","Steady":"Steady","Up":"Dose increased","Down":"Dose decreased"})
            order, colors = ["No insulin","Steady","Dose increased","Dose decreased","Unknown"], ["#08764e","#3caa7b","#83cba6","#b0dec5","#bdc9c1"]
        counts = series.value_counts().reindex(order).dropna().astype(int)
        chart_data = pd.DataFrame({"segment":counts.index,"count":counts.values})
        pie = go.Figure(go.Pie(labels=chart_data.segment, values=chart_data["count"], hole=.58, sort=False,
                                marker={"colors":colors[:len(chart_data)],"line":{"color":"white","width":3}},
                                textinfo="label+percent", hovertemplate="%{label}<br>%{value:,} encounters (%{percent})<extra></extra>"))
        pie.update_layout(template="plotly_white", height=330, margin={"l":5,"r":5,"t":5,"b":5}, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", font={"family":"Segoe UI","color":"#18372c","size":12})
        pie_event = st.plotly_chart(pie, key="distribution_pie", on_select="rerun", selection_mode="points", width="stretch", config={"displayModeBar":False})
    with detail_col:
        points = getattr(getattr(pie_event,"selection",None),"points",[])
        segment = points[0].get("label") if points else None
        if segment not in set(chart_data.segment):
            point_index = points[0].get("point_index",points[0].get("point_number",0)) if points else 0
            segment = chart_data.iloc[point_index].segment if len(chart_data) else None
        if segment is not None:
            selected_series = historical[series == segment]
            stats = [float(selected_series["actual_readmitted_within_30_days"].mean()), float(selected_series.risk_probability.mean()), float(selected_series.risk_category.eq("Higher").mean())]
            st.markdown(f"#### {segment}")
            st.caption(f"{len(selected_series):,} encounters · select another pie sector to update")
            detail = go.Figure(go.Bar(x=["Actual readmission","Average model risk","Higher-risk share"], y=stats,
                                      marker_color=["#08764e","#3b9c72","#e0aa3e"], text=[f"{v:.1%}" for v in stats], textposition="outside", cliponaxis=False))
            detail.update_layout(template="plotly_white", height=300, margin={"l":8,"r":8,"t":24,"b":30},
                                 yaxis={"range":[0,1],"tickformat":".0%","tickfont":{"color":"#18372c"},"gridcolor":"#e2ece5"},
                                 xaxis={"tickfont":{"color":"#18372c","size":10}}, paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="rgba(0,0,0,0)", font={"family":"Segoe UI","color":"#18372c"})
            st.plotly_chart(detail, width="stretch", config={"displayModeBar":False})

elif page == "Historical records":
    st.subheader("Historical encounters")
    st.caption("Choose any held-out record to see why it received its score and the holdout fairness context for its groups. Records are ordered by estimated risk.")
    records = historical.sort_values("risk_probability", ascending=False).reset_index(drop=True)
    table = records.copy()
    table.insert(0,"Record",np.arange(1,len(table)+1))
    table["actual_readmitted_within_30_days"] = table["actual_readmitted_within_30_days"].map({1:"Yes",0:"No"})
    table["risk_probability"] = table.risk_probability.map(lambda value:f"{value:.1%}")
    table = table.rename(columns={"age":"Age band","gender":"Gender","admission_type_id":"Admission type","actual_readmitted_within_30_days":"Actual outcome","risk_probability":"Estimated risk"})
    columns = [name for name in ["Record","risk_category","Age band","Gender","Admission type","Actual outcome","Estimated risk"] if name in table]
    table = table.rename(columns={"risk_category":"Risk level"})
    columns = ["Risk level" if name == "risk_category" else name for name in columns]
    row_event = st.dataframe(table[columns], width="stretch", hide_index=True, height=380, on_select="rerun", selection_mode="single-row", key="historical_records_compact_v1")
    chosen = getattr(getattr(row_event,"selection",None),"rows",[])
    if chosen:
        record = records.iloc[chosen[0]]
        st.markdown(f"### Record {chosen[0]+1} · {record.age} · {record.gender} · {record.race}")
        explain_estimate(record.to_dict(),"historical")
        render_fairness(record.to_dict(),"historical")

else:
    st.subheader("New encounter")
    st.caption("Enter the encounter details below. Fields are compact; the trained pipeline handles the remaining inputs.")
    with st.container(border=True):
        with st.form("new_encounter"):
            c1,c2,c3,c4 = st.columns(4)
            age = c1.selectbox("Age band",["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)","[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"])
            gender = c2.selectbox("Gender",["Female","Male","Unknown"])
            race = c3.selectbox("Race",["Caucasian","AfricanAmerican","Hispanic","Asian","Other","Unknown"])
            admission_type = c4.selectbox("Admission type",["Emergency","Urgent","Elective","Newborn","Trauma Center","Unknown"])
            c1,c2,c3,c4 = st.columns(4)
            admission_source = c1.selectbox("Admission source",["Emergency Room","Physician Referral","Transfer","Clinic Referral","Unknown"])
            duration = c2.number_input("Hospital days",min_value=1,max_value=100,value=4,step=1)
            inpatient = c3.number_input("Prior inpatient visits",min_value=0,max_value=100,value=0,step=1)
            emergency = c4.number_input("Prior emergency visits",min_value=0,max_value=100,value=0,step=1)
            c1,c2,c3,c4 = st.columns(4)
            outpatient = c1.number_input("Prior outpatient visits",min_value=0,max_value=100,value=0,step=1)
            diagnoses = c2.number_input("Number of diagnoses",min_value=1,max_value=20,value=5,step=1)
            medications = c3.number_input("Number of medications",min_value=0,max_value=100,value=10,step=1)
            a1c = c4.selectbox("A1C result",["Unknown","None","Norm",">7",">8"])
            insulin = st.selectbox("Insulin",["No","Steady","Up","Down"])
            submitted = st.form_submit_button("Estimate readmission risk",type="primary")
    if submitted:
        patient = {column:np.nan for column in feature_columns}
        patient.update({"age":age,"gender":gender,"race":race,"admission_type_id":admission_type,"admission_source_id":admission_source,
                        "time_in_hospital":duration,"number_inpatient":inpatient,"number_emergency":emergency,"number_outpatient":outpatient,
                        "number_diagnoses":diagnoses,"num_medications":medications,"a1cresult":a1c,"insulin":insulin})
        risk, features = explain_estimate(patient,"new")
        render_fairness({"age":age,"gender":gender,"race":race},"new")
