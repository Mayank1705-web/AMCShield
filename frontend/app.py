"""
frontend/app.py
Owner: Mayank Ingole [EN24CS3T10017]
Phase: 7 (rides alongside Phases 1-7, see phases.md Phase 7 table)

Two tabs: Demo (live inference) and Dashboard (read-only results viewer).

CONSTRAINT (see CONSTRAINTS.md): the Dashboard tab must ONLY read
results/metrics.json and results/plots/ -- it must NEVER re-run training
or attacks. Live computation belongs exclusively in the Demo tab.

CONSTRAINT: Dashboard tab defaults to STATIC charts, not live interactive
Plotly. Interactive is an opt-in Week 9 stretch only, never a blocking
dependency for any other phase. See DECISIONS.md "Frontend Defaults to
Static Charts" for why.

GOTCHA: the Demo tab uses the SAME config.yaml epsilon/step values as the
offline evaluation pipeline. If config.yaml changes after metrics.json was
generated, Demo tab (live) and Dashboard tab (precomputed) will disagree.
Regenerate metrics.json whenever attack parameters change.
"""

import streamlit as st
import json
import os

st.set_page_config(page_title="RF-AMC Robustness", layout="wide")

tab_demo, tab_dashboard = st.tabs(["Demo", "Dashboard"])

with tab_demo:
    st.header("Live Demo: Clean vs. Attacked Prediction")
    st.caption(
        "Week 3+: wire this to checkpoints/baseline_model.pt for clean "
        "predictions. Week 4+: add attack selection via src/attacks.py. "
        "Week 6+: add robust-model toggle."
    )
    # TODO (Person C, incrementally per phases.md Phase 7 table):
    # 1. Signal selector (pick a test sample or upload)
    # 2. Attack type + epsilon selector
    # 3. Model selector (baseline vs robust) -- added Week 6
    # 4. Run live inference + attack, display clean vs attacked prediction
    # 5. Plot the I/Q waveform (clean vs perturbed)
    st.info("Not yet implemented — see TODOs in this file.")

with tab_dashboard:
    st.header("Results Dashboard")
    st.caption("Reads results/metrics.json only. Never re-runs training or attacks.")

    metrics_path = "results/metrics.json"
    if not os.path.exists(metrics_path):
        st.warning(
            f"{metrics_path} not found yet. Run src/evaluate.py to generate it "
            "(Phase 5, Week 7)."
        )
    else:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        # TODO (Person C, Week 7):
        # 1. Render accuracy-vs-SNR curve (static image or matplotlib figure)
        # 2. Render confusion matrices
        # 3. Render the generalization-gap heatmap (attack type x SNR) --
        #    STATIC by default; interactive Plotly only as Week 9 stretch
        #    if Weeks 7-8 finished on schedule (see phases.md Phase 7).
        st.json(metrics)  # placeholder raw view until charts are implemented
