"""Results display — aggregated counts + per-image detail + CSV download."""

from __future__ import annotations

import io
import csv
import streamlit as st
import pandas as pd
from typing import List, Dict, Any
from collections import defaultdict


def render_results(results: List[Dict[str, Any]]) -> None:
    """
    Display analysis results:
    1. Total aggregated counts per feature (summed across all images)
    2. Detail table per image
    3. CSV download button

    Each result dict has: {filename, status, counts, [error]}
    """
    if not results:
        st.warning("No results to display.")
        return

    st.markdown("---")
    st.markdown("### 📊 Analysis Results")

    # ── Separate successful vs failed ────────────────────────
    successful = [r for r in results if r.get("status") == 200]
    failed_500 = [r for r in results if r.get("status") == 500]
    failed_404 = [r for r in results if r.get("status") == 404]
    if failed_404:
        r = failed_404[0]
        err = r.get("reason", f"Status {r.get('reason', 'unknown')}")
        st.warning(f"**{r['filename']}**: {err} , status code: {r['status']}")
        return
    if failed_500:
        r = failed_500[0]
        err = r.get("error", f"Status {r.get('error', 'unknown')}")
        st.warning(f"**{r['filename']}**: {err}, status code: {r['status']}")
        
    # ── Collect all unique features ──────────────────────────
    all_features = set()
    for r in successful:
        all_features.update(r.get("counts", {}).keys())
    all_features = sorted(all_features)

    # ── Aggregate totals ─────────────────────────────────────
    totals: Dict[str, int] = defaultdict(int)
    for r in successful:
        counts = r.get("counts", {})
        for feat in all_features:
            val = counts.get(feat, 0)
            if val > 0:  # Only count positive (present)
                totals[feat] += val

    # Display total counts
    st.markdown("#### 🔢 Total Feature Counts (across all images)")

    total_cols = st.columns(min(len(all_features), 4))
    for idx, feat in enumerate(all_features):
        with total_cols[idx % len(total_cols)]:
            st.metric(
                label=feat.replace("_", " ").title(),
                value=totals.get(feat, 0),
            )
    

    # ── Detail table ─────────────────────────────────────────
    st.markdown("#### 📋 Detail per Image")

    rows = []
    for r in successful:
        row = {"filename": r["filename"]}
        counts = r.get("counts", {})
        for feat in all_features:
            row[f"count_{feat}"] = counts.get(feat, 0)
        rows.append(row)

    # Add failed images with zeros
 
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)



    # ── CSV download ─────────────────────────────────────────
    st.markdown("#### 💾 Download CSV")

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)

    # Header: filename, count_feature1, count_feature2, ...
    header = ["filename"] + [f"count_{feat}" for feat in all_features]
    writer.writerow(header)

    for row in rows:
        csv_row = [row["filename"]] + [row.get(f"count_{feat}", 0) for feat in all_features]
        writer.writerow(csv_row)

    csv_data = csv_buffer.getvalue()

    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_data,
        file_name="feature_analysis_results.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )

    
