"""Streamlit dashboard for the News Topic Intelligence API."""

from __future__ import annotations

import os
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

DEFAULT_API_BASE = os.getenv("NEWS_API_BASE", "http://localhost:8000")
DEFAULT_API_KEY = os.getenv("NEWS_API_KEY")


@lru_cache(maxsize=1)
def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"accept": "application/json"})
    return session


def _make_headers(api_key: Optional[str], header_name: str) -> Dict[str, str]:
    if not api_key:
        return {}
    return {header_name: api_key}


def _build_url(api_base: str, path: str) -> str:
    base = api_base.rstrip("/") + "/"
    return urljoin(base, path.lstrip("/"))


def call_api(
    method: str,
    path: str,
    api_base: str,
    headers: Dict[str, str],
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    url = _build_url(api_base, path)
    try:
        response = _session().request(
            method,
            url,
            headers=headers,
            timeout=30,
            **kwargs,
        )
    except requests.RequestException as exc:
        st.error(f"Request to {url} failed: {exc}")
        return None

    if response.status_code >= 400:
        try:
            payload = response.json()
        except ValueError:
            payload = response.text
        hint = ""
        trailing_segment = api_base.rstrip("/").split("/")[-1]
        if response.status_code == 404 and "/" in trailing_segment:
            hint = " (check base URL/root path)"
        if response.status_code == 401:
            hint = " (provide a valid API key in the sidebar" " to authorize calls)"
        st.error(
            " ".join(
                [
                    method.upper(),
                    path,
                    "returned",
                    str(response.status_code) + ":",
                    str(payload),
                ]
            )
            + hint
        )
        return None

    if not response.content:
        return None

    try:
        return response.json()
    except ValueError:
        st.error("API response was not valid JSON")
        return None


def render_classification_panel(
    api_base: str,
    headers: Dict[str, str],
) -> None:
    st.subheader("Classify an article")
    with st.form("classify_form"):
        title = st.text_input("Title", "Breaking Markets Rally")
        text = st.text_area(
            "Article Text",
            (
                "Stocks surged today as markets reacted to strong earnings "
                "reports across the technology sector."
            ),
            height=180,
        )
        submitted = st.form_submit_button("Classify")

    if not submitted:
        return

    if len(text.strip()) < 10:
        st.warning("Please provide at least 10 characters of text.")
        return

    payload = {"title": title.strip() or None, "text": text.strip()}
    result = call_api(
        "post",
        "/classify_news",
        api_base,
        headers,
        json=payload,
    )
    if not result:
        return

    st.success(
        f"Top category: {result['top_category']} "
        f"(score {result['confidence_score']:.2f})"
    )

    categories = result.get("categories", [])
    if categories:
        chart_df = pd.DataFrame(categories)
        chart_df = chart_df.rename(columns={"name": "Category", "prob": "Probability"})
        chart_df["Probability"] = chart_df["Probability"].astype(float)
        fig = px.bar(
            chart_df,
            x="Category",
            y="Probability",
            text="Probability",
            title="Classification probabilities",
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(yaxis_range=[0, 1], margin=dict(t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

    meta_cols = st.columns(2)
    meta_cols[0].metric("Model Version", result.get("model_version", "?"))
    meta_cols[1].metric("Latency (ms)", f"{result.get('latency_ms', 0):.1f}")

    if result.get("suggestion"):
        st.info(result["suggestion"])


def render_summarization_panel(api_base: str, headers: Dict[str, str]) -> None:
    st.subheader("Summarize text")
    with st.form("summarize_form"):
        text = st.text_area(
            "Content",
            (
                "The central bank announced new measures to stabilize the "
                "currency following a period of volatility in international "
                "markets."
            ),
            height=180,
        )
        max_len = st.slider(
            "Max summary length",
            min_value=32,
            max_value=256,
            value=120,
            step=8,
        )
        min_len = st.slider(
            "Min summary length",
            min_value=8,
            max_value=max_len,
            value=25,
            step=1,
        )
        submitted = st.form_submit_button("Summarize")

    if not submitted:
        return

    if len(text.strip()) < 10:
        st.warning("Please provide at least 10 characters of text.")
        return

    payload = {
        "text": text.strip(),
        "max_len": max_len,
        "min_len": min_len,
    }
    result = call_api("post", "/summarize", api_base, headers, json=payload)
    if not result:
        return

    st.write("### Generated Summary")
    st.write(result["summary"])

    meta_cols = st.columns(3)
    meta_cols[0].metric("Model Version", result.get("model_version", "?"))
    meta_cols[1].metric("Latency (ms)", f"{result.get('latency_ms', 0):.1f}")
    meta_cols[2].metric("Cached", "Yes" if result.get("cached") else "No")


def render_trends_panel(api_base: str, headers: Dict[str, str]) -> None:
    st.subheader("Topic trends")
    window = st.slider(
        "Window (days)",
        min_value=1,
        max_value=90,
        value=14,
        step=1,
    )
    data = call_api(
        "get",
        "/trends",
        api_base,
        headers,
        params={"window": window},
    )
    if not data:
        return

    buckets = data.get("buckets", [])
    if buckets:
        df = pd.DataFrame(buckets)
        df["date"] = pd.to_datetime(df["date"])
        fig = px.area(
            df,
            x="date",
            y="count",
            color="label",
            title="Topic volume by day",
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Articles",
            legend_title="Topic",
            margin=dict(t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    totals = data.get("totals", [])
    if totals:
        totals_df = pd.DataFrame(totals)
        totals_df = totals_df.rename(columns={"label": "Topic", "count": "Total"})
        fig_totals = px.bar(
            totals_df,
            x="Topic",
            y="Total",
            text="Total",
            title="Total articles per topic",
        )
        fig_totals.update_traces(
            texttemplate="%{text}",
            textposition="outside",
        )
        fig_totals.update_layout(margin=dict(t=60, b=40))
        st.plotly_chart(fig_totals, use_container_width=True)

    st.caption(f"Generated at {data.get('generated_at')}")


def render_review_panel(api_base: str, headers: Dict[str, str]) -> None:
    st.subheader("Active learning review queue")

    label_catalog = call_api("get", "/labels", api_base, headers)
    if label_catalog and label_catalog.get("labels"):
        labels = label_catalog["labels"]
        label_count = label_catalog.get("count", len(labels))
        st.markdown(f"**Label catalog ({label_count})**")

        def _pretty_label(raw: str) -> str:
            return raw.replace("_", " ").title()

        grouped: Dict[str, list[str]] = defaultdict(list)
        for label in labels:
            head, *tail = label.split("_", 1)
            grouped[head].append(label)

        for family in sorted(grouped):
            members = sorted(grouped[family])
            family_title = _pretty_label(family)
            with st.expander(f"{family_title} ({len(members)})", expanded=False):
                columns = st.columns(3)
                for idx, raw_label in enumerate(members):
                    friendly = _pretty_label(raw_label)
                    columns[idx % len(columns)].markdown(f"- {friendly}")

    stats = call_api("get", "/review/active-learning", api_base, headers)
    if stats:
        cols = st.columns(4)
        cols[0].metric("Labeled review items", stats.get("review_labeled", 0))
        cols[1].metric("Labeled feedback", stats.get("feedback_labeled", 0))
        cols[2].metric("Total examples", stats.get("total_examples", 0))
        cols[3].metric(
            "Ready for training",
            "Yes" if stats.get("ready_for_training") else "No",
        )
        latest = stats.get("latest_label_at")
        if latest:
            st.caption(f"Most recent label: {latest}")
        labels = stats.get("distinct_labels") or []
        if labels:
            st.caption("Distinct labels: " + ", ".join(labels))

    st.write("### Pending queue")
    queue = call_api(
        "get",
        "/review/queue",
        api_base,
        headers,
        params={"limit": 20},
    )
    if not queue:
        st.info("No items currently waiting for review.")
        return

    for item in queue:
        header = (
            f"#{item['id']} • pred={item['predicted_label']} "
            f"• score={item['confidence_score']:.2f}"
        )
        with st.expander(header):
            st.write(item.get("text", ""))
            top_labels = item.get("top_labels") or []
            if top_labels:
                top_df = pd.DataFrame(top_labels)
                top_df = top_df.rename(columns={"name": "Label", "prob": "Probability"})
                top_df["Probability"] = top_df["Probability"].astype(float)
                top_view = (
                    top_df.sort_values("Probability", ascending=False)
                    .head(5)
                    .reset_index(drop=True)
                )
                st.write("Top probabilities")
                fig_top = px.bar(
                    top_view.sort_values("Probability"),
                    x="Probability",
                    y="Label",
                    orientation="h",
                    text="Probability",
                    range_x=[0, 1],
                    title="Top 5 labels",
                )
                fig_top.update_traces(
                    texttemplate="%{text:.2f}", textposition="outside"
                )
                fig_top.update_layout(margin=dict(t=50, b=10))
                st.plotly_chart(
                    fig_top,
                    use_container_width=True,
                    key=f"top-chart-{item['id']}",
                )
                st.dataframe(top_view, use_container_width=True)
            default_label = item.get("predicted_label", "")
            with st.form(f"label_form_{item['id']}"):
                true_label = st.text_input(
                    "True label",
                    value=default_label,
                    help="Provide the human-reviewed category",
                )
                submitted = st.form_submit_button("Submit label")
                if submitted:
                    if not true_label.strip():
                        st.warning("Enter a label before submitting.")
                    else:
                        payload = {
                            "item_id": item["id"],
                            "true_label": true_label.strip(),
                        }
                        resp = call_api(
                            "post",
                            "/review/label",
                            api_base,
                            headers,
                            json=payload,
                        )
                        if resp and resp.get("updated"):
                            st.success("Label saved.")
                            st.rerun()


def render_metrics_panel(api_base: str, headers: Dict[str, str]) -> None:
    st.subheader("Service metrics")
    data = call_api("get", "/metrics", api_base, headers)
    if not data:
        return

    summary_cols = st.columns(4)
    summary_cols[0].metric("Backend", data.get("backend", "?"))
    summary_cols[1].metric(
        "Classifier Version",
        data.get("model_version", "?"),
    )
    summary_cols[2].metric("Labels", data.get("label_count", 0))
    summary_cols[3].metric("Feedback Entries", data.get("feedback_total", 0))

    latency_payload = data.get("latency_ms") or {}
    if latency_payload:
        st.write("### Latency (ms)")
        latency_rows: list[dict[str, Any]] = []
        for route, stats in latency_payload.items():
            row = {
                "Route": route,
                "Count": int(stats.get("count", 0)),
                "Avg": float(stats.get("avg_ms", 0.0)),
                "p50": float(stats.get("p50_ms", 0.0)),
                "p95": float(stats.get("p95_ms", 0.0)),
            }
            recent = stats.get("recent") or {}
            if recent:
                row["Recent p95"] = float(recent.get("p95_ms", 0.0))
                row["Recent window"] = int(recent.get("window", 0))
            latency_rows.append(row)
        if latency_rows:
            latency_df = pd.DataFrame(latency_rows)
            latency_df = latency_df.sort_values("Avg", ascending=False)
            st.dataframe(latency_df, use_container_width=True)
            fig_latency = px.bar(
                latency_df,
                x="Route",
                y=["Avg", "p95"],
                title="Average vs p95 latency",
                barmode="group",
            )
            fig_latency.update_layout(margin=dict(t=60, b=40))
            st.plotly_chart(fig_latency, use_container_width=True)

    counters_payload = data.get("request_counters") or {}
    if counters_payload:
        st.write("### Request counters")
        counter_rows: list[dict[str, Any]] = []
        for key, value in counters_payload.items():
            route, _, status = key.partition(":")
            counter_rows.append(
                {
                    "Route": route,
                    "Status": status or "?",
                    "Count": int(value),
                }
            )
        counter_df = pd.DataFrame(counter_rows)
        counter_df = counter_df.sort_values("Count", ascending=False)
        st.dataframe(counter_df, use_container_width=True)
        fig_counters = px.bar(
            counter_df,
            x="Route",
            y="Count",
            color="Status",
            title="Request volume by route/status",
        )
        fig_counters.update_layout(margin=dict(t=60, b=40))
        st.plotly_chart(fig_counters, use_container_width=True)

    st.write("### Raw Metrics Payload")
    st.json(data)


def main() -> None:
    st.set_page_config(
        page_title="News Intelligence Dashboard",
        page_icon="🗞️",
        layout="wide",
    )
    st.title("News Intelligence Dashboard")
    st.caption(
        "Interact with the FastAPI service for classification, "
        "summarization, and trends."
    )

    with st.sidebar:
        st.header("Connection")
        api_base = st.text_input("API Base URL", value=DEFAULT_API_BASE)
        api_key = st.text_input(
            "API Key",
            value=DEFAULT_API_KEY or "",
            type="password",
        )
        header_name = st.text_input(
            "API Key Header",
            value=os.getenv("API_KEY_HEADER", "x-api-key"),
        )

    headers = _make_headers(api_key.strip() or None, header_name.strip())

    tabs = st.tabs(["Classify", "Summarize", "Trends", "Review", "Metrics"])

    with tabs[0]:
        render_classification_panel(api_base, headers)
    with tabs[1]:
        render_summarization_panel(api_base, headers)
    with tabs[2]:
        render_trends_panel(api_base, headers)
    with tabs[3]:
        render_review_panel(api_base, headers)
    with tabs[4]:
        render_metrics_panel(api_base, headers)


if __name__ == "__main__":
    main()
