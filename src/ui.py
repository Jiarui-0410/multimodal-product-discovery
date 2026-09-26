"""Streamlit client for the Java product-discovery API."""

import os
from pathlib import Path

import requests
import streamlit as st


API_URL = os.getenv("JAVA_API_URL", "http://127.0.0.1:8080")


def api(method: str, path: str, **kwargs):
    response = requests.request(method, f"{API_URL}{path}", timeout=120, **kwargs)
    if not response.ok:
        try:
            message = response.json().get("message", response.text)
        except requests.JSONDecodeError:
            message = response.text
        raise RuntimeError(message or f"Request failed ({response.status_code})")
    return response.json()


def preference_chart(title: str, values: dict) -> None:
    st.markdown(f"**{title}**")
    if values:
        st.bar_chart(dict(list(values.items())[:8]))
    else:
        st.caption("No preference signal yet")


st.set_page_config(page_title="Personalized Product Discovery", page_icon="🧭", layout="wide")
st.title("Personalized Multimodal Product Discovery")
st.caption("CLIP + FAISS retrieval, reranked with long-term user preferences")

try:
    users = api("GET", "/api/users")
except Exception as exc:
    st.error(f"Java API is unavailable: {exc}")
    st.stop()

with st.sidebar:
    st.header("User")
    if users:
        selected = st.selectbox("Active user", users, format_func=lambda item: item["username"])
        user_id = selected["userId"]
    else:
        user_id = None
        st.info("Create the first user to begin.")
    with st.form("create-user", clear_on_submit=True):
        username = st.text_input("New username")
        if st.form_submit_button("Create user", use_container_width=True) and username.strip():
            api("POST", "/api/users", json={"username": username.strip()})
            st.rerun()

    st.header("Ranking")
    personalized = st.toggle("Use personalization", value=True)
    top_k = st.slider("Number of results", 5, 30, 10, 5)

if user_id is None:
    st.stop()

search_tab, profile_tab = st.tabs(["Search", "User profile"])

with search_tab:
    query_col, image_col = st.columns(2)
    with query_col:
        query = st.text_input("Text query", placeholder="black running shoes")
    with image_col:
        uploaded = st.file_uploader(
            "Image query",
            type=["png", "jpg", "jpeg", "webp"],
            max_upload_size=20,
            help="Maximum file size: 20 MB",
        )

    if st.button("Search", type="primary", use_container_width=True):
        try:
            with st.spinner("Retrieving and reranking candidates…"):
                if uploaded is not None:
                    result = api(
                        "POST",
                        "/api/search/image",
                        params={"userId": user_id, "topK": top_k, "personalized": personalized},
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    )
                elif query.strip():
                    result = api(
                        "POST",
                        "/api/search/text",
                        json={
                            "userId": user_id,
                            "query": query.strip(),
                            "topK": top_k,
                            "personalized": personalized,
                        },
                    )
                else:
                    raise RuntimeError("Enter a text query or upload an image")
            st.session_state["results"] = result
            st.session_state["result_user_id"] = user_id
        except Exception as exc:
            st.error(str(exc))

    result = (
        st.session_state.get("results")
        if st.session_state.get("result_user_id") == user_id
        else None
    )
    if result:
        st.subheader(f"Results · {result['mode'].replace('_', ' ').title()}")
        for row_start in range(0, len(result["results"]), 4):
            columns = st.columns(4)
            for column, item in zip(columns, result["results"][row_start : row_start + 4]):
                with column:
                    image_path = item.get("imagePath")
                    if image_path and Path(image_path).is_file():
                        st.image(str(Path(image_path)), use_container_width=True)
                    st.markdown(f"**{item['name']}**")
                    st.caption(" · ".join(filter(None, [item.get("category"), item.get("color"), item.get("usage")])))
                    st.metric("Final score", f"{item['finalScore']:.3f}")
                    st.caption(
                        f"Query {item['queryScore']:.3f} · User {item['userPreferenceScore']:.3f} "
                        f"· Metadata {item['metadataPreferenceScore']:.3f}"
                    )
                    action_cols = st.columns(3)
                    for action_col, action in zip(action_cols, ["CLICK", "LIKE", "SAVE"]):
                        if action_col.button(action.title(), key=f"{action}-{item['productId']}"):
                            api("POST", "/api/interactions", json={
                                "userId": user_id,
                                "productId": item["productId"],
                                "type": action,
                            })
                            st.toast(f"{action.title()} recorded")

with profile_tab:
    try:
        profile = api("GET", f"/api/users/{user_id}/profile")
        left, middle, right = st.columns(3)
        with left:
            preference_chart("Top categories", profile["topCategories"])
        with middle:
            preference_chart("Color preference", profile["colorPreferences"])
        with right:
            preference_chart("Usage preference", profile["usagePreferences"])
        st.subheader("Interaction statistics")
        stats = profile["interactionStatistics"]
        stat_columns = st.columns(max(1, len(stats)))
        for column, (name, count) in zip(stat_columns, stats.items()):
            column.metric(name.title(), count)
        st.subheader("Semantic profile")
        dimensions = len(profile["semanticEmbedding"])
        st.write(f"{dimensions}-dimensional normalized CLIP embedding" if dimensions else "Not available yet")
    except Exception as exc:
        st.error(str(exc))
