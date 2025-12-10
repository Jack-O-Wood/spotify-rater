import streamlit as st
import pandas as pd
import html
import time

from playlist_backend import (
    sp,                    # authenticated Spotify client
    best_xgb_full,         # trained model
    best_threshold_full,   # F1-optimal threshold
    rate_playlist          # the function
)

st.set_page_config(page_title="Playlist Rater", page_icon="🎧", layout="wide")

# --- Custom CSS for Spotify-style dark theme ---
st.markdown("""
<style>

/* ---------- GLOBAL / THEME ---------- */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #e5e7eb;
}

/* Page background */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #1f2937 0%, #020617 45%, #020617 100%);
}

/* Main content width */
.main .block-container {
    max-width: 900px;
    margin: 0 auto;
    padding-top: 2rem;
}

/* Kill the rounded "bubbles" between sections */
[data-testid="stVerticalBlock"] > div {
    background: transparent !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}

/* Remove default header background */
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* Default typography */
h1, h2, h3 {
    color: #f9fafb !important;
}

p, label, span {
    color: #e5e7eb !important;
}

/* ---------- HERO TITLE + BLURB (TOP) ---------- */

.title-container {
    text-align: center;
    margin-top: 2.5rem;
    margin-bottom: 1.8rem;
}

.title-text {
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    margin-bottom: 0.9rem;
    background: linear-gradient(90deg, #1db954, #22c55e, #a5b4fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.blurb-text {
    font-size: 1.08rem;
    max-width: 720px;
    margin: 0 auto;
    line-height: 1.6;
    color: #e5e7eb;
    opacity: 0.95;
    font-weight: 300;
}

/* ---------- SECTION TITLES & RATING TEXT ---------- */

/* e.g. "Playlist Rating" */
.section-title {
    font-size: 2.5rem;
    font-weight: 700;
    text-align: center;
    margin-top: 3rem;
    margin-bottom: 0.6rem;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #1db954, #22c55e, #a5b4fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Big % number */
.rating-number,
.big-rating {  /* keep old class name too, just in case */
    font-size: 4rem;
    font-weight: 800;
    text-align: center;
    margin-top: 0.2rem;
    color: #facc15;  /* gold */
}

/* Label line under % (e.g. "Uniquely Niche — deep cuts only") */
.rating-tagline,
.rating-label {
    text-align: center;
    font-size: 1.2rem;
    color: #f9fafb;
    margin-top: 0.3rem;
}

/* Subtext under label (e.g. "Based on 43 tracks") */
.rating-subtext {
    font-size: 0.9rem;
    text-align: center;
    color: #9ca3af;
    margin-top: 0.2rem;
}

/* e.g. "Top 3 Tracks (Cover Preview)" */
.subsection-title {
    font-size: 1.5rem;
    font-weight: 600;
    text-align: center;
    margin-top: 2.5rem;
    margin-bottom: 1rem;
    color: #e5e7eb;
}

/* ---------- RATING CARD / PANELS ---------- */

.card {
    background: #020617;
    border-radius: 18px;
    padding: 1.8rem 2rem;
    margin: 1.2rem 0;
    box-shadow: 0 12px 30px rgba(0,0,0,0.45);
    border: 1px solid rgba(148,163,184,0.35);
}

/* ---------- BUTTON ---------- */

.stButton > button {
    background: #020617 !important;
    color: #f9fafb !important;
    border-radius: 999px;
    border: 1px solid #64748b;
    padding: 0.45rem 1.4rem;
    font-weight: 600;
    font-size: 0.95rem;
}

.stButton > button:hover {
    background: #1f2937 !important;
    border-color: #1db954 !important;
}

/* ---------- CUSTOM SONG TABLES ---------- */

.cool-table {
    width: 100%;
    border-collapse: collapse;
    background: #020617;
    color: #f1f5f9;
    border-radius: 12px;
    overflow: hidden;
    font-size: 0.95rem;
}

.cool-table thead {
    background: #020617;
}

.cool-table th, .cool-table td {
    padding: 10px 12px;
    text-align: left;
}

.cool-table th {
    font-weight: 600;
    border-bottom: 1px solid #1f2937;
}

.cool-table tr:nth-child(even) td {
    background: #020617;
}

.cool-table tr:nth-child(odd) td {
    background: #020818;
}

.cool-table tr:hover td {
    background: #1f2937;
}

/* Hide empty spacer block right under the button */
[data-testid="stVerticalBlock"] > div:empty {
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}
            
/* Force-remove the empty container that appears after the button */
div[data-testid="stVerticalBlock"] div:has(div[data-testid="stVerticalBlock"] > div:empty),
div[data-testid="stVerticalBlock"] > div:empty {
    display: none !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Simplest removal — works on older browsers */
div[data-testid="stVerticalBlock"] > div:empty {
    display: none !important;
}


</style>
""", unsafe_allow_html=True)


# --- Custom Styled Header ---
st.markdown(
    """
    <style>
    /* Spotify-style header font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    .title-container {
        text-align: center;
        margin-top: 2.5rem;
        margin-bottom: 1.8rem;
        font-family: 'Inter', sans-serif;
    }

    .title-text {
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: -0.04em;
        margin-bottom: 0.9rem;
        background: linear-gradient(90deg, #1db954, #22c55e, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .blurb-text {
        font-size: 1.08rem;
        max-width: 720px;
        margin: 0 auto;
        line-height: 1.6;
        color: #e5e7eb;
        opacity: 0.95;
        font-weight: 300;
    }
    </style>

    <div class="title-container">
        <div class="title-text">Playlist Popularity Analyzer</div>
        <div class="blurb-text">
            This tool uses a machine learning model to score how <strong>“hit-like”</strong>
            the songs in your playlist are. A gradient-boosted model analyzes each track’s
            audio features and compares them to songs that performed well in our training data.
            The higher you score, the more mainstream your playlist. Lower, and look how cool
            and indie you are.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


st.caption("Paste a public Spotify playlist URL below to see how your taste stacks up.")

# --- Input ---

default_url = "https://open.spotify.com/playlist/0vurNqxrcDS4TYOpQvNxNA?si=pdUuGqKhRwiBfCrbv0unLg"

playlist_url = st.text_input(
    "Spotify playlist URL:",
    value=default_url,
    help="Make sure the playlist is public."
)

rate_button = st.button("Rate this playlist 🚀")

# --- When user clicks ---

if rate_button:
    if not playlist_url.strip():
        st.error("Please paste a valid Spotify playlist URL.")
    else:
        with st.spinner("Scoring your playlist..."):
            try:
                # 1) Model feature names
                model_features = list(best_xgb_full.get_booster().feature_names)

                # 2) Call core function
                summary, top5, bottom5, df_scored = rate_playlist(
                    playlist_url=playlist_url,
                    sp=sp,
                    model=best_xgb_full,
                    model_features=model_features,
                    threshold=best_threshold_full,
                    cache_buster=time.time(),
                )

                # --- Big final rating section ---
                final_pct = summary.get("final_score_pct", 0.0)
                label = summary.get("label", "")

                # Card container
                # st.markdown("<div class='card'>", unsafe_allow_html=True)

                # Spotify-styled section header + rating
                st.markdown(
                    "<div class='section-title'>Playlist Rating</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"<div class='rating-number'>{final_pct:.1f}%</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"<div class='rating-tagline'>{label}</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"<div class='rating-subtext'>Based on {len(df_scored)} tracks</div>",
                    unsafe_allow_html=True,
                )

                # st.markdown("</div>", unsafe_allow_html=True)


                # --- Top 3 covers strip (from top5) ---
                top3 = top5.head(3)

                # Only show if actually have image URLs
                if "album_image_url" in top3.columns:
                    st.markdown(
                        "<h3 style='text-align:center; color:#f9fafb; margin-top:1.5rem;'>"
                        "🔥 Top 3 Tracks (Cover Preview)"
                        "</h3>",
                        unsafe_allow_html=True,
                    )

                    cols = st.columns(3)

                    for i, (_, row) in enumerate(top3.iterrows()):
                        with cols[i]:
                            img_url = row["album_image_url"]
                            if img_url:
                                st.image(img_url, use_container_width=True)
                            st.markdown(
                                f"""
                                <div style='text-align:center; color:#f1f5f9; font-size:0.9rem; margin-top:0.5rem;'>
                                    <strong>{html.escape(str(row['track_name']))}</strong><br>
                                    <span style='color:#cbd5e1;'>{html.escape(str(row['artist_name']))}</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )



                # --- Top 5 section ---
                def render_song_table(df, title_emoji, title_text):
                    rows = []
                    for _, row in df.iterrows():
                        track = html.escape(str(row["track_name"]))
                        artist = html.escape(str(row["artist_name"]))
                        year = html.escape(str(row["year"]))
                        score = f"{row['hit_score']:.3f}"

                        # no leading spaces, no Markdown code block
                        rows.append(
                            f"<tr>"
                            f"<td>{track}</td>"
                            f"<td>{artist}</td>"
                            f"<td>{year}</td>"
                            f"<td>{score}</td>"
                            f"</tr>"
                        )

                    table_html = (
                        "<div class='card'>"
                        f"<h3>{title_emoji} {title_text}</h3>"
                        "<table class='cool-table'>"
                        "<thead>"
                        "<tr>"
                        "<th>Track</th>"
                        "<th>Artist</th>"
                        "<th>Year</th>"
                        "<th>Hit score</th>"
                        "</tr>"
                        "</thead>"
                        "<tbody>"
                        + "".join(rows) +
                        "</tbody>"
                        "</table>"
                        "</div>"
                    )

                    st.markdown(table_html, unsafe_allow_html=True)



                # --- Top 5 section ---
                render_song_table(top5, "🔥", "Top 5 most 'hit-like' tracks")

                # --- Bottom 5 section ---
                render_song_table(bottom5, "🧊", "Bottom 5 least 'hit-like' tracks")


                # Optional: expandable full table
                with st.expander("See full scored playlist"):
                    st.dataframe(df_scored)

            except Exception as e:
                st.error(f"Something went wrong: {e}")
