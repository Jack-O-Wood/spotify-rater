import pandas as pd
import numpy as np
import re
from joblib import load
import streamlit as st


# ----------------------------------------------------------
# 1) SPOTIFY CLIENT
# ---------------------------------------------------------
#
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.exceptions import SpotifyException

SPOTIPY_CLIENT_ID = st.secrets["SPOTIPY_CLIENT_ID"]
SPOTIPY_CLIENT_SECRET = st.secrets["SPOTIPY_CLIENT_SECRET"]
SPOTIPY_REDIRECT_URI = st.secrets["SPOTIPY_REDIRECT_URI"]


SCOPE = "playlist-read-private playlist-read-collaborative"

sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope=SCOPE,
        show_dialog=True,
        open_browser=True,
    )
)

# ----------------------------------------------------------
# 2) TRAINED MODEL
# ----------------------------------------------------------
best_xgb_full = load("best_xgb_full.joblib")

# ----------------------------------------------------------
# 3) DECISION THRESHOLD
# ----------------------------------------------------------
best_threshold_full = 0.871

# ----------------------------------------------------------
# 4) HELPER FUNCTIONS 
# ----------------------------------------------------------
# Required functions:
#extract_playlist_id()
#load_playlist_tracks()
#fetch_audio_features()
#fetch_artist_info()
#follower_bucket()
#genres_to_flags()
#tempo_bucket_code_func()
#enrich_playlist_for_model()
#extract_year()

#label_from_score()
#summarize_playlist()
#rate_playlist()
#
# full function definitions below:

# --- load_playlist_tracks() ---

def extract_playlist_id(playlist_ref: str) -> str:
    """
    Accepts:
      - plain ID: "37i9dQZF1DXcBWIGoYBM5M"
      - full URL: "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M?..."
    Returns the bare playlist ID.
    """
    if "open.spotify.com/playlist" in playlist_ref:
        m = re.search(r"playlist/([a-zA-Z0-9]+)", playlist_ref)
        if m:
            return m.group(1)
    return playlist_ref


def load_playlist_tracks(playlist_ref: str, sp_client, cache_buster=None) -> pd.DataFrame:
    """
    Pull all tracks from a playlist and basic track/artist metadata.
    Returns df with columns: track_id, track_name, artist_name, artist_id, album_release_date, album_image_url.
    """
    playlist_id = extract_playlist_id(playlist_ref)

    items = []
    limit = 100
    offset = 0

    while True:
        results = sp_client.playlist_items(
            playlist_id=playlist_id,
            additional_types=("track",),
            limit=limit,
            offset=offset
        )
        batch = results.get("items", [])
        if not batch:
            break

        for item in batch:
            track = item.get("track")
            if track is None:
                continue
            if track.get("type") != "track":
                continue  # skip podcasts, etc.

            tid = track.get("id")
            tname = track.get("name")
            artists = track.get("artists", [])
            if not artists:
                continue
            main_artist = artists[0]
            aid = main_artist.get("id")
            aname = main_artist.get("name")

            album = track.get("album", {})
            release_date = album.get("release_date")

            # album cover URL (take first image if present)
            images = album.get("images", [])
            album_image_url = images[0]["url"] if images else None

            items.append(
                {
                    "track_id": tid,
                    "track_name": tname,
                    "artist_id": aid,
                    "artist_name": aname,
                    "album_release_date": release_date,
                    "album_image_url": album_image_url, 
                }
            )

        if results.get("next") is None:
            break

        offset += limit

    df = pd.DataFrame(items)
    print(f"Loaded {len(df)} playlist tracks (with ids).")
    return df

# --- artist info

def fetch_audio_features(track_ids, sp_client) -> pd.DataFrame:
    """
    Batch-fetch audio features for a list of track IDs.

    NOTE (2025): Spotify closed its API sadly: Spotify's /v1/audio-features endpoint now returns 403
    for many apps (deprecated / restricted). If that happens, we just
    return an empty DataFrame and continue without audio features.
    """
    track_ids = [tid for tid in track_ids if tid is not None]

    audio_rows = []
    try:
        for i in range(0, len(track_ids), 100):
            batch = track_ids[i:i+100]
            feats = sp_client.audio_features(batch)
            for af in feats:
                if af is None:
                    continue
                audio_rows.append(af)

        df_audio = pd.DataFrame(audio_rows)
        return df_audio

    except SpotifyException as e:
        print("⚠️ Could not fetch audio features (likely 403/deprecated endpoint). "
              "Continuing without audio features.")
        print(e)
        # Return empty DF so downstream code can handle missing columns
        return pd.DataFrame()

def fetch_artist_info(artist_ids, sp_client) -> pd.DataFrame:
    """
    Batch-fetch artist popularity, followers, and genres.
    """
    artist_rows = []
    artist_ids = list({aid for aid in artist_ids if aid is not None})

    for i in range(0, len(artist_ids), 50):
        batch = artist_ids[i:i+50]
        arts = sp_client.artists(batch)["artists"]
        for art in arts:
            if art is None:
                continue
            artist_rows.append(
                {
                    "artist_id": art["id"],
                    "artist_popularity_raw": art.get("popularity", 0),
                    "artist_followers_raw": art.get("followers", {}).get("total", 0),
                    "artist_genres_raw": art.get("genres", []),
                }
            )

    df_art = pd.DataFrame(artist_rows)
    return df_art

#---- 

def follower_bucket(n_followers: float) -> str:
    """
    Same logic you used when building followers_* features.
    Adjust thresholds if needed to match original code.
    """
    if n_followers >= 5_000_000:
        return "star"
    elif n_followers >= 1_000_000:
        return "big"
    elif n_followers >= 200_000:
        return "medium"
    elif n_followers >= 20_000:
        return "small"
    else:
        return "tiny"


def genres_to_flags(genre_list):
    """
    Map a list of Spotify genres to genre_* flags.
    We treat genres case-insensitively and search substrings.
    """
    if not isinstance(genre_list, list):
        genre_list = []

    g = " ".join(genre_list).lower()

    return {
        "genre_pop":       int("pop" in g),
        "genre_rock":      int("rock" in g),
        "genre_hip_hop":   int("hip hop" in g or "hip-hop" in g or "rap" in g),
        "genre_rap":       int("rap" in g),
        "genre_r&b":       int("r&b" in g or "rnb" in g),
        "genre_soul":      int("soul" in g),
        "genre_electronic":int("electronic" in g or "electro" in g),
        "genre_edm":       int("edm" in g),
        "genre_dance":     int("dance" in g),
        "genre_latin":     int("latin" in g),
        "genre_country":   int("country" in g),
        "genre_jazz":      int("jazz" in g),
        "genre_blues":     int("blues" in g),
        "genre_folk":      int("folk" in g),
        "genre_metal":     int("metal" in g),
        "num_genres":      len(genre_list),
    }


def tempo_bucket_code_func(t):
    if pd.isna(t):
        return 1  # treat missing as 'mid'
    t = float(t)
    if t < 80:   return 0  # slow
    if t < 110:  return 1  # mid
    if t < 140:  return 2  # upbeat
    return 3              # fast


def enrich_playlist_for_model(df_playlist_meta, sp_client) -> pd.DataFrame:
    # 1) audio features (may fail / be empty)
    df_audio = fetch_audio_features(df_playlist_meta["track_id"].tolist(), sp_client)

    # 2) artist info
    df_art = fetch_artist_info(df_playlist_meta["artist_id"].tolist(), sp_client)

    # 3) start from playlist meta
    df = df_playlist_meta.copy()

    # Merge audio features if available
    if not df_audio.empty:
        df = df.merge(df_audio, left_on="track_id", right_on="id", how="left")

    # Merge artist info
    df = df.merge(df_art, on="artist_id", how="left")

    # ensure raw audio feature columns exist even if audio-features failed ---
    audio_cols = [
        "energy",
        "valence",
        "danceability",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "tempo",
        "duration_ms",
    ]
    for col in audio_cols:
        if col not in df.columns:
            df[col] = np.nan

    # 4) basic fields: year / decade
    def extract_year(date_str):
        try:
            return int(str(date_str)[:4])
        except:
            return np.nan

    df["year"] = df["album_release_date"].apply(extract_year)
    df["decade"] = (df["year"] // 10) * 10

    # 5) fame features
    df["artist_popularity"] = df["artist_popularity_raw"].fillna(0).astype(float)
    df["artist_followers"] = df["artist_followers_raw"].fillna(0).astype(float)
    df["artist_followers_log"] = np.log1p(df["artist_followers"].clip(lower=0))

    df["followers_bucket"] = df["artist_followers"].apply(follower_bucket)
    for bucket in ["tiny", "small", "medium", "big", "star"]:
        df[f"followers_{bucket}"] = (df["followers_bucket"] == bucket).astype(int)

    # 6) genres flags
    genre_flags = df["artist_genres_raw"].apply(genres_to_flags).apply(pd.Series)
    df = pd.concat([df, genre_flags], axis=1)

    # 7) simple "is_cover" placeholder: assume 0 (original)
    df["is_cover"] = 0

    # 8) engineered features
    # These will just be NaN if the base audio features are NaN
    df["energy_valence"] = df["energy"] * df["valence"]
    df["dance_energy"] = df["danceability"] * df["energy"]
    df["loudness_energy"] = df["loudness"] * df["energy"]
    df["speech_energy"] = df["speechiness"] * df["energy"]

    df["energy_minus_valence"] = df["energy"] - df["valence"]
    df["dance_minus_acoustic"] = df["danceability"] - df["acousticness"]
    df["instrumental_minus_speech"] = df["instrumentalness"] - df["speechiness"]

    df["log_tempo"] = np.log1p(df["tempo"].clip(lower=0))
    df["log_duration"] = np.log1p(df["duration_ms"].clip(lower=0))

    df["tempo_bucket_code"] = df["tempo"].apply(tempo_bucket_code_func)

    return df


# ---  ---

def label_from_score(score_pct: float) -> str:
    """
    Map 0–100 playlist score to a fun label.
    You can tweak boundaries + text however you want.
    """
    if score_pct < 20:
        return "⚗️ Uniquely Niche — deep cuts only"
    elif score_pct < 50:
        return "🌱 Pretty unique — not bad"
    elif score_pct < 70:
        return "😎 Solid mix — balanced taste"
    elif score_pct < 85:
        return "🔥 Very mainstream — a bit basic"
    else:
        return "🚨 Algorithm’s Favorite Child — playlist built by Spotify itself 🚨"

# summarize_playlist()
def summarize_playlist(df_playlist_enriched, k=20, soft_threshold=0.70):

    MU_BG = 0.29   
    SIGMA_BG = 0.1

    scores = df_playlist_enriched["hit_score"]

    mean_score = scores.mean()

    # Make sure k isn't bigger than playlist length
    k_eff = min(k, len(scores))
    top_k_mean = scores.nlargest(k_eff).mean()

    playlist_index = 0.2 * mean_score + 0.8 * top_k_mean

    df_playlist_enriched["predicted_hit_soft"] = (
        df_playlist_enriched["hit_score"] >= soft_threshold
    ).astype(int)
    hit_rate_soft = df_playlist_enriched["predicted_hit_soft"].mean()

    # z score 
    z = (playlist_index - MU_BG) / SIGMA_BG

    rating = 40 + 20 * z
    rating = float(np.clip(rating, 0, 100).round(1))
    final_score_pct = round(playlist_index * 100, 1)
    label = label_from_score(rating)

    summary = {
        "mean_score": mean_score,
        "top_k_mean": top_k_mean,
        "playlist_index": playlist_index,
        "final_score_pct": rating, # either rating (z score), or final_score_pct (regular index)
        "label": label,
        "soft_threshold": soft_threshold,
        "soft_hit_rate": hit_rate_soft,
    }
    return summary

# rate_playlist()

def rate_playlist(
    playlist_url: str,
    sp,
    model,
    model_features,
    threshold: float,
    soft_threshold: float = 0.70,
    top_k: int = 5,
    cache_buster=None,
):
    """
    Given a Spotify playlist URL, return:
      - summary dict
      - top_k most 'hit-like' tracks (DataFrame)
      - bottom_k least 'hit-like' tracks (DataFrame)
      - full scored playlist DataFrame
    """
    # 1) Load + enrich playlist
    df_playlist_meta = load_playlist_tracks(playlist_url, sp)
    df_playlist_enriched = enrich_playlist_for_model(df_playlist_meta, sp)

    # 2) Ensure every model feature exists
    for col in model_features:
        if col not in df_playlist_enriched.columns:
            df_playlist_enriched[col] = 0

    # 3) Types → numeric
    df_playlist_enriched[model_features] = (
        df_playlist_enriched[model_features]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    # 4) Build X and predict
    X_pl = df_playlist_enriched[model_features]
    y_pl_probs = model.predict_proba(X_pl)[:, 1]

    df_playlist_enriched["hit_score"] = y_pl_probs
    df_playlist_enriched["predicted_hit"] = (
        df_playlist_enriched["hit_score"] >= threshold
    ).astype(int)

    # 5) Summary using existing logic
    summary = summarize_playlist(
        df_playlist_enriched,
        k=20,
        soft_threshold=soft_threshold,
    )

    # 6) Top / bottom tables for display
    display_cols = ["track_name", "artist_name", "year", "hit_score", "album_image_url"]

    # Remove duplicates
    deduped = df_playlist_enriched.drop_duplicates(
        subset=["track_name", "artist_name"]
    )

    # Top K
    top = (
        deduped
        .sort_values("hit_score", ascending=False)
        [display_cols]
        .head(top_k)
        .reset_index(drop=True)
    )

    # Bottom K
    bottom = (
        deduped
        .sort_values("hit_score", ascending=True)
        [display_cols]
        .head(top_k)
        .reset_index(drop=True)
    )


    return summary, top, bottom, df_playlist_enriched

