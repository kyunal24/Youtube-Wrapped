import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import IsolationForest
from googleapiclient.discovery import build
import re
import isodate
from dotenv import load_dotenv
import os
from googleapiclient.discovery import build


def load_api_key():
    with open("key.txt", "r") as f:
        return f.read().strip()
api_key = load_api_key()


#styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@800&display=swap');
    .main {
        background: linear-gradient(45deg, #1DB954, #191414);
        color: white;
    }
    .big-header {
        font-family: 'Montserrat', sans-serif;
        font-size: 4em !important;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        margin-bottom: 0.5em;
    }
    .section-header {
        font-family: 'Montserrat', sans-serif;
        font-size: 2.5em !important;
        border-left: 5px solid #1DB954;
        padding-left: 1rem;
        margin: 2rem 0;
    }
    .stats-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def extract_video_id(url):
    if isinstance(url, str):
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
        if match:
            return match.group(1)
    return None

language_to_country = {
    'en': 'USA/UK',
    'en-US': 'USA',
    'en-GB': 'UK',
    'hi': 'India',
    'es': 'Spain/Mexico',
    'fr': 'France',
    'de': 'Germany',
    'ja': 'Japan',
    'ko': 'South Korea',
    'zh-Hans': 'China',
    'zh-Hant': 'Taiwan/Hong Kong',
    'pt': 'Portugal',
    'pt-BR': 'Brazil',
    'ru': 'Russia',
    'ar': 'Arab Countries',
    'ta': 'India (Tamil)',
    'te': 'India (Telugu)',
    'ml': 'India (Malayalam)',
    'mr': 'India (Marathi)',
    'bn': 'India/Bangladesh (Bengali)',
    'ur': 'Pakistan',
    'th': 'Thailand',
    'tr': 'Turkey',
    'vi': 'Vietnam',
    'id': 'Indonesia'
}

def fetch_video_metadata(video_ids):
    youtube = build("youtube", "v3", developerKey=api_key)
    results = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        req = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=",".join(batch)
        )
        res = req.execute()
        for item in res.get("items", []):
            stats = item.get("statistics", {})
            snippet = item.get("snippet", {})
            content = item.get("contentDetails", {})
            results.append({
                'videoId': item['id'],
                'title_from_api': snippet.get('title'),
                'channelTitle': snippet.get('channelTitle'),
                'channelId': snippet.get('channelId'),
                'categoryId': snippet.get('categoryId'),
                'country': language_to_country.get(snippet.get('defaultAudioLanguage', ''), 'Unknown'),
                'publishedAt': snippet.get('publishedAt'),
                'duration': content.get('duration'),
                'viewCount': int(stats.get('viewCount', 0)),
                'likeCount': int(stats.get('likeCount', 0))
            })
    return pd.DataFrame(results)

def parse_duration(duration):
    try:
        return isodate.parse_duration(duration).total_seconds()
    except:
        return None

def map_category_to_genre(cat_id):
    mapping = {
        '1': 'Film',
        '2': 'Vehicles',
        '10': 'Music',
        '15': 'Animals',
        '17': 'Sports',
        '18': 'Shorts',
        '20': 'Gaming',
        '22': 'Vlogs',
        '23': 'Comedy',
        '24': 'Entertainment',
        '25': 'News',
        '26': 'Lifestyle',
        '27': 'Education',
        '28': 'Tech',
        '29': 'Nonprofits'
    }
    return mapping.get(str(cat_id), 'Other')

def is_ad_row(row):
    details = row.get('details', None)
    if isinstance(details, list):
        return any(d.get('name') == 'From Google Ads' for d in details if isinstance(d, dict))
    return False

def main():
    st.markdown('<div class="big-header">YouTube Rewind</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your `watch-history.json`", type='json')
    year = st.number_input("Enter the Year for Rewind", min_value=2000, max_value=2100, value=2024)

    if uploaded_file is not None:
        data = json.load(uploaded_file)
        df = pd.DataFrame(data)

        if 'time' not in df.columns:
            st.error("No 'time' field found.")
            return

        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df[df['time'].dt.year == year].dropna(subset=['time'])

        if df.empty:
            st.warning(f"No videos found for {year}.")
            return

        df['videoId'] = df['titleUrl'].apply(extract_video_id)
        df = df.dropna(subset=['videoId'])
        video_ids = df['videoId'].unique().tolist()
        df['is_ad'] = df.apply(is_ad_row, axis=1)

        ads_df = df[df['is_ad'] == True]
        df = df[df['is_ad'] == False]

        meta_df = fetch_video_metadata(video_ids)
        meta_df['video_seconds'] = meta_df['duration'].apply(parse_duration)
        meta_df['watch_time'] = df.groupby('videoId')['time'].diff(-1).dt.total_seconds().abs()
        df = pd.merge(df, meta_df, on="videoId", how="left")

        df['genre'] = df['categoryId'].apply(map_category_to_genre)
        filtered_df = df[df['genre'] != "Other"]

        # Watch Completion
        df['watch_ratio'] = df['watch_time'] / df['video_seconds']
        percent_watched_full = df[df['watch_ratio'] >= 0.9].shape[0] / df.shape[0] * 100

        # Controversial video
        df['like_ratio'] = df['likeCount'] / (df['viewCount'] + 1)
        controversial = df.sort_values('like_ratio',ascending=False).iloc[0]

        # Regional affinity
        region_counts = df['country'].value_counts()
        top_region = region_counts.idxmax() if not region_counts.empty else "Unknown"

        # Channel analysis
        channels = [item['name'] for sublist in df['subtitles'].dropna() for item in sublist]
        channel_counts = Counter(channels)

        ads_count = len(ads_df)
        ads_total_time = ads_df['video_seconds'].sum() if 'video_seconds' in ads_df.columns else 0

        # Binge behavior
        df = df.sort_values('time')
        df['time_diff'] = df['time'].diff().dt.total_seconds() / 60
        df['session_id'] = (df['time_diff'] > 30).cumsum()
        sessions = df.groupby('session_id').agg(
            video_count=('title', 'count'),
            start=('time', 'min'),
            end=('time', 'max')
        ).reset_index()
        sessions['duration'] = (sessions['end'] - sessions['start']).dt.total_seconds() / 60
        model = IsolationForest(contamination=0.1, random_state=42)
        sessions['anomaly'] = model.fit_predict(sessions[['video_count', 'duration']])
        binges = sessions[sessions['anomaly'] == -1]


        col1, col2, col3 = st.columns(3)
        with col1:
            top_genre = filtered_df['genre'].mode().iloc[0]
            st.markdown(f'<div class="stats-card"><h3>🎮 Top Genre</h3><h1>{top_genre}</h1></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="stats-card"><h3>📺 Total Videos</h3><h1>{len(df):,}</h1></div>', unsafe_allow_html=True)
        with col3:
            top_channel = channel_counts.most_common(1)[0][0] if channel_counts else "N/A"
            st.markdown(f'<div class="stats-card"><h3>⭐ Top Channel</h3><h1>{top_channel}</h1></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">💥 Most Controversial Video</div>', unsafe_allow_html=True)
        st.write(f"**{controversial['title_from_api']}** by *{controversial['channelTitle']}* — Like Ratio: {controversial['like_ratio']:.2%}")

        st.markdown('<div class="section-header">🌍 Regional Viewing Style</div>', unsafe_allow_html=True)
        st.write(f"Your watching behavior is most similar to viewers from: **{top_region}**")

        st.markdown('<div class="section-header">⏱️ Watch Completion</div>', unsafe_allow_html=True)
        st.write(f"You watched **{percent_watched_full:.1f}%** of videos completely.")

        st.markdown('<div class="section-header">Content DNA</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots()
        filtered_df['genre'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        ax.set_ylabel('')
        st.pyplot(fig)

        st.markdown('<div class="section-header">Binge Behavior</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Longest Session", f"{binges['duration'].max():.1f} mins" if not binges.empty else "N/A")
        with col2:
            st.metric("Most Videos in a Session", f"{binges['video_count'].max()}" if not binges.empty else "N/A")

        st.markdown('<div class="section-header">Peak Watching Time</div>', unsafe_allow_html=True)
        peak_hour = df['time'].dt.hour.mode()[0]
        st.write(f"**{peak_hour}:00 - {peak_hour + 1}:00** was your peak watching hour.")

        st.markdown('<div class="section-header">🎯 Ad Consumption</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Ads Watched", f"{ads_count}")
        with col2:
            st.metric("Time Wasted on Ads", f"{ads_total_time / 60:.1f} mins")


if __name__ == "__main__":
    main()
