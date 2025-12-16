# Spotify Playlist Rater

An interactive data science application that analyzes Spotify playlists and assigns a model-driven “Hit Index” score based on audio features and historical performance data.

## Overview
The Spotify Playlist Rater combines Spotify audio features with external Billboard chart and metadata sources to evaluate how closely a playlist resembles historically successful tracks. The project applies machine learning, statistical analysis, and data visualization to translate complex model outputs into clear, user-facing insights.

## Key Features
- Playlist-level scoring using supervised classification models trained on historical performance data
- Integration of Spotify audio features and external performance data
- Automated analysis pipeline built in Python
- Interactive Streamlit app for real-time playlist evaluation
- Clear visual storytelling and explainable model outputs

## How It Works
1. User submits a Spotify playlist URL  
2. Audio features and metadata are retrieved via the Spotify API  
3. Tracks are processed through a trained classification model  
4. Track-level predictions are aggregated into a playlist-level score  
5. Results are presented through an interactive Streamlit dashboard

## Tech Stack
- Python
- Pandas, NumPy
- scikit-learn
- Streamlit
- Git & version control

## Example Use Case
Users paste a Spotify playlist URL and receive a quantitative score along with interpretive metrics that explain how the playlist compares to historical benchmarks.

## Screenshots
Example views from the Streamlit application showing playlist scoring output and interpretive metrics.

<img width="1920" height="1027" alt="image" src="https://github.com/user-attachments/assets/bc11c076-6d39-4d69-bbb4-9e6a5ccc0ebb" />

<img width="1919" height="988" alt="image" src="https://github.com/user-attachments/assets/4a61340f-2a87-4bc3-9383-fa5f5ab4ca40" />

## Limitations & Future Work
Current models are trained on a fixed historical window and may be sensitive to genre and era effects. Future work includes expanded time-series modeling, improved handling of newer releases, and enhanced model explainability.
