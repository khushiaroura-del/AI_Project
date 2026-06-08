import streamlit as st

st.title("Video Test")

video = st.file_uploader("Upload Video", type=["mp4"])

if video:
    st.video(video)
