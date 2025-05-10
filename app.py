import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

from transformers import pipeline
from speech_to_text import convert_voice_to_text
import preprocessor, helper

sentiment_model = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# Sidebar
st.sidebar.title("📊 WhatsApp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a WhatsApp chat text file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)
    st.info("Performing sentiment analysis... This may take time.")
    df['Sentiment'] = df['message'].apply(
    lambda x: sentiment_model(x)[0]['label'] if x.strip() != '' else "Neutral"
)
    # User filter
    user_list = df['user'].unique().tolist()
    user_list = [user for user in user_list if user != 'group_notification']
    user_list.sort()
    user_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):

        # --- STATS ---
        st.header("📌 Top Statistics")
        num_messages, words, num_media, num_links = helper.fetch_stats(selected_user, df)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Messages", num_messages)
        col2.metric("Total Words", words)
        col3.metric("Media Shared", num_media)
        col4.metric("Links Shared", num_links)

        # --- SENTIMENT PIE ---
        st.header("🧠 Sentiment Distribution")
        if selected_user != "Overall":
            temp_df = df[df['user'] == selected_user]
        else:
            temp_df = df
        sentiment_counts = temp_df['Sentiment'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)


        # --- Timeline ---
        st.header("📅 Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig3, ax3 = plt.subplots()
        ax3.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig3)
        # --- Daily Timeline ---
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)
         # --- Activity Map ---
        st.title("Activity Map")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Most Busy Day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.subheader("Most Busy Month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # --- Activity Heatmap ---
        st.header("🗓️ Weekly Activity Heatmap")
        heatmap = helper.activity_heatmap(selected_user, df)
        fig4, ax4 = plt.subplots()
        ax4 = sns.heatmap(heatmap)
        st.pyplot(fig4)
         # --- Most Busy Users ---
        if selected_user == 'Overall':
            st.title("Most Busy Users")
            x, new_df = helper.most_busy_users(df)
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)
        # --- WordCloud ---
        st.header("☁️ Word Cloud")
        wc = helper.create_wordcloud(selected_user, df)
        fig5, ax5 = plt.subplots()
        ax5.imshow(wc)
        st.pyplot(fig5)

        # --- Emoji Usage ---
        st.header("😂 Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df)
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig6, ax6 = plt.subplots()
            ax6.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
            st.pyplot(fig6)


# 🎙️ Voice to Text
st.sidebar.header("🎤 Voice Transcription")
voice_file = st.sidebar.file_uploader("Upload WhatsApp voice message (.opus)", type=["opus"])
if voice_file is not None:
    with open("uploaded.opus", "wb") as f:
        f.write(voice_file.getbuffer())
    st.subheader("🗣️ Transcription Output")
    transcription = convert_voice_to_text("uploaded.opus")
    st.success(transcription)
