import streamlit as st
import pandas as pd
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px

# Load preprocessed data
with open("preprocessed_data.pkl", "rb") as f:
    data = pickle.load(f)

all_words = data["all_words"]
bigram_freq = data["bigram_freq"]
lda_topics = data["lda_topics"]
df = data["df"]

st.header("Research Trends in Hyperspectral Imaging")

st.write("This dashboard provides an overview of research trends in hyperspectral imaging based on a dataset of scientific papers from arXiv. The dataset contains information about the papers' titles, abstracts, authors, and publication dates. We will explore the most common words, bigrams, and topics in the dataset to gain insights into the research landscape.")

# 1. WordCloud Visualization

st.subheader("1. WordCloud of Most Frequent Words")
st.write("The WordCloud below shows the most frequent words in the dataset based on the abstracts of the papers. Of caurse, the most frequent words are the common stopwords, but after adding some custerised stopwords, some relevant terms can be analyzed.")
st.image("wordcloud-2.png", caption="WordCloud of Most Frequent Words", use_column_width=True)

# 2. Keyword Trends Over Time

df['year'] = pd.to_datetime(df['Published Date']).dt.year
keywords = ['deep learning', 'backpropagation', 'support vector', 
            'decision trees', 'random forest', 'knn', 
            'logistic regression', 'bayes']
for keyword in keywords:
    df[keyword + '_count'] = df['cleaned_text'].apply(lambda x: x.count(keyword))

yearly_keyword_counts = df.groupby('year')[[f"{kw}_count" for kw in keywords]].sum()

# Reshape the data into long format for Plotly
plot_data = yearly_keyword_counts.reset_index().melt(
    id_vars='year', 
    var_name='Keyword', 
    value_name='Count'
)

# Create an interactive line plot with Plotly
fig = px.line(
    plot_data, 
    x='year', 
    y='Count', 
    color='Keyword', 
    title='Applied Model Trends Over Time',
    labels={'year': 'Year', 'Count': 'Keyword Count'}
)

# Customize the plot layout
fig.update_layout(
    hovermode='x unified',
    legend_title_text='Methods',
    template='plotly_white',
    xaxis_title='Year',
    yaxis_title='Frequency'
)

# Adjust x-axis for better readability
fig.update_xaxes(
    dtick=1,
    tickmode='linear',
    tickangle=45  
)

# Render the Plotly figure in Streamlit
st.subheader("2. Applied Model Trends Over Time")
st.write("The line plot below shows the frequency of selected keywords related to machine learning models over time. You can hover over the lines to see the exact counts for each year. It is intersting to see that 'deep learning' has a booming trends over time, and 'Naive Bayes' is contiously applied thogh in a very low ratio.")
st.plotly_chart(fig)


# 3. LDA Topic Distribution Pie Chart

# topic weights and titles
lda_weights = [0.031264, 0.715539, 0.018380, 0.021485, 0.018234, 0.036456, 0.033198, 0.020555, 0.023916, 0.024475, 0.038826, 0.017672]


topic_titles = [
    "Quantum and Semiconductor Physics", "Machine Learning for Spatial Analysis", 
    "Computational Methods and Applications", "Tracking and Memory in Computational Systems", 
    "Specialized Spectroscopy and Analytical Techniques", "Infrared and Spectroscopy Technologies", 
    "Infrared in Biological and Medical Contexts", "Climate and Remote Sensing Applications", 
    "Spectral and Chemical Analysis", "Biomedical Imaging and Analysis", 
    "Spectral Unmixing and Data Variability", "Geoscience and High-Performance Computing"
]
    
# Prepare the data
data = pd.DataFrame({
    "Topic": topic_titles,
    "Weight": lda_weights
})

# Create an interactive pie chart with Plotly
fig = px.pie(
    data,
    values="Weight",
    names="Topic",
    title="Topic Distribution",
)

# Customize hover template
fig.update_traces(
    hovertemplate="<b>%{label}</b><br>Weight: %{value:.4f}"
)

# Display the chart in Streamlit
st.subheader("3. Topic Distribution")
st.write("The pie chart below shows the distribution of topics based on the Latent Dirichlet Allocation (LDA) model applied to the dataset. The weights represent the proportion of each topic in the dataset. You can hover over the chart to see the exact weights for each topic. Apart of the 'Machine Learning algorithm' topic, mostly reported in in the reaserch papers, the reports on 'Medical Imaging', 'Remote Sensing' and 'Chemical analysing' were being paid close attention.")
st.plotly_chart(fig, use_container_width=True)
