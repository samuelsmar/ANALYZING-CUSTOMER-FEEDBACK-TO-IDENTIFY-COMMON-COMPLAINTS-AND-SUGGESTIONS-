# Step 1: Import necessary libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from collections import Counter
import re

# Download NLTK stopwords (only once)
nltk.download('stopwords')

# Step 2: Load Data
df = pd.read_csv('Womens Clothing E-Commerce Reviews.csv')
df = df.dropna(subset=['Review Text'])
df[['Rating', 'Recommended IND', 'Review Text']].head()

from textblob import TextBlob

# Function to label sentiment
def get_sentiment(text):
    analysis = TextBlob(text)
    return 'Positive' if analysis.sentiment.polarity > 0 else 'Negative'

# Apply sentiment function to the reviews column
df['Sentiment Label'] = df['Review Text'].apply(get_sentiment)

# Check the updated dataframe
print(df[['Review Text', 'Sentiment Label']].head())

# Step 3: Get stopwords
stop_words = set(stopwords.words('english'))

# Step 4: Clean function to remove stopwords
def clean_words(text):
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())  # Only words with 3+ letters
    return [word for word in words if word not in stop_words]

# Step 5: Extract negative (complaints) and positive (suggestions) reviews
neg_reviews = df[df['Sentiment Label'] == 'Negative']['Review Text']
pos_reviews = df[df['Sentiment Label'] == 'Positive']['Review Text']

# Step 6: Extract and count words for complaints (Negative reviews)
neg_words = Counter()
for review in neg_reviews:
    neg_words.update(clean_words(review))

# Step 7: Extract and count words for suggestions (Positive reviews)
pos_words = Counter()
for review in pos_reviews:
    pos_words.update(clean_words(review))
# Apply sentiment function to the reviews column
df['Sentiment Label'] = df['Review Text'].apply(get_sentiment)

# Check the updated dataframe
print(df[['Review Text', 'Sentiment Label']].head())

# Step 3: Get stopwords
stop_words = set(stopwords.words('english'))

# Step 4: Clean function to remove stopwords
def clean_words(text):
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())  # Only words with 3+ letters
    return [word for word in words if word not in stop_words]

# Step 5: Extract negative (complaints) and positive (suggestions) reviews
neg_reviews = df[df['Sentiment Label'] == 'Negative']['Review Text']
pos_reviews = df[df['Sentiment Label'] == 'Positive']['Review Text']

# Step 6: Extract and count words for complaints (Negative reviews)
neg_words = Counter()
for review in neg_reviews:
    neg_words.update(clean_words(review))

# Step 7: Extract and count words for suggestions (Positive reviews)
pos_words = Counter()
for review in pos_reviews:
    pos_words.update(clean_words(review))

# Step 8: Output top words for complaints and suggestions
print('Top Complaint Words:', neg_words.most_common(10))
print('Top Suggestion Words:', pos_words.most_common(10))
print(df.columns)
# Step 6: Extract and count words for complaints (Negative reviews)
neg_words = Counter()
for review in neg_reviews:
    neg_words.update(clean_words(review))

# Step 7: Extract and count words for suggestions (Positive reviews)
pos_words = Counter()
for review in pos_reviews:
    pos_words.update(clean_words(review))

# Step 8: Output top words for complaints and suggestions
print('Top Complaint Words:', neg_words.most_common(10))
print('Top Suggestion Words:', pos_words.most_common(10))
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Generate word cloud for complaints
neg_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(neg_words)

# Plot the word cloud for complaints
plt.figure(figsize=(10, 5))
plt.imshow(neg_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for Complaints")
plt.show()

# Generate word cloud for suggestions
pos_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(pos_words)

# Plot the word cloud for suggestions
plt.figure(figsize=(10, 5))
plt.imshow(pos_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for Suggestions")
plt.show()
# Plotting top complaint words (negative reviews)
complaints = neg_words.most_common(10)
complaints_words, complaints_counts = zip(*complaints)

plt.figure(figsize=(10, 6))
plt.barh(complaints_words, complaints_counts, color='red')
plt.xlabel('Frequency')
plt.title('Top 10 Complaint Words')
plt.gca().invert_yaxis()  # To display the most frequent words at the top
plt.show()

# Plotting top suggestion words (positive reviews)
suggestions = pos_words.most_common(10)
suggestions_words, suggestions_counts = zip(*suggestions)

plt.figure(figsize=(10, 6))
plt.barh(suggestions_words, suggestions_counts, color='green')
plt.xlabel('Frequency')
plt.title('Top 10 Suggestion Words')
plt.gca().invert_yaxis()  # To display the most frequent words at the top
plt.show()
