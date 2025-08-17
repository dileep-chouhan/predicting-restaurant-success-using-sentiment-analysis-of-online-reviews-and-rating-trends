import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Download VADER lexicon if not already present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_restaurants = 50
num_reviews = 200
# Generate synthetic data
data = {
    'RestaurantID': np.random.randint(1, num_restaurants + 1, size=num_reviews),
    'Date': pd.to_datetime(np.random.choice(pd.date_range(start='2022-01-01', periods=365), size=num_reviews)),
    'Rating': np.random.randint(1, 6, size=num_reviews), # Ratings from 1 to 5
    'ReviewText': [' '.join(np.random.choice(['good', 'bad', 'excellent', 'terrible', 'average', 'delicious', 'awful', 'amazing'], size=np.random.randint(3,10))) for i in range(num_reviews)]
}
df = pd.DataFrame(data)
# --- 2. Sentiment Analysis ---
analyzer = SentimentIntensityAnalyzer()
df['Sentiment'] = df['ReviewText'].apply(lambda review: analyzer.polarity_scores(review)['compound'])
# --- 3. Data Aggregation and Analysis ---
# Group by RestaurantID and calculate average rating and sentiment
restaurant_stats = df.groupby('RestaurantID').agg({'Rating': 'mean', 'Sentiment': 'mean'})
# --- 4. Visualization ---
# Scatter plot of average rating vs. average sentiment
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Rating', y='Sentiment', data=restaurant_stats)
plt.title('Average Rating vs. Average Sentiment')
plt.xlabel('Average Rating')
plt.ylabel('Average Sentiment')
plt.grid(True)
plt.tight_layout()
output_filename = 'rating_sentiment_scatter.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
# Histogram of average ratings
plt.figure(figsize=(10, 6))
sns.histplot(restaurant_stats['Rating'], kde=True)
plt.title('Distribution of Average Restaurant Ratings')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
output_filename = 'rating_histogram.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
#Further analysis could involve time series analysis of ratings and sentiment over time for individual restaurants or building a predictive model (e.g., regression) to forecast restaurant success based on these features.  This example focuses on the initial data manipulation and visualization steps.