import nltk
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
import re
import json
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag

class DataPreprocessor:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def clean_data(self):
        text_fields = ['question_1_response', 'question_2_response', 'additional_information']

        # Ensure text fields exist before processing
        missing_fields = set(text_fields) - set(self.dataframe.columns)
        if missing_fields:
            raise ValueError(f"DataFrame is missing expected text fields: {missing_fields}")

        # Remove special characters
        for field in text_fields:
            self.dataframe[field] = self.dataframe[field].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)))

        # Fill missing values
        self.dataframe[text_fields] = self.dataframe[text_fields].fillna('missing')

        # Convert to lowercase
        self.dataframe[text_fields] = self.dataframe[text_fields].apply(lambda x: x.str.lower())

        # print(f"Cleaned text fields: {text_fields}")

    def convert_response_delay(self):
        if 'response_delay' not in self.dataframe.columns:
            raise ValueError("DataFrame is missing the 'response_delay' column")

        self.dataframe['response_delay_hours'] = self.dataframe['response_delay'] / 3600
        # print(f"Created 'response_delay_hours' column")
        


class SentimentAnalyzer:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        nltk.download('vader_lexicon')
        self.analyzer = SentimentIntensityAnalyzer()
    
    def categorize_sentiment(self):
        # Initialize counters for each sentiment
        positive_count = 0
        neutral_count = 0
        negative_count = 0
    
        # Use NLTK's SentimentIntensityAnalyzer to categorize sentiment
        def categorize(row):
            nonlocal positive_count, neutral_count, negative_count
            score = self.analyzer.polarity_scores(row)['compound']
            if score > 0.05:
                positive_count += 1
                return 1, 0, 0  # Positive
            elif score < -0.05:
                negative_count += 1
                return 0, 0, 1  # Negative
            else:
                neutral_count += 1
                return 0, 1, 0  # Neutral
    
        self.dataframe[['is_positive', 'is_neutral', 'is_negative']] = self.dataframe.apply(lambda row: categorize(row['question_1_response']), axis=1, result_type='expand')
    
        # Print the counts
        # print(f"Sentiment counts - Positive: {positive_count}, Neutral: {neutral_count}, Negative: {negative_count}")

class AspectExtractor:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        # Load stopwords once and reuse
        self.stop_words = set(stopwords.words('english'))

    def extract_aspects(self):
        # Define a function to filter for nouns and noun phrases
        def extract_nouns(text):
            # Tokenize the sentence
            tokens = word_tokenize(text)
            # Remove stopwords
            tokens = [word for word in tokens if word not in self.stop_words]
            # Part-of-speech tagging
            tagged_tokens = pos_tag(tokens)
            # Extracting nouns
            nouns = [word for word, tag in tagged_tokens if tag.startswith("NN")]
            return ", ".join(nouns)
        
        # Apply the function to each row in question_2_response
        self.dataframe['tags'] = self.dataframe['question_2_response'].apply(extract_nouns)


class DataVisualizer:
    def __init__(self, dataframe):
        self.dataframe = dataframe
    
    def visualize_sentiments_over_delay(self):
        # Ensure response_delay_hours is numeric
        self.dataframe['response_delay_hours'] = pd.to_numeric(self.dataframe['response_delay_hours'], errors='coerce')
        
        # Create bins for every hour of response delay
        max_hours = self.dataframe['response_delay_hours'].max()
        bins = np.arange(0, max_hours + 1, 1)  # +1 to ensure the last hour is included
        labels = [f"{int(x)}-{int(x+1)}" for x in bins[:-1]]  # Label each bin by its range
        self.dataframe['hourly_bins'] = pd.cut(self.dataframe['response_delay_hours'], bins=bins, labels=labels, right=False)

        # Calculate sentiment distribution per hourly bin, explicitly setting observed parameter
        sentiment_distribution = self.dataframe.groupby('hourly_bins', observed=True).agg({
            'is_positive': 'sum',
            'is_neutral': 'sum',
            'is_negative': 'sum'
        }).fillna(0)
        
                # Calculate the total responses per bin for the running total
        self.dataframe['total_responses'] = self.dataframe['is_positive'] + self.dataframe['is_neutral'] + self.dataframe['is_negative']
        total_responses_per_bin = self.dataframe.groupby('hourly_bins', observed=True)['total_responses'].sum().cumsum()
        
        plot_data = {
            'sentiment_distribution': sentiment_distribution.reset_index().to_dict(orient='records'),
            'total_responses_per_bin': total_responses_per_bin.reset_index().to_dict(orient='records'),
        }
        # Plotting setup
        fig, ax1 = plt.subplots(figsize=(14, 8))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Define colors
        
        # Plot bar chart
        sentiment_distribution.plot(kind='bar', stacked=True, color=colors, ax=ax1)
        ax1.set_ylabel('Number of Responses')
        ax1.set_xlabel('Response Delay (Hours)')
        
        # Add line chart for the running total
        ax2 = ax1.twinx()
        ax2.plot(total_responses_per_bin.index, total_responses_per_bin, 'r-o', linewidth=2, markersize=5)
        ax2.set_ylabel('Running Total of Responses', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Final adjustments
        plt.xticks(rotation=45, ha="right")  # Ensure x-ticks are properly rotated
        plt.title('Sentiment Distribution and Running Total of Responses Over Response Delay')
        plt.tight_layout()  # Adjust layout
        plt.savefig('sentiment_distribution_over_response_delay.png')
        plt.show()
        plt.close()

        return plot_data
        
    def export_to_json(self, data, filename):
        with open(filename, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4, default=convert_numpy_int64)


    def generate_tag_counts(self):
        # Assuming 'tags' column contains comma-separated tags
        self.dataframe['tag_list'] = self.dataframe['tags'].str.split(', ')
        
        # Initialize dictionaries to count tags for each sentiment
        positive_tags = {}
        neutral_tags = {}
        negative_tags = {}
        
        # Iterate through DataFrame to populate dictionaries
        for index, row in self.dataframe.iterrows():
            for tag in row['tag_list']:
                if row['is_positive'] == 1:
                    positive_tags[tag] = positive_tags.get(tag, 0) + 1
                elif row['is_neutral'] == 1:
                    neutral_tags[tag] = neutral_tags.get(tag, 0) + 1
                elif row['is_negative'] == 1:
                    negative_tags[tag] = negative_tags.get(tag, 0) + 1
        
        # Sort dictionaries by count in descending order
        positive_tags = dict(sorted(positive_tags.items(), key=lambda item: item[1], reverse=True))
        neutral_tags = dict(sorted(neutral_tags.items(), key=lambda item: item[1], reverse=True))
        negative_tags = dict(sorted(negative_tags.items(), key=lambda item: item[1], reverse=True))
        
        # Print the top 5 tags for each sentiment
        # print("Top 5 Positive Tags:", list(positive_tags.items())[:5])
        # print("Top 5 Neutral Tags:", list(neutral_tags.items())[:5])
        # print("Top 5 Negative Tags:", list(negative_tags.items())[:5])
        
        return positive_tags, neutral_tags, negative_tags
        
    def restore_capitalization_and_periods(self, text):
        # Tokenize the text into sentences
        sentences = sent_tokenize(text)
        # Capitalize the first letter of each sentence and ensure it ends with a period
        restored_sentences = [sentence[0].upper() + sentence[1:] if len(sentence) > 0 else "" for sentence in sentences]
        restored_sentences = [sentence if sentence.endswith('.') else sentence + '.' for sentence in restored_sentences]
        # Join the sentences back to form the full text
        restored_text = ' '.join(restored_sentences)
        return restored_text
        
    def find_highlight_review(self, tags_dict, sentiment_filter):
        if tags_dict:
            most_popular_tag = next(iter(tags_dict))
            # Filter dataframe for the sentiment and check if the most popular tag is in the tags list
            filtered_df = self.dataframe[self.dataframe['tag_list'].apply(lambda x: most_popular_tag in x) & sentiment_filter]
            if not filtered_df.empty:
                # Sort by response_delay (ascending) to select the review with the shortest delay
                return filtered_df.sort_values(by='response_delay', ascending=True).iloc[0]['question_2_response']
        return None

    def identify_highlight_reviews(self):
        # Assume generate_tag_counts has been called and we have dictionaries for tag counts
        positive_tags, neutral_tags, negative_tags = self.generate_tag_counts()

        highlight_reviews = {
            'positive': self.restore_capitalization_and_periods(self.find_highlight_review(positive_tags, self.dataframe['is_positive'] == 1)),
            'neutral': self.restore_capitalization_and_periods(self.find_highlight_review(neutral_tags, self.dataframe['is_neutral'] == 1)),
            'negative': self.restore_capitalization_and_periods(self.find_highlight_review(negative_tags, self.dataframe['is_negative'] == 1))
        }

        return highlight_reviews

# Before calling json.dump, convert numpy.int64 to int
def convert_numpy_int64(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


def main():
    # Download necessary NLTK resources
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)  # Ensure the sentiment analysis lexicon is also downloaded

    try:
        dataframe = pd.read_csv('twilio.csv')
    except FileNotFoundError:
        print("Error: The dataset file 'twilio.csv' was not found.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading the dataset: {e}")
        return

    try:
        preprocessor = DataPreprocessor(dataframe)
        sentiment_analyzer = SentimentAnalyzer(dataframe)
        aspect_extractor = AspectExtractor(dataframe)
        visualizer = DataVisualizer(dataframe)

        output_data = {}

        preprocessor.clean_data()
        preprocessor.convert_response_delay()

        sentiment_analyzer.categorize_sentiment()
        output_data['sentiment_counts'] = {
            'positive': dataframe['is_positive'].sum(),
            'neutral': dataframe['is_neutral'].sum(),
            'negative': dataframe['is_negative'].sum(),
        }

        aspect_extractor.extract_aspects()

        # Capture plot data
        plot_data = visualizer.visualize_sentiments_over_delay()
        output_data['plot_data'] = plot_data

        # Generate tag counts and identify highlight reviews
        top_tags = visualizer.generate_tag_counts()
        output_data['top_5_tags'] = {
            'positive': list(top_tags[0].items())[:5],
            'neutral': list(top_tags[1].items())[:5],
            'negative': list(top_tags[2].items())[:5],
        }

        highlight_reviews = visualizer.identify_highlight_reviews()
        output_data['highlight_reviews'] = highlight_reviews

        visualizer.export_to_json(output_data, 'analysis_results.json')
        # print("Analysis results exported to 'analysis_results.json'.")
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()
