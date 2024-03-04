import pandas as pd
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


class SurveyData:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def clean_data(self):
        """
        Clean and pre-process the survey data.

        - Remove special characters from text fields.
        - Fill missing values in text fields with a specific value (e.g., "missing").
        - Convert text data to lowercase.
        """

        text_fields = ["question_1_response", "question_2_response", "additional_information"]
        for field in text_fields:
            self.data[field] = self.data[field].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", str(x)))
            self.data[field] = self.data[field].fillna("missing")
            self.data[field] = self.data[field].str.lower()

    def convert_response_delay(self):
        """
        Convert response delay from seconds to hours.
        """

        self.data["response_delay_hours"] = self.data["response_delay"] / 3600

    def categorize_sentiment(self):
        """
        Categorize responses into positive, neutral, and negative using a sentiment analyzer.

        - Add new columns: is_positive, is_neutral, is_negative.
        - Assign 1 to the relevant sentiment category based on the score.
        """

        analyzer = SentimentIntensityAnalyzer()

        def categorize_row(row):
            score = analyzer.polarity_scores(row["question_1_response"])["compound"]
            if score > 0.05:
                return 1, 0, 0
            elif score < -0.05:
                return 0, 0, 1
            else:
                return 0, 1, 0

        sentiment_data = self.data.apply(categorize_row, axis=1, result_type="expand")
        self.data["is_positive"], self.data["is_neutral"], self.data["is_negative"] = sentiment_data

    def extract_aspects(self):
        """
        Extract aspects (nouns and noun phrases) from question_2_response and store them as a comma-separated list in a new column named "tags".

        - Remove stopwords.
        - Extract nouns and noun phrases using part-of-speech tagging.
        """

        stop_words = set(stopwords.words("english"))

        def extract_nouns(text):
            tokens = word_tokenize(text)
            filtered_tokens = [word for word in tokens if word not in stop_words]
            tagged_tokens = pos_tag(filtered_tokens)
            nouns = [word for word, tag in tagged_tokens if tag.startswith("NN")]
            return ", ".join(nouns)

        self.data["tags"] = self.data["question_2_response"].apply(extract_nouns)

    def get_number_of_responses_over_time(self, interval="hour"):
        """
        Return a DataFrame showing the number of responses over time, grouped by the specified interval.
    
        Args:
            interval (str, optional): The desired time interval for grouping.
                Supported values: "hour", "day", "week", "month". Defaults to "hour".
    
        Returns:
            pd.DataFrame: A DataFrame with the response counts for each time interval.
        """
    
        if interval == "hour":
            return self.data.groupby(pd.Grouper(level="response_delay_hours", freq="H"))["question_1_response"].count()
        elif interval == "day":
            return self.data.groupby(pd.Grouper(key="response_delay_hours", freq="D"))["question_1_response"].count()
        elif interval == "week":
            return self.data.groupby(pd.Grouper(key="response_delay_hours", freq="W"))["question_1_response"].count()
        elif interval == "month":
            return self.data.groupby(pd.Grouper(key="response_delay_hours", freq="M"))["question_1_response"].count()
        else:
            raise ValueError(f"Invalid interval: {interval}. Supported values: 'hour', 'day', 'week', 'month'.")

    def get_tag_counts_by_sentiment(self):
        """
        Return a dictionary containing three sub-dictionaries for positive, neutral, and negative sentiment,
        each containing tag counts sorted by frequency (highest first).
        """
    
        tag_counts = {"positive": {}, "neutral": {}, "negative": {}}
        for index, row in self.data.iterrows():
            sentiment = "positive" if row["is_positive"] == 1 else (
                "neutral" if row["is_neutral"] == 1 else "negative"
            )
    
            if sentiment:  # Check if any sentiment category is True
                tags = row["tags"].split(", ")
                for tag in tags:
                    tag_counts[sentiment][tag] = tag_counts[sentiment].get(tag, 0) + 1
    
        # Sort tag counts by frequency (descending order) for each sentiment
        for sentiment, counts in tag_counts.items():
            tag_counts[sentiment] = dict(
                sorted(counts.items(), key=lambda item: item[1], reverse=True)
            )
    
        return tag_counts

    def get_highlight_reviews(self):
            """
            Return a dictionary containing the most prominent review (question_2_response) for each sentiment category (positive, neutral, negative).
    
            - Use the tag counts from get_tag_counts_by_sentiment to identify the most popular tag for each sentiment.
            - Filter responses based on sentiment and presence of the most popular tag.
            - Select the review with the shortest response delay (most recent) as the highlight review.
            - Handle cases where no reviews meet the criteria.
            """
    
            tag_counts = self.get_tag_counts_by_sentiment()
            highlight_reviews = {"positive": None, "neutral": None, "negative": None}
    
            for sentiment, counts in tag_counts.items():
                if counts:
                    most_popular_tag = next(iter(counts))
                    filtered_df = self.data[
                        (self.data["is_" + sentiment] == 1)
                        & (self.data["tags"].str.contains(most_popular_tag))
                    ]
                    if not filtered_df.empty:
                        # Select review with shortest response delay (ascending order)
                        highlight_reviews[sentiment] = (
                            filtered_df.sort_values(by="response_delay", ascending=True)
                            .iloc[0]["question_2_response"]
                        )
            print(highlight_reviews)
            return highlight_reviews


if __name__ == "__main__":
    survey_data = SurveyData("twilio.csv")

    # Clean and pre-process data
    survey_data.clean_data()
    survey_data.convert_response_delay()

    # Sentiment analysis and aspect extraction
    survey_data.categorize_sentiment()
    survey_data.extract_aspects()

    # Get number of responses over time (implement your desired logic)
    # number_of_responses_over_time = survey_data.get_number_of_responses_over_time()

    # Get tag counts by sentiment
    tag_counts_by_sentiment = survey_data.get_tag_counts_by_sentiment()
    print(f"\nTag counts by sentiment:\n{tag_counts_by_sentiment}")

    # Get highlight reviews
    highlight_reviews = survey_data.get_highlight_reviews()
    print(f"\nHighlight reviews:\n{highlight_reviews}")
