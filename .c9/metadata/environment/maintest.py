{"filter":false,"title":"maintest.py","tooltip":"/maintest.py","undoManager":{"mark":19,"position":19,"stack":[[{"start":{"row":0,"column":0},"end":{"row":97,"column":60},"action":"insert","lines":["import pandas as pd","import re","import nltk","from nltk.sentiment import SentimentIntensityAnalyzer","from nltk.corpus import stopwords","from nltk.tokenize import word_tokenize, sent_tokenize","from nltk.tag import pos_tag","","nltk.download('vader_lexicon', quiet=True)","nltk.download('punkt', quiet=True)","nltk.download('stopwords', quiet=True)","","","class SurveyData:","    def __init__(self, file_path):","        self.data = pd.read_csv(file_path)","","    def clean_data(self):","        \"\"\"","        Clean and pre-process the survey data.","","        - Remove special characters from text fields.","        - Fill missing values in text fields with a specific value (e.g., \"missing\").","        - Convert text data to lowercase.","        \"\"\"","","        text_fields = [\"question_1_response\", \"question_2_response\", \"additional_information\"]","        for field in text_fields:","            self.data[field] = self.data[field].apply(lambda x: re.sub(r\"[^a-zA-Z0-9\\s]\", \"\", str(x)))","            self.data[field].fillna(\"missing\", inplace=True)","            self.data[field] = self.data[field].str.lower()","","    def convert_response_delay(self):","        \"\"\"","        Convert response delay from seconds to hours.","        \"\"\"","","        self.data[\"response_delay_hours\"] = self.data[\"response_delay\"] / 3600","","    def categorize_sentiment(self):","        \"\"\"","        Categorize responses into positive, neutral, and negative using a sentiment analyzer.","","        - Add new columns: is_positive, is_neutral, is_negative.","        - Assign 1 to the relevant sentiment category based on the score.","        \"\"\"","","        analyzer = SentimentIntensityAnalyzer()","","        def categorize_row(row):","            score = analyzer.polarity_scores(row[\"question_1_response\"])[\"compound\"]","            if score > 0.05:","                return 1, 0, 0","            elif score < -0.05:","                return 0, 0, 1","            else:","                return 0, 1, 0","","        sentiment_data = self.data.apply(categorize_row, axis=1, result_type=\"expand\")","        self.data[\"is_positive\"], self.data[\"is_neutral\"], self.data[\"is_negative\"] = sentiment_data","","    def extract_aspects(self):","        \"\"\"","        Extract aspects (nouns and noun phrases) from question_2_response and store them as a comma-separated list in a new column named \"tags\".","","        - Remove stopwords.","        - Extract nouns and noun phrases using part-of-speech tagging.","        \"\"\"","","        stop_words = set(stopwords.words(\"english\"))","","        def extract_nouns(text):","            tokens = word_tokenize(text)","            filtered_tokens = [word for word in tokens if word not in stop_words]","            tagged_tokens = pos_tag(filtered_tokens)","            nouns = [word for word, tag in tagged_tokens if tag.startswith(\"NN\")]","            return \", \".join(nouns)","","        self.data[\"tags\"] = self.data[\"question_2_response\"].apply(extract_nouns)","","    def get_number_of_responses_over_time(self):","        \"\"\"","        Return a DataFrame showing the number of responses over time (can be hourly, daily, etc., based on your preference).","        \"\"\"","","        # Implement your desired logic to group and count responses based on time intervals","        # Example: Count responses by hour","        # hourly_counts = self.data.groupby(pd.Grouper(level=\"response_delay_hours\", freq=\"H\"))[\"question_1_response\"].count()","        # return hourly_counts","","    def get_tag_counts_by_sentiment(self):","        \"\"\"","        Return a dictionary containing three sub-dictionaries for positive, neutral, and negative sentiment, each containing tag counts sorted by frequency (highest first).","        \"\"\"","","        tag_counts = {\"positive\": {}, \"neutral\": {}, \"negative\": {}}","        for index, row in self.data.iterrows():","            sentiment = \"positive\" if row[\"is_positive\"] == "],"id":1}],[{"start":{"row":90,"column":0},"end":{"row":97,"column":60},"action":"remove","lines":["    def get_tag_counts_by_sentiment(self):","        \"\"\"","        Return a dictionary containing three sub-dictionaries for positive, neutral, and negative sentiment, each containing tag counts sorted by frequency (highest first).","        \"\"\"","","        tag_counts = {\"positive\": {}, \"neutral\": {}, \"negative\": {}}","        for index, row in self.data.iterrows():","            sentiment = \"positive\" if row[\"is_positive\"] == "],"id":2},{"start":{"row":90,"column":0},"end":{"row":99,"column":25},"action":"insert","lines":["    def get_tag_counts_by_sentiment(self):","        \"\"\"","        Return a dictionary containing three sub-dictionaries for positive, neutral, and negative sentiment,","        each containing tag counts sorted by frequency (highest first).","        \"\"\"","","        tag_counts = {\"positive\": {}, \"neutral\": {}, \"negative\": {}}","        for index, row in self.data.iterrows():","            sentiment = \"positive\" if row[\"is_positive\"] == 1 else (","                \"neutral\""]}],[{"start":{"row":99,"column":25},"end":{"row":100,"column":0},"action":"insert","lines":["",""],"id":3},{"start":{"row":100,"column":0},"end":{"row":100,"column":16},"action":"insert","lines":["                "]}],[{"start":{"row":100,"column":16},"end":{"row":164,"column":0},"action":"insert","lines":["            if sentiment:  # Add check to avoid KeyError if sentiment is None/empty","                tags = row[\"tags\"].split(\", \")","                for tag in tags:","                    tag_counts[sentiment][tag] = tag_counts[sentiment].get(tag, 0) + 1","","        # Sort tag counts by frequency (descending order) for each sentiment","        for sentiment, counts in tag_counts.items():","            tag_counts[sentiment] = dict(","                sorted(counts.items(), key=lambda item: item[1], reverse=True)","            )","","        return tag_counts","","    def get_highlight_reviews(self):","        \"\"\"","        Return a dictionary containing the most prominent review (question_2_response) for each sentiment category (positive, neutral, negative).","","        - Use the tag counts from get_tag_counts_by_sentiment to identify the most popular tag for each sentiment.","        - Filter responses based on sentiment and presence of the most popular tag.","        - Select the review with the shortest response delay (most recent) as the highlight review.","        - Handle cases where no reviews meet the criteria.","        \"\"\"","","        tag_counts = self.get_tag_counts_by_sentiment()","        highlight_reviews = {\"positive\": None, \"neutral\": None, \"negative\": None}","","        for sentiment, counts in tag_counts.items():","            if counts:","                most_popular_tag = next(iter(counts))","                filtered_df = self.data[","                    (self.data[\"is_\" + sentiment] == 1)","                    & (self.data[\"tags\"].str.contains(most_popular_tag))","                ]","                if not filtered_df.empty:","                    # Select review with shortest response delay (ascending order)","                    highlight_reviews[sentiment] = (","                        filtered_df.sort_values(by=\"response_delay\", ascending=True)","                        .iloc[0][\"question_2_response\"]","                    )","","        return highlight_reviews","","","if __name__ == \"__main__\":","    survey_data = SurveyData(\"twilio.csv\")","","    # Clean and pre-process data","    survey_data.clean_data()","    survey_data.convert_response_delay()","","    # Sentiment analysis and aspect extraction","    survey_data.categorize_sentiment()","    survey_data.extract_aspects()","","    # Get number of responses over time (implement your desired logic)","    # number_of_responses_over_time = survey_data.get_number_of_responses_over_time()","","    # Get tag counts by sentiment","    tag_counts_by_sentiment = survey_data.get_tag_counts_by_sentiment()","    print(f\"\\nTag counts by sentiment:\\n{tag_counts_by_sentiment}\")","","    # Get highlight reviews","    highlight_reviews = survey_data.get_highlight_reviews()","    print(f\"\\nHighlight reviews:\\n{highlight_reviews}\")",""],"id":4}],[{"start":{"row":90,"column":0},"end":{"row":111,"column":25},"action":"remove","lines":["    def get_tag_counts_by_sentiment(self):","        \"\"\"","        Return a dictionary containing three sub-dictionaries for positive, neutral, and negative sentiment,","        each containing tag counts sorted by frequency (highest first).","        \"\"\"","","        tag_counts = {\"positive\": {}, \"neutral\": {}, \"negative\": {}}","        for index, row in self.data.iterrows():","            sentiment = \"positive\" if row[\"is_positive\"] == 1 else (","                \"neutral\"","                            if sentiment:  # Add check to avoid KeyError if sentiment is None/empty","                tags = row[\"tags\"].split(\", \")","                for tag in tags:","                    tag_counts[sentiment][tag] = tag_counts[sentiment].get(tag, 0) + 1","","        # Sort tag counts by frequency (descending order) for each sentiment","        for sentiment, counts in tag_counts.items():","            tag_counts[sentiment] = dict(","                sorted(counts.items(), key=lambda item: item[1], reverse=True)","            )","","        return tag_counts"],"id":5},{"start":{"row":90,"column":0},"end":{"row":113,"column":21},"action":"insert","lines":["def get_tag_counts_by_sentiment(self):","    \"\"\"","    Return a dictionary containing three sub-dictionaries for positive, neutral, and negative sentiment,","    each containing tag counts sorted by frequency (highest first).","    \"\"\"","","    tag_counts = {\"positive\": {}, \"neutral\": {}, \"negative\": {}}","    for index, row in self.data.iterrows():","        sentiment = \"positive\" if row[\"is_positive\"] == 1 else (","            \"neutral\" if row[\"is_neutral\"] == 1 else \"negative\"","        )","","        if sentiment:  # Check if any sentiment category is True","            tags = row[\"tags\"].split(\", \")","            for tag in tags:","                tag_counts[sentiment][tag] = tag_counts[sentiment].get(tag, 0) + 1","","    # Sort tag counts by frequency (descending order) for each sentiment","    for sentiment, counts in tag_counts.items():","        tag_counts[sentiment] = dict(","            sorted(counts.items(), key=lambda item: item[1], reverse=True)","        )","","    return tag_counts"]}],[{"start":{"row":115,"column":0},"end":{"row":115,"column":4},"action":"remove","lines":["    "],"id":6}],[{"start":{"row":157,"column":5},"end":{"row":157,"column":6},"action":"remove","lines":[" "],"id":16},{"start":{"row":157,"column":4},"end":{"row":157,"column":5},"action":"remove","lines":["#"]},{"start":{"row":157,"column":0},"end":{"row":157,"column":4},"action":"remove","lines":["    "]}],[{"start":{"row":157,"column":0},"end":{"row":157,"column":4},"action":"insert","lines":["    "],"id":17}],[{"start":{"row":157,"column":4},"end":{"row":157,"column":6},"action":"insert","lines":["# "],"id":18}],[{"start":{"row":29,"column":12},"end":{"row":29,"column":60},"action":"remove","lines":["self.data[field].fillna(\"missing\", inplace=True)"],"id":19},{"start":{"row":29,"column":12},"end":{"row":29,"column":65},"action":"insert","lines":["self.data[field] = self.data[field].fillna(\"missing\")"]}],[{"start":{"row":90,"column":0},"end":{"row":90,"column":4},"action":"insert","lines":["    "],"id":20},{"start":{"row":91,"column":0},"end":{"row":91,"column":4},"action":"insert","lines":["    "]},{"start":{"row":92,"column":0},"end":{"row":92,"column":4},"action":"insert","lines":["    "]},{"start":{"row":93,"column":0},"end":{"row":93,"column":4},"action":"insert","lines":["    "]},{"start":{"row":94,"column":0},"end":{"row":94,"column":4},"action":"insert","lines":["    "]},{"start":{"row":95,"column":0},"end":{"row":95,"column":4},"action":"insert","lines":["    "]},{"start":{"row":96,"column":0},"end":{"row":96,"column":4},"action":"insert","lines":["    "]},{"start":{"row":97,"column":0},"end":{"row":97,"column":4},"action":"insert","lines":["    "]},{"start":{"row":98,"column":0},"end":{"row":98,"column":4},"action":"insert","lines":["    "]},{"start":{"row":99,"column":0},"end":{"row":99,"column":4},"action":"insert","lines":["    "]},{"start":{"row":100,"column":0},"end":{"row":100,"column":4},"action":"insert","lines":["    "]},{"start":{"row":101,"column":0},"end":{"row":101,"column":4},"action":"insert","lines":["    "]},{"start":{"row":102,"column":0},"end":{"row":102,"column":4},"action":"insert","lines":["    "]},{"start":{"row":103,"column":0},"end":{"row":103,"column":4},"action":"insert","lines":["    "]},{"start":{"row":104,"column":0},"end":{"row":104,"column":4},"action":"insert","lines":["    "]},{"start":{"row":105,"column":0},"end":{"row":105,"column":4},"action":"insert","lines":["    "]},{"start":{"row":106,"column":0},"end":{"row":106,"column":4},"action":"insert","lines":["    "]},{"start":{"row":107,"column":0},"end":{"row":107,"column":4},"action":"insert","lines":["    "]},{"start":{"row":108,"column":0},"end":{"row":108,"column":4},"action":"insert","lines":["    "]},{"start":{"row":109,"column":0},"end":{"row":109,"column":4},"action":"insert","lines":["    "]},{"start":{"row":110,"column":0},"end":{"row":110,"column":4},"action":"insert","lines":["    "]},{"start":{"row":111,"column":0},"end":{"row":111,"column":4},"action":"insert","lines":["    "]},{"start":{"row":112,"column":0},"end":{"row":112,"column":4},"action":"insert","lines":["    "]},{"start":{"row":113,"column":0},"end":{"row":113,"column":4},"action":"insert","lines":["    "]}],[{"start":{"row":115,"column":0},"end":{"row":115,"column":4},"action":"insert","lines":["    "],"id":21},{"start":{"row":116,"column":0},"end":{"row":116,"column":4},"action":"insert","lines":["    "]},{"start":{"row":117,"column":0},"end":{"row":117,"column":4},"action":"insert","lines":["    "]},{"start":{"row":118,"column":0},"end":{"row":118,"column":4},"action":"insert","lines":["    "]},{"start":{"row":119,"column":0},"end":{"row":119,"column":4},"action":"insert","lines":["    "]},{"start":{"row":120,"column":0},"end":{"row":120,"column":4},"action":"insert","lines":["    "]},{"start":{"row":121,"column":0},"end":{"row":121,"column":4},"action":"insert","lines":["    "]},{"start":{"row":122,"column":0},"end":{"row":122,"column":4},"action":"insert","lines":["    "]},{"start":{"row":123,"column":0},"end":{"row":123,"column":4},"action":"insert","lines":["    "]},{"start":{"row":124,"column":0},"end":{"row":124,"column":4},"action":"insert","lines":["    "]},{"start":{"row":125,"column":0},"end":{"row":125,"column":4},"action":"insert","lines":["    "]},{"start":{"row":126,"column":0},"end":{"row":126,"column":4},"action":"insert","lines":["    "]},{"start":{"row":127,"column":0},"end":{"row":127,"column":4},"action":"insert","lines":["    "]},{"start":{"row":128,"column":0},"end":{"row":128,"column":4},"action":"insert","lines":["    "]},{"start":{"row":129,"column":0},"end":{"row":129,"column":4},"action":"insert","lines":["    "]},{"start":{"row":130,"column":0},"end":{"row":130,"column":4},"action":"insert","lines":["    "]},{"start":{"row":131,"column":0},"end":{"row":131,"column":4},"action":"insert","lines":["    "]},{"start":{"row":132,"column":0},"end":{"row":132,"column":4},"action":"insert","lines":["    "]},{"start":{"row":133,"column":0},"end":{"row":133,"column":4},"action":"insert","lines":["    "]},{"start":{"row":134,"column":0},"end":{"row":134,"column":4},"action":"insert","lines":["    "]},{"start":{"row":135,"column":0},"end":{"row":135,"column":4},"action":"insert","lines":["    "]},{"start":{"row":136,"column":0},"end":{"row":136,"column":4},"action":"insert","lines":["    "]},{"start":{"row":137,"column":0},"end":{"row":137,"column":4},"action":"insert","lines":["    "]},{"start":{"row":138,"column":0},"end":{"row":138,"column":4},"action":"insert","lines":["    "]},{"start":{"row":139,"column":0},"end":{"row":139,"column":4},"action":"insert","lines":["    "]},{"start":{"row":140,"column":0},"end":{"row":140,"column":4},"action":"insert","lines":["    "]},{"start":{"row":141,"column":0},"end":{"row":141,"column":4},"action":"insert","lines":["    "]},{"start":{"row":142,"column":0},"end":{"row":142,"column":4},"action":"insert","lines":["    "]}],[{"start":{"row":80,"column":0},"end":{"row":88,"column":30},"action":"remove","lines":["    def get_number_of_responses_over_time(self):","        \"\"\"","        Return a DataFrame showing the number of responses over time (can be hourly, daily, etc., based on your preference).","        \"\"\"","","        # Implement your desired logic to group and count responses based on time intervals","        # Example: Count responses by hour","        # hourly_counts = self.data.groupby(pd.Grouper(level=\"response_delay_hours\", freq=\"H\"))[\"question_1_response\"].count()","        # return hourly_counts"],"id":22},{"start":{"row":80,"column":0},"end":{"row":101,"column":108},"action":"insert","lines":["def get_number_of_responses_over_time(self, interval=\"hour\"):","    \"\"\"","    Return a DataFrame showing the number of responses over time, grouped by the specified interval.","","    Args:","        interval (str, optional): The desired time interval for grouping.","            Supported values: \"hour\", \"day\", \"week\", \"month\". Defaults to \"hour\".","","    Returns:","        pd.DataFrame: A DataFrame with the response counts for each time interval.","    \"\"\"","","    if interval == \"hour\":","        return self.data.groupby(pd.Grouper(level=\"response_delay_hours\", freq=\"H\"))[\"question_1_response\"].count()","    elif interval == \"day\":","        return self.data.groupby(pd.Grouper(key=\"response_delay_hours\", freq=\"D\"))[\"question_1_response\"].count()","    elif interval == \"week\":","        return self.data.groupby(pd.Grouper(key=\"response_delay_hours\", freq=\"W\"))[\"question_1_response\"].count()","    elif interval == \"month\":","        return self.data.groupby(pd.Grouper(key=\"response_delay_hours\", freq=\"M\"))[\"question_1_response\"].count()","    else:","        raise ValueError(f\"Invalid interval: {interval}. Supported values: 'hour', 'day', 'week', 'month'.\")"]}],[{"start":{"row":80,"column":0},"end":{"row":80,"column":4},"action":"insert","lines":["    "],"id":23},{"start":{"row":81,"column":0},"end":{"row":81,"column":4},"action":"insert","lines":["    "]},{"start":{"row":82,"column":0},"end":{"row":82,"column":4},"action":"insert","lines":["    "]},{"start":{"row":83,"column":0},"end":{"row":83,"column":4},"action":"insert","lines":["    "]},{"start":{"row":84,"column":0},"end":{"row":84,"column":4},"action":"insert","lines":["    "]},{"start":{"row":85,"column":0},"end":{"row":85,"column":4},"action":"insert","lines":["    "]},{"start":{"row":86,"column":0},"end":{"row":86,"column":4},"action":"insert","lines":["    "]},{"start":{"row":87,"column":0},"end":{"row":87,"column":4},"action":"insert","lines":["    "]},{"start":{"row":88,"column":0},"end":{"row":88,"column":4},"action":"insert","lines":["    "]},{"start":{"row":89,"column":0},"end":{"row":89,"column":4},"action":"insert","lines":["    "]},{"start":{"row":90,"column":0},"end":{"row":90,"column":4},"action":"insert","lines":["    "]},{"start":{"row":91,"column":0},"end":{"row":91,"column":4},"action":"insert","lines":["    "]},{"start":{"row":92,"column":0},"end":{"row":92,"column":4},"action":"insert","lines":["    "]},{"start":{"row":93,"column":0},"end":{"row":93,"column":4},"action":"insert","lines":["    "]},{"start":{"row":94,"column":0},"end":{"row":94,"column":4},"action":"insert","lines":["    "]},{"start":{"row":95,"column":0},"end":{"row":95,"column":4},"action":"insert","lines":["    "]},{"start":{"row":96,"column":0},"end":{"row":96,"column":4},"action":"insert","lines":["    "]},{"start":{"row":97,"column":0},"end":{"row":97,"column":4},"action":"insert","lines":["    "]},{"start":{"row":98,"column":0},"end":{"row":98,"column":4},"action":"insert","lines":["    "]},{"start":{"row":99,"column":0},"end":{"row":99,"column":4},"action":"insert","lines":["    "]},{"start":{"row":100,"column":0},"end":{"row":100,"column":4},"action":"insert","lines":["    "]},{"start":{"row":101,"column":0},"end":{"row":101,"column":4},"action":"insert","lines":["    "]}],[{"start":{"row":154,"column":4},"end":{"row":154,"column":8},"action":"insert","lines":["    "],"id":24}],[{"start":{"row":154,"column":8},"end":{"row":154,"column":12},"action":"insert","lines":["    "],"id":25}],[{"start":{"row":154,"column":12},"end":{"row":154,"column":13},"action":"insert","lines":["p"],"id":26},{"start":{"row":154,"column":13},"end":{"row":154,"column":14},"action":"insert","lines":["r"]},{"start":{"row":154,"column":14},"end":{"row":154,"column":15},"action":"insert","lines":["i"]},{"start":{"row":154,"column":15},"end":{"row":154,"column":16},"action":"insert","lines":["n"]},{"start":{"row":154,"column":16},"end":{"row":154,"column":17},"action":"insert","lines":["t"]}],[{"start":{"row":154,"column":17},"end":{"row":154,"column":19},"action":"insert","lines":["()"],"id":27}],[{"start":{"row":154,"column":18},"end":{"row":154,"column":19},"action":"insert","lines":["h"],"id":28},{"start":{"row":154,"column":19},"end":{"row":154,"column":20},"action":"insert","lines":["i"]},{"start":{"row":154,"column":20},"end":{"row":154,"column":21},"action":"insert","lines":["g"]},{"start":{"row":154,"column":21},"end":{"row":154,"column":22},"action":"insert","lines":["h"]},{"start":{"row":154,"column":22},"end":{"row":154,"column":23},"action":"insert","lines":["l"]},{"start":{"row":154,"column":23},"end":{"row":154,"column":24},"action":"insert","lines":["i"]},{"start":{"row":154,"column":24},"end":{"row":154,"column":25},"action":"insert","lines":["g"]},{"start":{"row":154,"column":25},"end":{"row":154,"column":26},"action":"insert","lines":["h"]},{"start":{"row":154,"column":26},"end":{"row":154,"column":27},"action":"insert","lines":["t"]}],[{"start":{"row":154,"column":18},"end":{"row":154,"column":27},"action":"remove","lines":["highlight"],"id":29},{"start":{"row":154,"column":18},"end":{"row":154,"column":35},"action":"insert","lines":["highlight_reviews"]}]]},"ace":{"folds":[],"scrolltop":445.20000000000016,"scrollleft":0,"selection":{"start":{"row":157,"column":0},"end":{"row":157,"column":0},"isBackwards":false},"options":{"guessTabSize":true,"useWrapMode":false,"wrapToView":true},"firstLineState":{"row":11,"state":"start","mode":"ace/mode/python"}},"timestamp":1709388940941,"hash":"4d79a70f67e7c783abd44a85f17a795d2c290143"}