# Big Data Analytics and Visualization: NLP Lab Exercises Report

## Introduction
This report presents two Natural Language Processing lab exercises completed in Python. The first lab focuses on text preprocessing, word frequency analysis, visualization, and TF-IDF representation. The second lab applies the same preprocessing ideas to hotel review sentiment analysis using a supervised machine learning classifier.

## Objectives
- Load text datasets into pandas DataFrames.
- Clean and tokenize text using NLTK.
- Remove punctuation, stopwords, and non-alphabetic tokens.
- Create word frequency tables, bar charts, word clouds, and word length plots.
- Apply TF-IDF and bigram TF-IDF vectorization.
- Train and evaluate a Multinomial Naive Bayes sentiment classifier.
- Export analysis results and visualizations for submission.

## Tools and Libraries
The project uses Python in a Jupyter Notebook with pandas for data handling, matplotlib and seaborn for plotting, NLTK for tokenization and stopwords, wordcloud for word cloud visualization, and scikit-learn for TF-IDF vectorization, train-test splitting, Naive Bayes modeling, and evaluation metrics.

## LAB_1 Methodology
The LAB_1 text dataset of 30 sentences was loaded into a pandas DataFrame. Text was converted to lowercase, punctuation was removed, words were tokenized with NLTK, English stopwords were removed, and cleaned text strings were created from the remaining tokens. Word frequencies were calculated using `Counter`. The analysis then produced a top-five word bar chart, word cloud, and word length distribution plot. TF-IDF was applied to the cleaned text, followed by bigram TF-IDF using `TfidfVectorizer(ngram_range=(1, 2))`. Extra sentences were added to observe how word frequencies and word clouds changed. Additional manual stopwords such as `data`, `fields`, and `related` were also removed.

## LAB_1 Results and Interpretation
The original LAB_1 word frequency table was exported as `word_frequency_lab1.csv`. The most frequent words were `data, learning, technology, ai, digital`. These terms show that the corpus is mainly about education, technology, data, and machine learning. The word cloud visually confirmed the same pattern. The word length distribution showed that many words were medium-length content words, which is expected after stopword removal. TF-IDF produced a numerical matrix showing the importance of each term in each sentence, while bigram TF-IDF captured meaningful phrases such as machine learning, data science, and decision making.

## LAB_2 Methodology
The hotel review dataset of 60 reviews was loaded into a DataFrame with `Text` and `Sentiment` columns. The labels were balanced with 20 positive, 20 negative, and 20 neutral reviews. The sentiment distribution was visualized to inspect class balance. Reviews were lowercased, tokenized, stripped of punctuation, filtered to alphabetic tokens, and cleaned of stopwords. The cleaned text was transformed using TF-IDF. The dataset was split into training and testing sets using `test_size=0.2` and `random_state=42`. A Multinomial Naive Bayes classifier was trained and evaluated on the test set.

## LAB_2 Results and Interpretation
The classifier achieved an accuracy of `0.17` on the test set. The classification report and confusion matrix were generated in the notebook. Test results were exported as `hotel_sentiment_results.csv`. The sentiment distribution was balanced, with 20 positive, 20 negative, and 20 neutral reviews, supporting fairer model training. The top positive words reflected favorable service and comfort themes, while top negative words reflected problems with service, cleanliness, noise, and maintenance. The new review, `The staff were very helpful and the room was clean`, was predicted as `positive`.

## Big Data and Power BI Connection`r`nThe LAB_1 workflow also connects to big data analytics. pandas DataFrames correspond to Spark DataFrames, word frequency counting corresponds to MapReduce-style aggregation, and TF-IDF can be scaled with Spark MLlib. The exported `word_frequency_lab1.csv` can be imported into Power BI to build bar charts and dashboard visuals.`r`n`r`n## Discussion
The labs show the importance of preprocessing before analysis or machine learning. Removing stopwords and punctuation makes vocabulary patterns easier to interpret. Frequency analysis and word clouds are useful for quick exploration, while TF-IDF provides a more structured numerical representation. In LAB_2, Naive Bayes was suitable for the small text classification task and produced understandable results.

## Limitations
The datasets are still limited for real-world modeling, but they are larger and more balanced than the initial lab version. With 60 hotel reviews, the 20% train-test split produces 12 test reviews, so accuracy is more informative than before but can still change with a different split. The model also uses a bag-of-words style representation, meaning it does not deeply understand context, sarcasm, or complex sentiment expressions.

## Conclusion
Both labs were completed successfully. LAB_1 demonstrated text cleaning, frequency analysis, visualizations, and TF-IDF features. LAB_2 demonstrated a complete sentiment analysis workflow from data loading to preprocessing, model training, evaluation, visualization, result export, and prediction on a new review.

## References
- Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python.
- pandas documentation: https://pandas.pydata.org/
- Matplotlib documentation: https://matplotlib.org/
- seaborn documentation: https://seaborn.pydata.org/
- wordcloud documentation: https://amueller.github.io/word_cloud/

