## **Overview**

Over the years, the society has increasingly consumed news on social Media sites like Facebook, Twitter, Youtube, Instagram etc. Social Media companies have been under intense scrutiny to stem the flow of fake and misleading information/news on their websites. The simplest approach to this problem is creating a fake news detection system that can help classify the news articles and flag it. NLP methods to detect fake news have consistently improved over time and we want to explore some methods to build our own model. With available labeled datasets, we want to build an NLP classification model to determine if a new article is fake.

We believe that this kind of system will be a great value add to any platform that allows users to share news articles published on a different website. Even a small tag indicating that a news article might be fake, allows the user to view the article from a different perspective.


## **Our Target Customer**

Our aim is to build a classification model that can help identify fake news articles. With this classification, we want to provide consumers of news articles with an indication that the news that they are reading might be fake. 

Our target users are mainly divided into two categories:
1. Web Browser Plugin developers that allow users to download their plugin to be used on any website to indicate if a particular article/text may be fake
2. Social Media websites that allow users to share news articles that may originate from websites that have satire based news articles or click bait articles. Platforms like Facebook and Twitter allow users to share links for articles and our model, can be used by these companies to create a new feature that allows users to see if a shared post may be fake or not through smart tagging. 




## **Type of learning that we have investigated**

One of the most challenging problems in ML is one that deals with Natural Language Processing. We have explored supervised learning techniques for a classificatioon problem. Specifically, we looked into Random Forest Classifier and LSTM based Classifer to detect Fake news. The Random Forest Model provides and explainable approach as compared to the LSTM model and we discuss the pros and cons of using both these models later. 

## **The domain of the problem we are trying to solve and identifying the T, P, and E. Discussion on how we assessed our work (the P part)**

The Domain that we plan to investigate is NLP classification to detect fake news using a supervised approach. 

The Task ‘T’ is identifying whether a piece of text is fake or not.
The Experience ‘E’ will come from the trained model with labeled data.
The Performance ‘P’ will be measured as the accuracy of classifying unseen data as fake. 

In order to determine news as real or fake, we will be making use of a number of models for Classification like:

- Random Forest model
- LSTM Model - using transfer learning from GloVe Embeddings

The best measure of performance for classification models would be to measure classification accuracy, precison, recall and F1 scores. We want to focus on reducing false positives since we want to avoid wrongly classifying fake news as "real" for each of the models.
But in this case, we also want to reduce the number of false negatives which would classify an article with real fact based news as false. 

We find that in this special case, the importance of one kind of error may not be more important than the other. It is in the best interest to reduce both. We understand that maybe in news articles which discuss Covid or election related information, both false negatives and false positives are undesirable. This is why we try to look at improving both precision and recall. We also discuss measure to overcome to overcome this issue programmatically later. 

## **Motivation for this work**

A little under half (48%) of U.S. adults say they get news from social media “often” or “sometimes” (https://www.pewresearch.org/journalism/2021/09/20/news-consumption-across-social-media-in-2021/). Most social media posts about news contain a couple of headlines and a link that then leads to a news article. With the current overhaul of Twitter by Elon Musk, the midterm elections in the United States and multiple world events capturing the attention of audiences around the world, it is important to have a way to curb the spread of misinformation. After removing the barriers for getting verified on Twitter, the amount of misinformation on Twitter has exponentially increased leading to repercussions on the global stock market as well. According to this article, https://mashable.com/article/twitter-fake-verified-posts-worse-elon-musk the number of fake accounts has risen and this has led to a downward spiral much more dangerous than anything we have seen before. 

At this time, even a small tag or indication that a news article “might be fake” will really help consumers. We believe that by providing such information to the users, the platforms can put some power in the hands of the user to decide whether they want to believe a particular piece of information. The warning could potentially curb the rampant spread of information. One step further, the platforms can limit the resharing of news article links that are tagged as possibly fake.

## **Briefly discuss the data you plan to use. Where did you find it? What is its structure?**


There are multiple datasets available online and we have decided to experiment on 2 different datasets:



1.  Dataset 1: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=True.csv
>* This source has 2 files  - one file containing “real” articles and another file containing “fake” news articles.<br>
>* The real news file has 21,192 unique articles and the fake news file has 22851 unique articles.<br> 
>* The dataset is very basic as it just contains the Article title, the article text content and Date of the article.<br>
>* We aim to feature engineer and extract other features that might help with our classification.


 
2.   Dataset 2: https://www.kaggle.com/datasets/mrisdal/fake-news
>* The data from this source contains text and metadata from 244 websites and represents 12,999 posts in total from the webhose.io API, with 12,999 unique values.<br>
>* The dataset contains the author, publish date, news title. article text, language, site url, article type, and a label to indicate real or fake for verification.<br> 

On exploration of the datasets, we noticed that the first dataset is much better suited to our needs due to the following reasons:
1. Higher number of data points (~45k records in Dataset 1 and ~2k records in Dataset 2)
2. More Variety of data points (Since there are more data points and we have a subject column, we can say that the first dataset has more variability in the data which is better for ML models)

Thus, we have implemented all our work for the first dataset
