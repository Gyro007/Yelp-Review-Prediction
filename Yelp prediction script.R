# Set working directory
setwd("C:\\Users\\Asus\\Documents\\Chin Howe\\UG\\Modules\\Year 03_23-24\\EC349 Data Science For Economists\\Coursework\\Yelp Dataset\\")

# Load required libraries
library(jsonlite)
library(caret)
library(tidyverse)
library(gam)
library(tidytext)
library(textdata)
library(dplyr)
library(randomForest)

# Load required datasets
load("yelp_review_small.Rda")
load("yelp_user_small.Rda")
business_data <- stream_in(file("C:\\Users\\Asus\\Documents\\Chin Howe\\UG\\Modules\\Year 03_23-24\\EC349 Data Science For Economists\\Coursework\\Yelp Dataset\\yelp_academic_dataset_business.json"))

# Data cleaning for user data
user_data_small_trimmed <- user_data_small[, c(1,3,5,6,7,10,11)]
user_data_small_trimmed <- user_data_small_trimmed %>%
  filter_all(all_vars(!is.na(.)))
user_data_small_trimmed <- user_data_small_trimmed %>%
  filter_all(all_vars(. != 'None'))
user_data_small_trimmed <- user_data_small_trimmed %>% rename(user_review_count = review_count)
user_data_small_trimmed <- user_data_small_trimmed %>% rename(user_useful = useful)
user_data_small_trimmed <- user_data_small_trimmed %>% rename(user_funny = funny)
user_data_small_trimmed <- user_data_small_trimmed %>% rename(user_cool = cool)
user_data_small_trimmed <- user_data_small_trimmed %>% rename(user_leniency = average_stars)

# Data cleaning for business data
business_data <- do.call(data.frame, business_data)

business_data_trimmed <- business_data[, c(1,4,7,8,9,10,15,35)]
business_data_trimmed <- business_data_trimmed %>%
  filter_all(all_vars(!is.na(.)))
business_data_trimmed <- business_data_trimmed %>%
  filter_all(all_vars(. != 'None'))

business_data_trimmed <- business_data_trimmed %>% rename(business_rating = stars)
business_data_trimmed <- business_data_trimmed %>% rename(business_review_count = review_count)

# Replace strings with countable numeric for modelling later in 'attributes.NoiseLevel'
noise_counts <- table(business_data_trimmed$attributes.NoiseLevel)
noise_counts
business_data_trimmed$attributes.NoiseLevel <- gsub("u?'very_loud'|very_loud", "3", business_data_trimmed$attributes.NoiseLevel)
business_data_trimmed$attributes.NoiseLevel <- gsub("u?'loud'|loud", "2", business_data_trimmed$attributes.NoiseLevel)
business_data_trimmed$attributes.NoiseLevel <- gsub("u?'average'|average", "1", business_data_trimmed$attributes.NoiseLevel)
business_data_trimmed$attributes.NoiseLevel <- gsub("u?'quiet'|quiet", "0", business_data_trimmed$attributes.NoiseLevel)

# K-means clustering inspection
latitude <- business_data_trimmed$latitude
longitude <- business_data_trimmed$longitude
ggplot(business_data_trimmed, aes(x = longitude, y = latitude)) +
  geom_point(alpha = 0.5) +
  labs(title = "Data Points by Latitude and Longitude",
       x = "Longitude", y = "Latitude") +
  theme_minimal()

# Data Mastersheet
merged_data <- merge(review_data_small, user_data_small_trimmed, by = "user_id", all.x = FALSE)
merged_data <- merge(merged_data, business_data_trimmed, by = "business_id", all.x = FALSE)

# 80-20 train-test-split
# Training set
set.seed(1)
test_index <- createDataPartition(merged_data$stars, times = 1, p = 0.333, list = FALSE)
train_set <- merged_data[-test_index,]
train_set <- train_set %>% 
  slice(1:20000)

# Test set
test_set <- merged_data[test_index,]
test_set <- test_set %>% 
  slice(1:10000)

# Sentiment analysis
data(stop_words)
obtain_sentiment = function(input) {
  input %>%
    mutate(text_1 = text) %>%
    unnest_tokens(word,text_1) %>% # detach out all the word
    anti_join(stop_words, by="word") %>% # Remove stop words that do not convey feeling
    inner_join(get_sentiments("bing")) %>% # positive/negative classification using bing
    mutate(sentiment_score = ifelse(sentiment == "positive", 1, -1)) %>% # (1) for positive, (-1) for negative
    mutate(overall_score = 0) %>% # start overall score count from 0
    group_by(review_id) %>% # determine sentiment score for each review_id
    mutate(overall_score = sum(sentiment_score)) %>%
    ungroup() %>%
    distinct(review_id, .keep_all=TRUE) %>% # removing all extra row. 1 row for each review
    select(-c(word,sentiment,sentiment_score)) # removing extra columns
}
only_stop_words = function(input1, input2) {
  input1 %>%
    mutate(overall_score = 0) %>%
    subset(! review_id %in% input2$review_id) %>%
    bind_rows(input2)
} 
test_set <- only_stop_words(test_set,obtain_sentiment(test_set))
train_set <- only_stop_words(train_set,obtain_sentiment(train_set))

# Inspection of sentiment distribution for test
hist(test_set$overall_score)

# Model 1: Simple Linear Regression
# Training the Linear Regression model
model <- train(stars ~ overall_score + useful + funny + cool + fans + user_review_count + user_useful + user_funny + user_cool + user_leniency + business_rating + business_review_count + attributes.RestaurantsPriceRange2   + attributes.NoiseLevel,
               method = "lm",
               data = train_set)
# Making predictions on the test set
predictions <- predict(model, newdata = test_set)
model_eval <- cbind(test_set$stars, predictions)
colnames(model_eval) <-c("actual", "predicted")
model_eval <- as.data.frame(model_eval)
model_eval_rounded <- model_eval %>%
  mutate(predicted = round(as.numeric(as.character(model_eval$predicted)), digits = 0))
mean(model_eval_rounded$predicted == model_eval_rounded$actual)
remove(model, model_eval, model_eval_rounded, predictions)

# Model 2: Random Forest to reduce variance
# Training the Random Forest model
model <- randomForest(stars ~ overall_score + useful + funny + cool + user_review_count + user_useful + user_funny + user_cool + fans + user_leniency + business_rating + business_review_count + attributes.RestaurantsPriceRange2 + attributes.NoiseLevel,
                      data = train_set)
# Making predictions on the test set
predictions <- predict(model, newdata = test_set)
model_eval <- cbind(test_set$stars, predictions)
colnames(model_eval) <-c("actual", "predicted")
model_eval <- as.data.frame(model_eval)
model_eval_rounded <- model_eval %>%
  mutate(predicted = round(as.numeric(as.character(model_eval$predicted)), digits = 0))
mean(model_eval_rounded$predicted == model_eval_rounded$actual)
