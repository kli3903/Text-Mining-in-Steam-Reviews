library(tidyverse)
library(tm)
library(SnowballC)
library(ggplot2)
library(RColorBrewer)
library(wordcloud)
library(syuzhet)
library(recommenderlab)
library(sentimentr)
library(dplyr)
library(cluster)
library(glmnet)

#setwd("C:/Users/pilow/OneDrive/Desktop/stats295 machine learning/project")
data <- read.csv("train.csv")

dim(data)
names(data)
head(data)
unique(data$title)

data$word_count <- length(strsplit(data$user_review," "))
for (x in 1:length(data$review_id)){
  data[x,]$word_count <- length(strsplit(data[x,]$user_review," ")[[1]])
}
head(data)

writeLines(strwrap(data[1, "user_review"], width = 65))

Reviews <- Corpus(VectorSource(data$user_review))

Reviews <- tm_map(Reviews, tolower)
Reviews <- tm_map(Reviews, removeWords, stopwords("english"))
Reviews <- tm_map(Reviews,removeWords,
                   c("game", "will", "this","just", "even", "can",
                     "thing", "time", "the", "realli", "player",
                     "much", "make", "ear", "you", "use", "tri",
                     "still", "peopl", "one", "now", "get", "card",
                     "access", "play"))
Reviews <- tm_map(Reviews, removePunctuation)
Reviews <- tm_map(Reviews, removeNumbers)
Reviews <- tm_map(Reviews, stemDocument)
writeLines(strwrap(lapply(Reviews[1], as.character), width = 75))

dtm1 <- TermDocumentMatrix(Reviews)
dtm_m <- as.matrix(dtm1)
dtm_v <- sort(rowSums(dtm_m),decreasing=TRUE)
dtm_d <- data.frame(word = names(dtm_v),freq=dtm_v)
head(dtm_d)
ggplot(dtm_d[1:25,], aes(x=word, y=freq)) + geom_bar(stat = "identity") + coord_flip()

threshold <- .01*length(Reviews)
words.10 <- findFreqTerms(dtm1, lowfreq=threshold)
length(words.10)

dtm.10 <- DocumentTermMatrix(Reviews, control = list(dictionary = words.10))
dtm.10

term.freq <- colSums(as.matrix(dtm.10))
words25 <- sort(term.freq, decreasing = TRUE)[1:25]
df <- data.frame(term = names(words25), freq = words25)
df$term <- factor(df$term, levels = df$term[order(df$freq)])
ggplot(df, aes(x=term, y=freq)) + geom_bar(stat = "identity") + coord_flip()

m <-as.matrix(dtm.10)
word.freq <- sort(colSums(m), decreasing = TRUE)
pal <- brewer.pal(8, "Dark2")
wordcloud(words = names(word.freq), freq = word.freq, min.freq = 2500,
          random.order = FALSE, colors = pal, scale = c(1,.5))


length(unique(data$title))

rec_num <- data %>% 
  group_by(title) %>% 
  summarize(reviews = length(unique(review_id)))
rec_num <- data.frame(rec_num)
rec_num
ggplot(rec_num, aes(x = title, y = reviews, fill = 'red')) + geom_bar(stat = "identity", show.legend = FALSE) + coord_flip() + geom_text(aes(label = reviews))

rec_year <- data %>% 
  group_by(year) %>% 
  summarize(user_suggestion = sum(user_suggestion))

rec_year
rec_year$year <- as.character(rec_year$year)
rec_year <- data.frame(rec_year)[1:8,]

rec_year %>%
  ggplot(aes(x = year, y = user_suggestion)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "User Suggestion by Year")

rec_id <- data %>% 
  group_by(year) %>% 
  summarize(count = length(unique(review_id)))
rec_id$year <- as.character(rec_id$year)
rec_id <- data.frame(rec_id)[1:8,]
rec_id

rec_id %>%
  ggplot(aes(x = year, y = count)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Number of Reviews by Year")

data[1,]$user_review
sentiment_by(data[1,]$user_review)
data$ave_sentiment <- sentiment_by(data$user_review)$ave_sentiment
head(data)
extract_sentiment_terms(data$user_review)
head(data)
names(data)
mean_by_games <- data %>%
  group_by(title) %>%
  summarise_at(vars(ave_sentiment), list(name = mean))
mean_by_games <- data.frame(mean_by_games)
mean_by_games

mean_by_games %>%
  ggplot(aes(name)) +
  geom_density(fill = "gray")

positiveRev <- subset(data, user_suggestion == "1")
head(positiveRev)
length(unique(positiveRev$review_id))

negativeRev <- subset(data, user_suggestion == "0")
head(negativeRev)
length(unique(negativeRev$review_id))

names(data)

positiveRev %>%
  ggplot(aes(x = word_count)) +
  geom_histogram(color = 'black', fill = 'blue', bins = 50) +
  labs(title = "Word Count for Positive Reviews")

negativeRev %>%
  ggplot(aes(x = word_count)) +
  geom_histogram(color = 'black', fill = 'red', bins = 50) +
  labs(title = "Word Count for Negative Reviews")

n <- nrow(data)
data$user_suggestion <- as.factor(data$user_suggestion)
test.index = sample(n, 10000)

data.test=data[test.index, -c(1:3)]
data.train=data[-test.index, -c(1:3)]
data2 <- as.matrix(term.freq)
data2
y <- data.train[,2]
x <- as.matrix(data.train[,-c(2)])
set.seed(2)
result.lasso <- cv.glmnet(x, y, alpha = 0.99, family = "binomial")
beta.lasso <- coef(result.lasso, s = "lambda.1se")
beta <- beta.lasso[which(beta.lasso != 0),]