path = "~/Desktop/kaggle/titanic/"
setwd(path)

library(tidyverse)
library(randomForest)

# 1. read data and clean data
## 1.1. read data
df.train.raw = read.csv("data/train.csv", header = TRUE,
                        na.strings = c(""), stringsAsFactors = FALSE)

df.test.raw = read.csv("data/test.csv", header = TRUE,
                       na.strings = c(""), stringsAsFactors = FALSE)

df.train = df.train.raw
df.test = df.test.raw

## 1.2. diagnostics 
### training
#### missing
v.miss.pct = sapply(df.train, function(x) round(sum(is.na(x)) / length(x), 4))

p1 = ggplot(data.frame(var = names(v.miss.pct),
                       miss_pct = v.miss.pct)) +
  geom_bar(aes(x = var, y = miss_pct), stat = "identity") + 
  coord_flip() + labs(title = "Missing % - Training Data")

p1

#### data format
summary(df.train)
sapply(df.train, function(x) mode(x))

c(length(unique(df.train$Name)), dim(df.train)[1]); head(df.train$Name, 10);
unique(df.train$Sex)
c(length(unique(df.train$Ticket)), dim(df.train)[1]); head(df.train$Ticket, 10);
unique(df.train$Embarked)

#### clean data (training)
#### "Cabin" and "Age" too much missing info, can't use as indenpendent vars
#### "PassengerId", "Name", "Ticket" are not helpful variables
#### remove 2 rows with missing "Embarked"
#### turn "Survied", "Pclass", "Sex", "Embarked" into factor

df.train = df.train %>% 
  select(-c(PassengerId, Cabin, Age, Ticket, Name)) %>%
  filter(!is.na(Embarked)) %>% 
  mutate(Survived = as.factor(Survived),
         Pclass = as.factor(Pclass),
         Sex = as.factor(Sex),
         Embarked = as.factor(Embarked))

# 2. model training

## 2.1. model fitting (random forests)
set.seed(1)
rf.titanic = randomForest(Survived ~ ., data = df.train, importance = TRUE)
rf.titanic

## 2.2. plot error rates
layout(matrix(c(1,2), 1, 2, byrow = TRUE),
       widths = c(4, 1))
par(mar = c(5.1, 4.1, 4.1, 0))
plot(rf.titanic)
par(mar = c(5.1, 0, 4.1, 2.1))
plot(0:1, ann = FALSE, type = "n", axes = FALSE)
legend("top", colnames(rf.titanic$err.rate),col=1:3,cex=0.4,fill=1:3)

## 2.3. variable importance
importance(rf.titanic)
varImpPlot(rf.titanic)

## 2.4. model fitting (logistic regression)
glm.fits = glm(Survived ~ ., data = df.train, family = binomial)
summary(glm.fits)

## as observed, Parch, Fare, are not significant
glm.fits = glm(Survived ~., data = df.train %>% select(-c(Parch, Fare)), 
               family = binomial)
summary(glm.fits)

# 3. predict

## 3.1. first clean up testing data
summary(df.train %>% select(-Survived)) # check the data format of training data

y.var = "Survived"
x.vars = names(df.train)[names(df.train) != y.var]

df.test = df.test %>% select(one_of(c(x.vars, "PassengerId")))

### check missing
### "Fare" is missing for 1 loan, using the average value 
### to imupte
sapply(df.test, function(x) sum(is.na(x)))

df.test = df.test %>%
  mutate(Fare = case_when(is.na(Fare) ~ mean(df.test$Fare, na.rm = TRUE),
                          TRUE ~ Fare))

### check data format
### "Pclass", "Sex", "Embarked" should be factor
summary(df.test)
unique(df.test %>% pull(Embarked))

df.test = df.test %>%
  mutate(Pclass = as.factor(Pclass),
         Sex = as.factor(Sex),
         Embarked = as.factor(Embarked))

## 3.2. now predict
### 3.2.1 using random forests
pred.rf = predict(rf.titanic, newdata = df.test)
result.rf = cbind(df.test %>% select(PassengerId), pred.rf) %>%
  rename(Survived = pred.rf)

### 3.2.2 using logistic regression
pred.glm = predict(glm.fits, newdata = df.test, type = "response")
result.glm = cbind(df.test %>% select(PassengerId), pred.glm) %>%
  mutate(Survived = case_when(pred.glm > 0.5 ~ 1,
                              TRUE ~ 0)) %>%
  select(PassengerId, Survived)

## 3.3. write our result
# write.csv(result.rf, "submission/submission_1.csv", row.names = FALSE)
write.csv(result.glm, "submission/submission_2.csv", row.names = FALSE)
