##### Breast Cancer #####

# Libraries:
library(corrplot)
library(caTools)
library(ROSE)
library(rpart)
library(randomForest)
library(ElemStatLearn)
library(class)
library(MASS)
library(neuralnet)


# Importing the dataset
url <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/00451/dataR2.csv'
dataset <- read.csv(url)

### Data Processing ###

head(dataset)
str(dataset)

# Looking at dataset
par(mfrow=c(3, 4))
hist(dataset$Age, main = "Age", xlab = "age", ylab = "frequency", 
     xlim = c(0, 100), freq = T, col = "blue", border = "navy")
hist(dataset$BMI, main = "Body Max Index", xlab = "BMI", ylab = "frequency",
     xlim = c(0, 50), freq = T, col = "blue", border = "navy")
hist(dataset$Glucose, main = "Glucose", xlab = "glucose", ylab = "frequency",
     xlim =c(0, 210), freq = T, col = "blue", border = "navy")
hist(dataset$Insulin, main = "Insulin", xlab = "insulin", ylab = "frequency",
     xlim = c(0, 60), freq = T, col = "blue", border = "navy")
hist(dataset$HOMA, main = "Homeostatic Model Assessment", xlab = "HOMA", ylab = "frequency",
     xlim = c(0, 30), freq = T, col = "blue", border = "navy")
hist(dataset$Leptin, main = "Leptin", xlab= "leptin", ylab = "frequency",
     xlim = c(0, 100), freq = T, col = "blue", border = "navy")
hist(dataset$Adiponectin, main = "Adiponectin", xlab = "adiponectin", ylab = "frequency",
     xlim = c(0, 40), freq = T, col = "blue", border = "navy")
hist(dataset$Resistin, main = "Resistin", xlab = "resistin", ylab = "frequency",
     xlim = c(0, 90), freq = T, col = "blue", border = "navy")
hist(dataset$MCP.1, main = "Monocyte Chemoattractant Protein- 1", xlab = "MCP-1", ylab = "frequency",
     xlim = c(0, 1700), freq = T, col = "blue", border = "navy")

# Correlation
data_cor <- cor(dataset)
sort(data_cor, decreasing = T)
par(mfrow= c(1, 1))
corrplot(data_cor, method = c("square"), type = "upper", title = "Correlation", 
         tl.col = "black", col = brewer.pal(n = 10, name = "PuOr"))
##cor = 1 -> HOMA and Insulin
##cor = (0.6, 0.8) -> HOMA vs Glucose
##cor = (0.4, 0.6) -> BMI vs Leptin
##cor = (0.4, 0.6) -> Glucose vs Insulin
##cor = (0.2, 0.4) -> Glucose vs Calssification &
##                    Resistin vs MCP-1

# Encoding the target feature as factor
dataset$Classification = factor(dataset$Classification)
str(dataset)
##class = 1 -> healthy
##class = 2 -> patient

# Splitting the dataset into the Training set and Test set
set.seed(123)
split = sample.split(dataset$Classification, SplitRatio = 0.70)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Checking if dataset is adequate for classification
table(training_set$Classification)
prop.table(table(training_set$Classification))
##data set is not balancedd 44% healthy vs 56% patients

##How much is accuracy affected?
imb <- rpart(Classification~., data = training_set)
pred.imb <- predict(imb, newdata = test_set)
accuracy.meas(test_set$Classification, pred.imb[,2])
##precision = 0.78 -> there are false positives
##recall = 0.37 -> high number of false negatives
##F = 0.25 -> weak accuracy
roc.curve(test_set$Classification, pred.imb[,2], plotit = F)

# Balancing data -> 4 methods
##Over sampling
blnc_over <- ovun.sample(Classification ~ ., data = training_set, method = "over",N = 90)$data
table(blnc_over$Classification)
##Under sampling
blnc_under <- ovun.sample(Classification ~ ., data = training_set, method = "under",N = 72)$data
table(blnc_under$Classification)
##Over/under sampling
blnc_both <- ovun.sample(Classification ~ ., data = training_set, method = "both",N = 200, 
                         p= 0.5, seed = 1)$data
table(blnc_both$Classification)
##Synthetically sampling
data.rose <- ROSE(Classification ~ ., data = training_set, seed = 1)$data
table(data.rose$Classification)
str(data.rose)

## feature scaling
blnc_over[-10] = scale(blnc_over[-10])
blnc_under[-10] = scale(blnc_under[-10])
blnc_both[-10] = scale(blnc_both[-10])
data.rose[-10] = scale(data.rose[-10])
test_set[-10] = scale(test_set[-10])

##
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

## blnc_over
over <- as.data.frame(lapply(blnc_over[-10], normalize))

## blnc_ounder
under <- as.data.frame(lapply(blnc_under[-10], normalize))

## blnc_over
both <- as.data.frame(lapply(blnc_both[-10], normalize))

## blnc_over
rose <- as.data.frame(lapply(data.rose[-10], normalize))

#### K-Nearest Neighbors (K-NN) ####

##Elbow Method for finding the optimal number of clusters
set.seed(123)
par(mfrow = c(2,2))
## Compute and plot wss for k = 2 to k = 15.
k.max <- 15
## blnc_over
wss_over <- sapply(1:k.max, 
              function(k){kmeans(over, k, nstart=50,iter.max = 15 )$tot.withinss})
wss_over
plot(1:k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares", main = "Over")
# k = 6

## blnc_under
wss_under <- sapply(1:k.max, 
                   function(k){kmeans(under, k, nstart=50,iter.max = 15 )$tot.withinss})
wss_under
plot(1:k.max, wss_under,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares", main = "Under")
# k = 5

## blnc_both
wss_both <- sapply(1:k.max, 
                    function(k){kmeans(both, k, nstart=50,iter.max = 15 )$tot.withinss})
wss_both
plot(1:k.max, wss_both,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares", main = "Under/Over")
# k = 4

## data.rose
wss_rose <- sapply(1:k.max, 
                    function(k){kmeans(rose, k, nstart=50,iter.max = 15 )$tot.withinss})
wss_rose
plot(1:k.max, wss_rose,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares", main = "ROSE")
# k = 2 

## Fitting K-NN to the Training set and Predicting the Test set results
## blnc_over
y_pred_KNN_over = knn(train = over[, -10],
             test = test_set[, -10],
             cl = over[, 10],
             k = 6,
             prob = TRUE)
## blnc_under
y_pred_KNN_under = knn(train = under[, -10],
                      test = test_set[, -10],
                      cl = under[, 10],
                      k = 5,
                      prob = TRUE)
## blnc_both
y_pred_KNN_both = knn(train = both[, -10],
                       test = test_set[, -10],
                       cl = both[, 10],
                       k = 4,
                       prob = TRUE)
## blnc_rose
y_pred_KNN_rose = knn(train = rose[, -10],
                       test = test_set[, -10],
                       cl =rose[, 10],
                       k = 2,
                       prob = TRUE)

## Making the Confusion Matrix
## blnc_over
cm_knn_over = table(test_set[, 10], y_pred_KNN_over)
cm_knn_over
#correct = 10 vs incorrect = 25

## blnc_under
cm_knn_under = table(test_set[, 10], y_pred_KNN_under)
cm_knn_under
## correct =16 vs incorrect = 19

## blnc_both
cm_knn_both = table(test_set[, 10], y_pred_KNN_both)
cm_knn_both
## correct =17 vs incorrect = 18

## blnc_rose
cm_knn_rose = table(test_set[, 10], y_pred_KNN_rose)
cm_knn_rose
## correct =12 vs incorrect = 23

# ROC
par(mfrow = c(1, 1))
## AUC oversampling
roc.curve(test_set$Classification, y_pred_KNN_over, col="blue", main= "KNN")
#AUC = 0.722
par(new = T)
##AUC undersampling
roc.curve(test_set$Classification, y_pred_KNN_under, col = "purple", main= "KNN")
#AUC = 0.518
par(new = T)
##AUC both
roc.curve(test_set$Classification, y_pred_KNN_both, col = "dark orange", main= "KNN")
#AUC = 0.528
par(new = T)
##AUC Rose
roc.curve(test_set$Classification, y_pred_KNN_rose, col = "dark green",main= "KNN")
##AUC = 0.664
##over balance data offers sthe best accuracy


#### Logistic Regresion ####


#### Random Forest ####

## fitting Random Forest Classification to the Training set
set.seed(123)
rf_over = randomForest(x = over[-10],
                       y = over$Classification,
                       ntree = 500)
rf_under = randomForest(x = under[-10],
                        y = under$Classification,
                        ntree = 500)
rf_both = randomForest(x = both[-10],
                       y = both$Classification,
                       ntree = 500)
rf_rose = randomForest(x = rose[-10],
                       y = rose$Classification,
                       ntree = 500)

## predicting the test_set results
y_pred_over = predict(rf_over, newdata = test_set[-10])
y_pred_under = predict(rf_under, newdata = test_set[-10])
y_pred_both = predict(rf_both, newdata = test_set[-10])
y_pred_rose = predict(rf_rose, newdata = test_set[-10])

## Making the Confusion Matrix
cm_over = table(test_set[, 10], y_pred_over)
cm_under = table(test_set[, 10], y_pred_under)
cm_both = table(test_set[, 10], y_pred_both)
cm_rose = table(test_set[, 10], y_pred_rose)
cm_over #13 correct vs 22 incorrect
cm_under # 14 correct vs 21 incorrect
cm_both # 14 correct vs 21 incorrect
cm_rose # 14 correct vs 21 incorrect

# ROC
## AUC oversampling
roc.curve(test_set$Classification, y_pred_over, col="blue", main= "Random Forest")
#AUC = 0.707
par(new = T)
##AUC undersampling
roc.curve(test_set$Classification, y_pred_under, col = "purple", main= "Random Forest")
#AUC = 0.559
par(new = T)
##AUC both
roc.curve(test_set$Classification, y_pred_both, col = "dark orange", main= "Random Forest")
#AUC = 0.612
par(new = T)
##AUC Rose
roc.curve(test_set$Classification, y_pred_rose, col = "dark green",main= "Random Forest")


#### Logistic Regresion ####
# Fitting Logistic Regression to the Training set
## blnc_over
lr_over = glm(formula = Classification ~ .,
              family = binomial,
              data = blnc_over)

## blnc_under
lr_under = glm(formula = Classification ~ .,
               family = binomial,
               data = blnc_under)

##blnc_both
lr_both = glm(formula = Classification ~ .,
              family = binomial,
              data = blnc_both)

##data.rose
lr_rose = glm(formula = Classification ~ .,
              family = binomial,
              data = data.rose)

# Predicting the Test set results
## blnc_over
prob_pred_over = predict(lr_over, type = 'response', newdata = test_set[-10])
y_pred_over = ifelse(prob_pred_over > 0.5, 1, 0)
## blnc_under
prob_pred_under = predict(lr_under, type = 'response', newdata = test_set[-10])
y_pred_under = ifelse(prob_pred_under > 0.5, 1, 0)
## blnc_both
prob_pred_both = predict(lr_both, type = 'response', newdata = test_set[-10])
y_pred_both = ifelse(prob_pred_both > 0.5, 1, 0)
## data.rose
prob_pred_rose = predict(lr_rose, type = 'response', newdata = test_set[-10])
y_pred_rose = ifelse(prob_pred_rose > 0.5, 1, 0)

# Making the Confusion Matrix
cm_over = table(test_set[, 10], y_pred_over > 0.5)
cm_under = table(test_set[, 10], y_pred_under> 0.5 )
cm_both = table(test_set[, 10], y_pred_both> 0.5)
cm_rose = table(test_set[, 10], y_pred_rose> 0.5)
cm_over #12 correct vs 23 incorrect
cm_under # 13 correct vs 22 incorrect
cm_both # 15 correct vs 20 incorrect
cm_rose # 11 correct vs 24 incorrect


# ROC
## AUC oversampling
roc.curve(test_set$Classification, y_pred_over, col="blue", main= "Logistic Regresion")
#AUC = 0.669
par(new = T)
##AUC undersampling
roc.curve(test_set$Classification, y_pred_under, col = "purple", main= "Logistic Regresion")
#AUC = 0.638
par(new = T)
##AUC both
roc.curve(test_set$Classification, y_pred_both, col = "dark orange", main= "Logistic Regresion")
#AUC = 0.586
par(new = T)
##AUC Rose
roc.curve(test_set$Classification, y_pred_rose, col = "dark green",main= "Logistic Regresion")
#AUC = 0.696
