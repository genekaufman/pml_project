# Practical Machine Learning: Course Project
Gene Kaufman  
September 3, 2016  
## Introduction
Based on data obtained from a variety of fitness monitors, use ML to predict what type ("classe") of exercise were performed by the members of the test data. The data for this project was obtained by and provided for our use at http://groupware.les.inf.puc-rio.br/har 

### Loading and cleaning the data
Set some global options

```r
require(knitr)
opts_chunk$set(echo=TRUE, results="asis", warning=FALSE, message=FALSE)
```

Load necessary libraries

```r
library(caret)
library(AppliedPredictiveModeling)
```

Load data files

```r
training_csv <- "pml-training.csv";
RawData <- read.csv(training_csv, na.strings=c("NA","#DIV/0!",""));
testing_csv <- "pml-testing.csv";
testData <- read.csv(testing_csv, na.strings=c("NA","#DIV/0!",""));
```

Clean data

```r
#First 7 columns identify the user and/or the time of the record, which is not relevant for our purpose
RawData <- RawData[,8:160];
RawData$classe <- as.factor(RawData$classe);
testDataRaw <- testData[,8:160];

# The test data has many columns for which all rows are NA. Since that doesn't
#   help us, we will remove those columns from both the test data and the
#   training set (since we won't be able to predict on them when testing).
# Create logical vector based on whether all of the rows for a given column
# are all NA's. one way to do this is to sum the value of is.na for it. If
# the sum is zero, that indicates that there are no NA's, and therefore we want
# to use that field
non_na_fields <- names(testDataRaw[,colSums(is.na(testDataRaw)) == 0]);
# training data's last column is classe, not problem_id
non_na_fields_train <- sub("problem_id","classe",non_na_fields);

TrainingData <- RawData[,c(non_na_fields_train)];
TestData <- testDataRaw[,c(non_na_fields)];
```

Create data partitions on training data; use 25% as validation set

```r
set.seed(42);

trainIndex = createDataPartition(y=TrainingData$classe,p=0.75,list=FALSE);
training = TrainingData[trainIndex,];
validation = TrainingData[-trainIndex,];
```

### Train the data
Build a Random Forest model, as these typically have the best accuracy results. (I considered comparing multiple models, but my computer is not powerful enough to run multiple models in the time that I had to do the project.)


```r
# Since the model took 15+ hours to build on my computer, I've wrapped it so that I 
#	don't have to run it each time after I've built it the first time
trainfitDataFile <- "trainFit.RData";
if (file.exists(trainfitDataFile)) {
  load(trainfitDataFile);
}

if (!exists("trainFit")) {
  trainFit <- train(classe~ .,data=training,method="rf",prox=TRUE);
}
```

### Make predictions against the validation data
The out of sample error should be very small with a Random Forest model.

```r
predTrain <- predict(trainFit,validation);

# let's see how we did
confusionMatrix(predTrain,validation$classe);
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    4    0    0    0
##          B    0  943    3    0    1
##          C    0    2  849   11    1
##          D    0    0    3  793    0
##          E    0    0    0    0  899
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9949          
##                  95% CI : (0.9925, 0.9967)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9936          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9937   0.9930   0.9863   0.9978
## Specificity            0.9989   0.9990   0.9965   0.9993   1.0000
## Pos Pred Value         0.9971   0.9958   0.9838   0.9962   1.0000
## Neg Pred Value         1.0000   0.9985   0.9985   0.9973   0.9995
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1923   0.1731   0.1617   0.1833
## Detection Prevalence   0.2853   0.1931   0.1760   0.1623   0.1833
## Balanced Accuracy      0.9994   0.9963   0.9948   0.9928   0.9989
```

### Make predictions against the test data

```r
predTest <- predict(trainFit,testData);

# Return results for use in the quiz
for (i in 1:length(predTest)) {
  print(paste(i,predTest[i]));
}
```

```
## [1] "1 B"
## [1] "2 A"
## [1] "3 B"
## [1] "4 A"
## [1] "5 A"
## [1] "6 E"
## [1] "7 D"
## [1] "8 B"
## [1] "9 A"
## [1] "10 A"
## [1] "11 B"
## [1] "12 C"
## [1] "13 B"
## [1] "14 A"
## [1] "15 E"
## [1] "16 E"
## [1] "17 A"
## [1] "18 B"
## [1] "19 B"
## [1] "20 B"
```
