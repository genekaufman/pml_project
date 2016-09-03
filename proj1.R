
library(caret)
library(AppliedPredictiveModeling)

#rm(list = ls())
rm(list=ls()[ls()!="trainFit" & ls()!="trainFit" & ls()!="trainFit"])

training_csv <- "pml-training.csv";
#RawData <- read.csv(training_csv, colClasses = "character");
RawData <- read.csv(training_csv, na.strings=c("NA","#DIV/0!",""));
#First 7 columns identify the user and/or the time of the record, which is not relevant for our purpose
RawData <- RawData[,8:160];
RawData$classe <- as.factor(RawData$classe);

testing_csv <- "pml-testing.csv";
testData <- read.csv(testing_csv, na.strings=c("NA","#DIV/0!",""));
testData <- testData[,8:160];
#testData$classe <- as.factor(testData$classe);

# The test data has many columns for which all rows are NA. Since that doesn't
#   help us, we will remove those columns from both the test data and the
#   training set (since we won't be able to predict on them when testing).
# Create logical vector based on whether all of the rows for a given column
# are all NA's. one way to do this is to sum the value of is.na for it. If
# the sum is zero, that indicates that there are no NA's, and therefore we want
# to use that field
non_na_fields <- names(testData[,colSums(is.na(testData)) == 0]);
non_na_fields_train <- sub("problem_id","classe",non_na_fields);
# training data's last column is classe, not problem_id
RawDataClean <- RawData[,c(non_na_fields_train)];
TestDataClean <- testData[,c(non_na_fields)];

set.seed(42);

trainIndex = createDataPartition(y=RawDataClean$classe,p=0.75,list=FALSE);
training = RawDataClean[trainIndex,];
validation = RawDataClean[-trainIndex,];

if (!exists("trainFit")) {
  print("generating trainFit");
  start.time <- Sys.time()
  trainFit <- train(classe~ .,data=training,method="rf",prox=TRUE);
  end.time <- Sys.time()
  time.taken <- paste( as.numeric(round(end.time - start.time,3)*1000), "milliseconds");
  print(time.taken);

}

if (!exists("predTrain")) {
  print("generating predTrain");
  predTrain.time.start <- Sys.time()
  predTrain <- predict(trainFit,validation);
  predTrain.time.end <- Sys.time()
  time.taken <- paste( as.numeric(round(predTrain.time.end - predTrain.time.start,3)*1000), "milliseconds");
  print(time.taken);

}

confusionMatrix(predTrain,validation$classe);

predTest <- predict(trainFit,testData);

for (i in 1:length(predTest)) {
  print(paste(i,predTest[i]));
}


