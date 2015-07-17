# Based on http://deblivingdata.net/machine-learning-the-digit-recognition-problem/

# Read and format the training data
digits = read.csv('input/train.csv', header=TRUE)
digits[1,]
digits$label <- as.factor(digits$label)

# Split the training data into a true training group and a smoketest group
library(caTools)
split <- sample.split(digits$label, SplitRatio=.6)
digits.train <- subset(digits, split==TRUE)
digits.test <- subset(digits, split==FALSE)

# Apply a random forest to the training set
library(randomForest)
forest <- randomForest(label ~ ., 
                      data=digits.train, 
                      ntree=100, 
                      do.trace=TRUE,
                      mtry=60,
                      nodesize=10) 

# Plot the most important pixels according to the forest
varImpPlot(forest, sort=TRUE) 

# Apply the prediction to the initial smoketest data
predictions = predict(forest, newdata=digits.test)

# Consolidate prediction and actual smoketest data for comparison
check <- data.frame(predicted=predictions, actual=digits.test$label)
head(check)

# Calculate the accuracy of the predictions on the temporary smoketest set
nrow(check[check$actual==check$predicted,])/nrow(check)

##################
# Real Test Data #
##################
# Read the data
testdata <- read.csv('input/test.csv', header=TRUE)

# Apply the prediction to the initial smoketest data
testpredictions <- predict(forest, newdata=testdata)

# Write to csv
write.table(testpredictions, file = "output/predictions.csv", row.names = TRUE, col.names = c("\"ImageId\",\"Label\""), quote=FALSE, sep=",")
