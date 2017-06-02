# example classifier (RandomForest)

# preprocessing data
## Importing the dataset
dataset = read.csv('data_name.csv')
#dataset = dataset[1:5] # selecting variables if needed

## Splitting data (for testing)
# For testing you would consider the test_set as unseen data, 
# while training using e.g. Cross-Validation on the training_set
library(caTools)
split = sample.split(dataset$outcome, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

## Loading caret package
# using grid search for hyperparameter tuning
# caret has the advantage of applying grid search natively
#install.packages('caret') 
library(caret) 

## setting up Cross-Validation (if needed)
tr = trainControl(method = 'cv', number = 10) # k-fold (10-fold) CV
#tr = trainControl(method = 'LOOCV') # Leave One Out CV

## Setting up the classifier
# form sets outcome and (possible) predictor variables
# data refers to dataset
# trControl allows e.g. Cross-Validation
# metric is optional Classification default are 'Accuracy and Kappa', regression defaults are 'RMSE & Rsquared'
# method refers to model see: http://topepo.github.io/caret/train-models-by-tag.html
# Possible candidates are Random Forest, other tree-based models or Supoprt Vector Machines
# just for Random Forest it offers: 
classifier = train(form = outcome ~ .,
                   data = training_set,
                   method = 'rf')
classifier # tunes the model and gives you the output
classifier$results
classifier$bestTune # optimal classifier
classifier$finalModel
# You can use this classifier or enter optimal hyperparameters in you classifier

# Save classifier
saveRDS(classifier, file = 'sampleClassifier.rds')

# Remove classifier
rm(classifier)

# load classifier
classifier = readRDS("irisrf.RData")

predict(classifier, input)