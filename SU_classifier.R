## Testing Algorithms

## Importance plot function
plot_feature_importance <- function(imp) {
  featureImportance <- data.frame(Feature=row.names(imp), Importance=imp[,1])
  # featureImportance <- featureImportance[rev(order(featureImportance$Importance)),]
  featureImportance$Feature <- factor(featureImportance$Feature, levels = featureImportance$Feature[rev(order(featureImportance$Importance))])
  
  p <- ggplot(featureImportance, aes(x=Feature,y=Importance)) +
    # p <- ggplot(featureImportance, aes(x=rev(reorder(Feature, Importance)), y=Importance)) +
    geom_bar(stat="identity", fill=col_viewovertown[4]) +
    # coord_flip() + 
    theme_light(base_size=20) +
    xlab("") +
    ylab("Importance") + 
    ggtitle(paste("Random Forest Feature Importance",sep=" ")) +
    theme(plot.title=element_text(size=18),
          axis.text.x = element_text (size = 6,color="#888888")) +
    coord_flip()
  # print(p)
}




## Preparing data
# Path for csv --CHANGE--
# USB_Fer "E:/Machine Learning BnD/Possible QA Article/Lara's work/170331_demo_zygo_aseg_output_QCcomplete_with_volume_and_thickness_FT.csv"

data2 = data.frame(read.table("E:/Machine Learning BnD/Possible QA Article/Lara's work/170331_demo_zygo_aseg_output_QCcomplete_with_volume_and_thickness_FT.csv",sep=",",header=TRUE))

# Renaming some variables
names(data2)[1] <- "id"
names(data2)[3] <- "Age"
names(data2)[4] <- "Gender"

# y is the outcome variable, in this case $Score
df=data.frame(y=data2$Score,age=data2$Age,data2[,c(7:39,46:72,83:90)])#,EstimatedTotalIntraCranialVol=data1$EstimatedTotalIntraCranialVol)
df = df[complete.cases(df),]

# y is restructured from 4 to 2 factors
df$y = as.factor(ifelse(df$y==4,"exclusion","inclusion"))

# maak twee sampels met een trainins set (train) en een testset (test)
set.seed(1) # deze kan je aan en uit setten om wel een random start punt te nemen
samp = sample(nrow(data2), 0.6 * nrow(data2))
df$y = as.factor(df$y) # Not sure what this is doing
train = df[samp, ]
test = df[-samp, ]

# alternatively
library(caTools)
set.seed(1)
split = sample.split(df$y, SplitRatio = 0.6)
training_set = subset(df, split == TRUE)
test_set = subset(df, split == FALSE)

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
# Possible candidates are Random Forest, other tree-based models or Support Vector Machines


# rf needs packages randomForest and e1071, it will load what it needs and asks to install is missing (though you might need to run it again) 
#install.packages('e1071')
#library(e1071)
#install.packages('randomForest')
#library(randomForest)


classifier = train(form = y ~ .,
                   data = training_set,
                   trControl = tr,
                   method = 'rf')
classifier # tunes the model and gives you the output
classifier$results
classifier$bestTune # optimal classifier
classifier$finalModel
# You can use this classifier or enter optimal hyperparameters in your classifier
print(classifier$finalModel)

#Importance
importance(classifier$finalModel)
imp <- importance(classifier$finalModel)
select <- as.data.frame(imp)
names <- rev(order(select$MeanDecreaseGini))[1:10] # order en select names of top 8 performers
select_order <- row.names(select)[names]
imp_pick = select[select_order,, drop=FALSE] # select variables as data.frame (drop = FALSE ensure it retains type)
p_imp = plot_feature_importance(imp_pick) # create plot
print(p_imp) # show plot

# Save classifier
saveRDS(classifier, file = 'sampleClassifier.rds')

# Remove classifier
rm(classifier)

# load classifier
classifier = readRDS("irisrf.RData")

predict(classifier, input)

#
col_viewovertown <- c("#FF5335","#B29C85","#306E73","#3B424C","#1D181F") # https://color.adobe.com/View-over-the-town-color-theme-1637621/
font_size = 3



## GXBoost
classifier2 = train(form = y ~ .,
                   data = training_set,
                   trControl = tr,
                   method = 'xgbTree')
classifier2 # tunes the model and gives you the output
classifier2$results
classifier2$bestTune # optimal classifier
classifier2$finalModel
# You can use this classifier or enter optimal hyperparameters in your classifier
print(classifier2$finalModel)


### Oblique Random Forest (ORFlog)
#install.packages('obliqueRF')
#library(obliqueRF)
classifier3 = train(form = y ~ .,
                    data = training_set,
                    trControl = tr,
                    method = 'ORFlog')
classifier3 # tunes the model and gives you the output
classifier3$results
classifier3$bestTune # optimal classifier
classifier3$finalModel
# You can use this classifier or enter optimal hyperparameters in your classifier
print(classifier3$finalModel)


#Importance does not work yet..
importance(classifier3$finalModel)
imp <- importance(classifier3$finalModel)
select <- as.data.frame(imp)
names <- rev(order(select$MeanDecreaseGini))[1:10] # order en select names of top 8 performers
select_order <- row.names(select)[names]
imp_pick2 = select[select_order,, drop=FALSE] # select variables as data.frame (drop = FALSE ensure it retains type)
p_imp = plot_feature_importance(imp_pick2) # create plot
print(p_imp) # show plot


## svm (linear,poly,radial) svmLinear svmPoly svmRadial
#install.packages('kernlab')
classifier.svm = train(form = y ~ .,
                    data = training_set,
                    trControl = tr,
                    method = 'svmLinear')
classifier.svm # tunes the model and gives you the output
classifier.svm$results
classifier.svm$bestTune # optimal classifier
classifier.svm$finalModel
# You can use this classifier or enter optimal hyperparameters in your classifier
print(classifier.svm$finalModel)

# svmPoly

classifier.svmP = train(form = y ~ .,
                       data = training_set,
                       trControl = tr,
                       method = 'svmPoly')
classifier.svmP # tunes the model and gives you the output
classifier.svmP$results
classifier.svmP$bestTune # optimal classifier
classifier.svmP$finalModel
# You can use this classifier or enter optimal hyperparameters in your classifier
print(classifier.svmP$finalModel)

# svmRadial
classifier.svmR = train(form = y ~ .,
                        data = training_set,
                        trControl = tr,
                        method = 'svmRadial')
classifier.svmR # tunes the model and gives you the output
classifier.svmR$results
classifier.svmR$bestTune # optimal classifier
classifier.svmR$finalModel
# You can use this classifier or enter optimal hyperparameters in your classifier
print(classifier.svmR$finalModel)



### Overall Comparison

rf_pred = predict(classifier, test_set)
xgb_pred = predict(classifier2, test_set)
orf_pred = predict(classifier3, test_set)
svmL_pred = predict(classifier.svm, test_set)
svmP_pred = predict(classifier.svmP, test_set)
svmR_pred = predict(classifier.svmR, test_set)

rf_pred
xgb_pred
orf_pred
svmL_pred
svmP_pred
svmR_pred

cm = table(test_set$y, rf_pred)  # rf
cm
cm2= table(test_set$y,xgb_pred)  # xgboost
cm2
cm3= table(test_set$y,orf_pred)  # orf
cm3
cm4= table(test_set$y,svmL_pred) # svmLinear
cm4
cm5= table(test_set$y,svmP_pred) # svmPolynomial
cm5
cm6= table(test_set$y,svmR_pred) # svmRadial
cm6