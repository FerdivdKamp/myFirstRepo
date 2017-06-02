#' QAtool
#' 
#' Applies a prebuilt model to Freesurfer output for QA
#' returns classification
#' @param input Can be .r .txt or .csv
#' output Is gonna be quite something
#' model 'general' or 'personalised' Default is 'general'
#' @examples QAtool('FreesurferData.csv', 'QAdone.csv', model='general') )
#' QAtool()

QAtool = function(input, output, model='general'){
  data(input) # load data (works for .r, .txt, .csv)
  if model=='general'
    classifier = readRDS('classifier.rds') # load model
    y_pred = predict(classifier, input) # apply model to input
    # output to be entered
   
}
