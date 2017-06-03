## Interesting models available in caret

# Tree-based
# Random Forest 'rf'

# Conditional Inference Random Forest:	                    cforest
# Oblique Random Forest	                                    ORFlog,ORFpls,ORFridge,ORFsvm

##regular (orthogonal) random forests use treshold on individual features
##ORF uses either random or guided hyperplanes to seperate feature space
##guidance occurs through multivariate models.....


# Random Forest by Randomization	                          extraTrees
# Forest Rule-Based Model	                                  rfRules
# Random Forest with Additional Feature Selection	          Boruta
# Regularized Random Forest	                                RRF, RFFglobal
# Rotation Forest	                                          rotationForest, rotationForestCp
# Weighted Subspace Random Forest	                          wsrf


# eXtreme Gradient Boosting	                                xgbTree


## Support Vector Machine Classifiers
# Support Vector Machines with Linear Kernel	              svmLinear
# Support Vector Machines with Polynomial Kernel	          svmPoly
# Support Vector Machines with Radial Basis Function Kernel	svmRadial
