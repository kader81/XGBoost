# Import libraries
library(caTools)
library(xgboost)
library(caret)
# set your working directory
setwd("C:/competition")

# read the files
train=read.csv("train.csv")

# replace na values with 0
train[is.na(train)]<-0

#Modify the target variable to have classes starting from zero as XGBoost expects the levels to start from zero
train$TripType=as.numeric(as.factor(train$TripType)-1

#Set seed values and split the training data - Cannot handle all data due to memory limitation
set.seed(101)
sample = sample.split(train$TripType,SplitRatio=.40)
train.part1 = subset(train, sample == TRUE)
nrow(train.part1)

#Remove target variable from train and create labels for XGboost, create matrix for XGboost data
y=train.part1$TripType
train.part1$TripType=NULL

#Convert factor variables to dummy variables
train.part1 = cbind(train.part1[-grep("DepartmentDescription",colnames(train.part1))],model.matrix(~ 0 + DepartmentDescription, train.part1))
train.part1 = cbind(train.part1[-grep("Weekday",colnames(train.part1))],model.matrix(~ 0 + Weekday, train.part1))

X=as.matrix(train.part1)

# XGBoost params - created as list of parameters for multiple iteration
params <- data.frame(shrinkage=c(0.04, 0.03, 0.03, 0.03, 0.02),
                      rounds = c(140, 160, 170, 140, 180),
                      depth = c(8, 7, 9, 10, 10),
                      gamma = c(0, 0, 0, 0, 0),
                      min.child = c(10, 11, 7,10 , 12),
                      colsample.bytree = c(0.7, 0.6, 0.65, 0.6, 0.85),
                      subsample = c(1, 0.9, 0.95, 1, 0.6))

# As of now - running XGBoost for one time. Will change this code to use multiple iterations based on params
xgboost.model <- xgboost(data = X, label = y.part1, max.depth = params$depth[1], eta = params$shrinkage[1],
                           nround = params$rounds[1], nthread = 4, objective = "multi:softprob", subsample=params$subsample[1],
                           colsample_bytree=params$colsample.bytree[1], gamma=params$gamma[1], min.child.weight=params$min.child[1],num_class=38)

#### Verification - Check accuracy - Take 1000 rows from training set and check accuracy
train.check = train[1:1000,]
y.check = train.check$TripType
train.check$TripType=NULL
train.check = cbind(train.check[-grep("DepartmentDescription",colnames(train.check))],model.matrix(~ 0 + DepartmentDescription, train.check))
train.check = cbind(train.check[-grep("Weekday",colnames(train.check))],model.matrix(~ 0 + Weekday, train.check))
X.train.check = as.matrix(train.check)
pred.result = predict(xgboost.model, X.train.check)
### Result will be in no of rows * number of classes in target variable. So need to create a matrix and transpose. Here the number of classes are 38
pred.result.mat = t(matrix(pred.result, nrow=38, ncol=length(pred.result)/38))
# Take the maximum class as final result
pred.result.mat = max.col(pred.result.mat,"last")
# Create table or confusion matrix - check the accuracy
confusionMatrix(y.check+1,pred.result.mat)
