#load packages
library(plyr); library(dplyr);
library(ggplot2); library(grid); library(gridExtra); 
library(glmnet); library(e1071); library(knitr)
library(gbm); library(corrplot); library(caret)
library(mlbench); library(gdata)

#load data
data = read.csv("train.csv")
data$id = NULL
data.submit = read.csv("test.csv")
id = data.submit$id

#Check for NA values
dim(na.omit(data)) == dim(data)

#Check distribution of response
density1 <- ggplot(data,aes(loss))+
  geom_density(fill = "palegreen2", alpha = 0.4) 

density2 <- ggplot(data,aes(log(loss)))+
  geom_density(fill = "skyblue", alpha = 0.4)

grid.arrange(density1,density2,nrow=2)

skewness(data$loss) #3.794898
skewness(log(data$loss)) #0.09297306
data$loss = log(data$loss)

#plots of continuous variables
p1 = ggplot(data=data, aes(x=cont2,y=loss))+
  geom_point(size=0.4)+ylab("log(loss)")

p2 = ggplot(data=data, aes(x=cont3,y=loss))+
  geom_point(size=0.4)+ylab("log(loss)")

p3 = ggplot(data=data, aes(x=cont13,y=loss))+
  geom_point(size=0.4)+ylab("log(loss)")

p4 = ggplot(data=data, aes(x=cont14,y=loss))+
  geom_point(size=0.4)+ylab("log(loss)")

grid.arrange(p1,p2,p3,p4,nrow = 2)
grid.arrange(p1,p2,p3,p4,nrow = 2)

#check correlation
data.cont = data[,117:131]
cormat = cor(data.cont[,-15])
corrplot(cormat, method = "circle")

#drop highly correlated variables
data$cont9 = NULL
data$cont12 = NULL

data.try = data

#Remove those categories with observations under 10
for(i in 1:116){
  var = eval(parse(text = paste("data.try$cat",as.character(i),sep="")))
  data.try = data.try[var %in% names(which(table(var) > 10)), ]
}
data.try = droplevels(data.try)

#Remove variables with NA coefficients in Lin.Reg
cat74 = data.try$cat74; data.try$cat74 = NULL
cat81 = data.try$cat81; data.try$cat81 = NULL
cat85 = data.try$cat85; data.try$cat85 = NULL
cat87 = data.try$cat87; data.try$cat87 = NULL
cat89 = data.try$cat89; data.try$cat89 = NULL
cat90 = data.try$cat90; data.try$cat90 = NULL
cat91 = data.try$cat91; data.try$cat91 = NULL
cat92 = data.try$cat92; data.try$cat92 = NULL
cat98 = data.try$cat98; data.try$cat98 = NULL
cat99 = data.try$cat99; data.try$cat99 = NULL
cat100 = data.try$cat100; data.try$cat100 = NULL 
cat101 = data.try$cat101; data.try$cat101 = NULL 
cat102 = data.try$cat102; data.try$cat102 = NULL
cat103 = data.try$cat103; data.try$cat103 = NULL 
cat104 = data.try$cat104; data.try$cat104 = NULL 
cat106 = data.try$cat106; data.try$cat106 = NULL
cat107 = data.try$cat107; data.try$cat107 = NULL
cat108 = data.try$cat108; data.try$cat108 = NULL
cat111 = data.try$cat111; data.try$cat111 = NULL
cat113 = data.try$cat113; data.try$cat113 = NULL
cat114 = data.try$cat114; data.try$cat114 = NULL
cat116 = data.try$cat116; data.try$cat116 = NULL


#Split to train and test.
set.seed(5)
n = dim(data.try)[1]
train = sample(n,floor(n*0.7))
data.train = data.try[train,]
data.test = data.try[-train,]

#Fit a linear model and predict.
lm.try = lm(loss~.,data.train)
pred.try = predict(lm.try,data.test)
MAE.try = mean(abs(exp(data.test$loss) - exp(pred.try)))
MAE.try 


#Lasso
set.seed(1000)
ydata.train = data.matrix(data.train[,"loss"])
Xdata.train = data.matrix(data.train[,!(colnames(data.train) %in% c("loss"))])
Xdata.test = data.matrix(data.test[,!(colnames(data.train) %in% c("loss"))])

#Cross-validation
grid = 10^seq(4, -20, length=1000)
fit.lasso = cv.glmnet(Xdata.train,
                      ydata.train,
                      alpha=1,
                      lambda=grid,
                      thresh=1e-12)

#Get the best lambda and predict
best.lambda=fit.lasso$lambda.min
pred.lasso = predict(fit.lasso,Xdata.test,s=best.lambda)
MAELasso = mean(abs(exp(data.test$loss) - exp(pred.lasso)))
MAELasso

#Boosting

#Create a validation set for grid search
n3 = dim(data.train)[1]
val = sample(n3,floor(0.2*n3)) 
data.val = data.train[val,]
data.train = data.train[-val,]

#Grid Search.
set.seed(1000)
lambd <-  seq(0.001, 0.01, by=0.003)
lambd <- c(lambd,seq(0.04,0.2,by=0.08))
ntree <- seq(200,800,by=200)
ntree <- c(ntree,seq(1000,3000,by=250))
depth <- c(1,2,4,6,8)
m <- length(lambd)
l <- length(ntree)
t <- length(depth)
testErr <- array(dim=c(m,l,t))
for (i in 1:m){
  for(d in 1:t){
    boostCol = gbm(loss ~., data = data.train,
                   distribution = "gaussian",
                   n.trees = 3000,
                   shrinkage = lambd[i],
                   interaction.depth = depth[d])
    for(k in 1:l){
      testPred = predict(boostCol,
                         data.val,
                         n.trees = ntree[k])
      testErr[i,k,d] = mean(abs(exp(data.val$loss) - exp(testPred)))
    }
    print(i)
  }
}

which(testErr == min(testErr),arr.ind = T)
bestlam = lambd[4]
bestntree = ntree[13]
bestdepth = depth[5]

#Create plots of hyperparameters.
MAEplotNtree = testErr[4,,5]
MAEplotInt = testErr[4,13,]
boostPlot =ggplot(data.frame(x=ntree,y=MAEplotNtree), aes(x=x, y=y)) +
  xlab("number of trees") + 
  ylab("Validation MAE") + 
  geom_point()+
  geom_line()
boostPlot

boostPlot2 =ggplot(data.frame(x=depth,y=MAEplotInt), aes(x=x, y=y)) +
  xlab("Interaction Depth") + 
  ylab("Validation MAE") + 
  geom_point()+
  geom_line()
boostPlot2

grid.arrange(boostPlot,
             boostPlot2,ncol=2)

#Predict for Boosting
set.seed(1000)
boost.fit <- gbm(loss ~ ., data = data.train,
                 distribution = "gaussian",
                 n.trees = bestntree,
                 shrinkage =bestlam,
                 interaction.depth = bestdepth)
boost.pred <- predict(boost.fit,
                      data.test,
                      n.trees = bestntree)
MAEBoost = mean(abs(exp(data.test$loss) - exp(boost.pred)))
MAEBoost 

#Get 10 most important variables from Boosting.
head(summary(boost.fit),10)

#Create a data set from non-zero coefs from Lasso
coefs_temp <- predict(fit.lasso,
                      s = fit.lasso$lambda.1se,
                      type = "coefficients")
coefs_temp2 <- data.frame(name = coefs_temp@Dimnames[[1]][coefs_temp@i + 1],
                          coefficient = coefs_temp@x)
names <- levels(coefs_temp2[,1])
names <- names[2:length(names)]
names <- c(names,"loss")
data.lasso = data.try[,names]
data.lasso.train = data.lasso[train,]
data.lasso.test = data.lasso[-train,]


# Boosting for smaller model.

#Create a validation set.
set.seed(1000)
n3 = dim(data.lasso.train)[1]
val = sample(n3,floor(0.2*n3)) 
data.lasso.val = data.lasso.train[val,]
data.lasso.train = data.lasso.train[-val,]


#Grid search to find hyperparameters.
set.seed(1000)
lambd <-  seq(0.001, 0.01, by=0.003)
lambd <- c(lambd,seq(0.04,0.2,by=0.08))
ntree <- seq(200,800,by=200)
ntree <- c(ntree,seq(1000,3000,by=250))
depth <- c(1,2,4,6,8)
m <- length(lambd)
l <- length(ntree)
t <- length(depth)
testErr2 <- array(dim=c(m,l,t))
for (i in 1:m){
  for(d in 1:t){
    boostCol = gbm(loss ~., data = data.lasso.train,
                   distribution = "gaussian",
                   n.trees = 3000,
                   shrinkage = lambd[i],
                   interaction.depth = depth[d])
    for(k in 1:l){
      testPred = predict(boostCol,
                         data.lasso.val,
                         n.trees = ntree[k])
      testErr2[i,k,d] = mean(abs(exp(data.lasso.val$loss) - exp(testPred)))
    }
    print(i)
  }
}
which(testErr2 == min(testErr2),arr.ind = T)
#Same results as before.


set.seed(1000)
boost.fit.lasso <- gbm(loss ~ ., data = data.lasso,
                       distribution = "gaussian",
                       n.trees = bestntree,
                       shrinkage =bestlam,
                       interaction.depth = bestdepth)
boost.pred.small <- predict(boost.fit.lasso,
                            data.submit,
                            n.trees = bestntree)
MAEBoost.small = mean(abs(exp(data.lasso.test$loss) - exp(boost.pred.small)))
MAEBoost.small


#Submit to Kaggle

#Use all the training data set for full model
boost.fit <- gbm(loss ~ ., data = data.train,
                 distribution = "gaussian",
                 n.trees = bestntree,
                 shrinkage =bestlam,
                 interaction.depth = bestdepth)

pred.submit = predict(boost.fit,
                      data.submit,
                      n.trees=bestntree)
Submit = data.frame(id,loss=exp(pred.submit))
write.csv(Submit,file="Submit2.csv",row.names=FALSE)


#Smaller model
pred.submit2 = predict(boost.fit.lasso,
                       data.submit,
                       n.trees=bestntree)
Submit = data.frame(id,loss=exp(pred.submit2))
write.csv(Submit,file="Submit3.csv",row.names=FALSE)

