
# loop to set current wd

current_dir = getwd()
current_wd = list()

library("psych") # used for describeBy
library("MASS") # used for boxcox, not currently used here

# set print options for Tukey
options(max.print = 2000)
options(scipen=999)

if ('C:/Users/jjonus/Documents' %in% current_dir == TRUE) current_wd = paste('C:\\Users\\','jjonus','\\Google Drive\\Kaggle\\Advanced House Prices',sep = "") else current_wd = paste('C:\\Users\\','jstnj','\\Google Drive\\Kaggle\\Advanced House Prices',sep="")

df_training = read.csv(paste(current_wd,'\\train.csv',sep=""))
df_test = read.csv(paste(current_wd,'\\test.csv',sep=""))

# summary stats
str(df_training)
summary(df_training)

describe.by(df_training)

library("caret")

# Variables to investigate for removal, high variance between values. Check scores for significance.
df_var_include = nearZeroVar(df_training,freqCut = 85/15,saveMetrics = TRUE,names=TRUE)

# metrics with high variance
metrics_out = nearZeroVar(df_training,freqCut = 85/15,saveMetrics = FALSE,names=TRUE)
metrics_out
drops <- c(metrics_out)

# take data set without dropped metrics
df_training_cut = df_training[ , !(names(df_training) %in% drops)]
df_test_cut = df_test[ , !(names(df_test) %in% drops)]

#summary of variables kept
str(df_training_cut)
summary(df_training_cut)

# run pca analysis
str(df_training_cut)
pcaObject = prcomp(df_training_cut)

idCol = grep("Id",names(df_training_cut))
idCol

df_pca = df_training_cut[, - idCol]

nums =  sapply(df_pca, is.numeric)
pca = df_pca[ , nums]

pca[is.na(pca)] = 0.001

library("corrplot")

correlations = cor(pca)
corrplot(correlations,order = "hclust")

pcaObject = prcomp(pca,center=TRUE, scale=TRUE)
percentVariance = pcaObject$sdev^2/sum(pcaObject$sdev^2)*100
percentVariance[1:3]

pcaObject$rotation[,1:3]

# analyze all variables kept to see what we can use  for prediction

# fence

df_transform = as.matrix(df_training_cut)
indx = which(is.na(df_training_cut$Fence)==TRUE)
df_training_cut$Fence = as.character(df_training_cut$Fence)
df_training_cut[indx, 49] = 'NaReplaced'
df_training_cut$Fence = as.factor(df_training_cut$Fence)


plot(df_training_cut$Fence,log(df_training_cut$SalePrice))
anova_fence = aov(log(df_training_cut$SalePrice)~df_training_cut$Fence)
summary(anova_fence)
TukeyHSD(anova_fence)

neigh_fence = table(df_training_cut$Neighborhood,df_training_cut$Fence)
addmargins(neigh_fence )

library("Hmisc")

# look at price
describeBy(df_training_cut$SalePrice)
hist(df_training_cut$SalePrice)
describe(df_training_cut$SalePrice)

# price vs rooms
boxplot(log(df_training_cut$SalePrice)~df_training_cut$TotRmsAbvGrd)
anova_room = aov(df_training_cut$SalePrice ~as.factor(df_training_cut$TotRmsAbvGrd))
summary(anova_room)
TukeyHSD(anova_room)

# neighourhood count

neighbourhood_count = table(df_training_cut$Neighborhood,df_training_cut$TotRmsAbvGrd)
addmargins(neighbourhood_count)

#boxplots and anova on area,price

par(mfrow=c(2,1))
boxplot(df_training_cut$LotArea~df_training_cut$Neighborhood,cex.axis = 0.55)
boxplot(df_training_cut$SalePrice~df_training_cut$Neighborhood,cex.axis=0.55)
par(mfrow=c(1,1))

# look at all neighbourhoods vs the mean values (+- standard deviation)

boxplot(df_training_cut$LotArea~df_training_cut$Neighborhood,cex.axis = 0.75)
boxplot(df_training_cut$SalePrice~df_training_cut$Neighborhood,cex.axis=0.50)
abline(h=mean(df_training_cut$SalePrice),col="darkred")
abline(h=(mean(df_training_cut$SalePrice)+sd(df_training_cut$SalePrice)))
abline(h=(mean(df_training_cut$SalePrice)- sd(df_training_cut$SalePrice)))

#log boxplots

par(mfrow=c(2,1))
boxplot(log(df_training_cut$LotArea)~df_training_cut$Neighborhood,ylab="Area",cex.axis = 0.75)
boxplot(log(df_training_cut$SalePrice)~df_training_cut$Neighborhood,ylab="Price",cex.axis=0.75)
par(mfrow=c(1,1))

# sales price by area anova also with log

anova_area = aov(df_training_cut$SalePrice ~  df_training_cut$Neighborhood)
summary(anova_area)

options(max.print = 2000)
TukeyHSD(anova_area)

#log area

anova_area_log = aov(log(df_training_cut$LotArea) ~  df_training_cut$Neighborhood)
summary(anova_area_log)

# turn tukey analysis into dataframe

xy = TukeyHSD(anova_area_log)
df <- as.data.frame(xy$`df_training_cut$Neighborhood`)

df_area_sig = df[df$`p adj` > 0.5,]
df_area_sig

#log price

anova_price_log = aov(log(df_training_cut$SalePrice) ~  df_training_cut$Neighborhood)
summary(anova_price_log)

TukeyHSD(anova_area_log)

# combine blueste and Npkill (bluest only 2 records but based on tukey's, very similar)

library("rockchalk")

combineLevels(df_training_cut$Neighborhood,levs = c("Blueste","NPkVill"),"BlueVill")
combineLevels(df_training_cut$Neighborhood,levs = c("BrDale","MeadowV"),"MeadowDale")

# build linear models and test

# area and price model

log_area_price_model = lm(log(df_training_cut$SalePrice)~ log(df_training_cut$LotArea))
summary(log_area_price_model)

par(mfrow=c(2,3))
plot(log_area_price_model,which=c(1),main="log area price model")
plot(log_area_price_model,which=c(2))
plot(log_area_price_model,which=c(3))
plot(log_area_price_model,which=c(4))
plot(log_area_price_model,which=c(5))
plot(log_area_price_model,which=c(6))
par(mfrow=c(1,1))

hist(log_area_price_model$residuals)

#Sales and area and neighbourhood

library("car")
area_price_model = lm(log(df_training_cut$SalePrice) ~ df_training_cut$LotArea * df_training_cut$Neighborhood)
summary(area_price_model)
vif(area_price_model)

area_price_model2 = lm(log(df_training_cut$SalePrice) ~ df_training_cut$LotArea * df_training_cut$Neighborhood + df_training_cut$BedroomAbvGr)
summary(area_price_model2)
vif(area_price_model2)
anova(area_price_model,area_price_model2)

skew(area_price_model2$residuals)

# examine residuals

par(mfrow=c(2,3))
plot(area_price_model,which=c(1),main = "Log Area model with Neigh")
plot(area_price_model,which=c(2))
plot(area_price_model,which=c(3))
plot(area_price_model,which=c(4))
plot(area_price_model,which=c(5))
plot(area_price_model,which=c(6))
par(mfrow=c(1,1))

par(mfrow=c(2,1))
plot(area_price_model$residuals)
hist(area_price_model$residuals)
par(mfrow=c(1,1))

par(mfrow=c(2,3))
plot(area_price_model2,which=c(1),main="Log Area Model with Neighbourhood/Room")
plot(area_price_model2,which=c(2))
plot(area_price_model2,which=c(3))
plot(area_price_model2,which=c(4))
plot(area_price_model2,which=c(5))
plot(area_price_model2,which=c(6))
par(mfrow=c(1,1))

par(mfrow=c(2,1))
plot(area_price_model2$residuals)
hist(area_price_model2$residuals)
par(mfrow=c(1,1))

# Attempt same model but add Grliv area
Grliv_model_ml = lm(log(df_training_cut$SalePrice)~log(df_training_cut$LotArea) + log(df_training_cut$GrLivArea))
summary(Grliv_model_ml)
vif(Grliv_model_ml)

par(mfrow=c(2,3))
plot(Grliv_model_ml,which=c(1),main = "Grliv")
plot(Grliv_model_ml,which=c(2))
plot(Grliv_model_ml,which=c(3))
plot(Grliv_model_ml,which=c(4))
plot(Grliv_model_ml,which=c(5))
plot(Grliv_model_ml,which=c(6))
par(mfrow=c(1,1))

plot(Grliv_model_ml$residuals)
hist(Grliv_model_ml$residuals)

skew(Grliv_model_ml$residuals)

sresid <- studres(Grliv_model_ml) 
hist(sresid, freq=FALSE, 
     main="Distribution of Studentized Residuals")
xfit<-seq(min(sresid),max(sresid),length=40) 
yfit<-dnorm(xfit) 
lines(xfit, yfit)

boxplot(Grliv_model_ml$residuals ~ df_training_cut$Neighborhood,cex.axis = 0.5)

anova(Grliv_model_ml,log_area_price_model) # test against original model

# model introducing the neighbourhoods

# neighbourhood count
neighbourhood_count = table(df_training_cut$Neighborhood,df_training_cut$TotRmsAbvGrd)
addmargins(neighbourhood_count)

df_training_cut = df_training_cut[-1299,]
area_model_streets = lm(log(df_training_cut$SalePrice)~log(df_training_cut$LotArea) + log(df_training_cut$GrLivArea) + df_training_cut$Neighborhood)
summary(area_model_streets)
hist(area_model_streets$residuals)
vif(area_model_streets)

par(mfrow=c(2,3))
plot(area_model_streets,which=c(1),main="GrLiv + Area +  Neighbourhood")
plot(area_model_streets,which=c(2))
plot(area_model_streets,which=c(3))
plot(area_model_streets,which=c(4))
plot(area_model_streets,which=c(5))
plot(area_model_streets,which=c(6))
par(mfrow=c(1,1))

boxplot(area_model_streets$residuals)
# residual analysis

#plot residuals against normal curve
sresid <- studres(area_model_streets) 
hist(sresid, freq=FALSE, 
     main="Distribution of Studentized Residuals")
xfit<-seq(min(sresid),max(sresid),length=40) 
yfit<-dnorm(xfit) 
lines(xfit, yfit)


boxplot(area_model_streets$residuals ~ df_training_cut$Neighborhood,cex.axis = 0.5)

skew(area_model_streets$residuals)

anova(Grliv_model_ml,area_model_streets) #compare against Grliv model 

# compare skewness of models

skew(area_price_model2$residuals)
skew(log_area_price_model$residuals)

# try adding more important variables

area_model_streets2 = lm(log(df_training_cut$SalePrice)~log(df_training_cut$LotArea) + log(df_training_cut$GrLivArea) + df_training_cut$Neighborhood + df_training_cut$TotRmsAbvGrd + df_training_cut$GarageCars + df_training_cut$HouseStyle + df_training_cut$OverallCond + df_training_cut$OverallQual)
summary(area_model_streets2)
vif(area_model_streets2)

par(mfrow=c(2,3))
plot(area_model_streets2,which=c(1),main="GrLiv + Neighbourhood + rooms")
plot(area_model_streets2,which=c(2))
plot(area_model_streets2,which=c(3))
plot(area_model_streets2,which=c(4))
plot(area_model_streets2,which=c(5))
plot(area_model_streets2,which=c(6))
par(mfrow=c(1,1))

anova(area_model_streets,area_model_streets2)
skew(area_model_streets2$residuals)

sresid_streets <- studres(area_model_streets2) 
hist(sresid_streets, freq=FALSE, 
     main="Distribution of Studentized Residuals")
xfit<-seq(min(sresid_streets),max(sresid_streets),length=40) 
yfit<-dnorm(xfit) 
lines(xfit, yfit)

library("car")

# check independence of residuals
durbinWatsonTest(log_area_price_model)
durbinWatsonTest(area_price_model)
durbinWatsonTest(area_model_streets)
durbinWatsonTest(area_model_streets2)

#testing lasso ridge and elastic net models

library("glmnet")

# explore correlations to show power of these methods

cor(df_training_cut[sapply(df_training_cut, is.numeric)])

area_model_streets2 = lm(log(SalePrice)~log(LotArea)+ Functional + Exterior1st + KitchenQual  + log(GrLivArea)  + Neighborhood + TotRmsAbvGrd + GarageCars + HouseStyle + OverallCond + OverallQual,data = df_training_cut)
summary(area_model_streets2)
vif(area_model_streets2)

# create x and y values for lasso 
xfactors = model.matrix(~log(LotArea) + Functional + Exterior1st + KitchenQual + log(GrLivArea) + Neighborhood + TotRmsAbvGrd + GarageCars + HouseStyle  + OverallCond + OverallQual,df_training)
x = as.matrix(data.frame(xfactors))
y=log(df_training_cut$SalePrice)

# run ridge,lasso and elastic net

glmod1 = glmnet(x,y=y,alpha=0) # ridge
glmod2 = glmnet(x,y=y,alpha=1) # lasso
glmod3 = glmnet(x,y=y,alpha=0.2) #elastic net

# plot all 3 side by side

par(mfrow=c(1,3))

plot(glmod1)
plot(glmod2)
plot(glmod3)

# show coefficients
print(glmod1)
print(glmod2)
print(glmod3)

# ridge
cvfitR = cv.glmnet(x, y=log(df_training_cut$SalePrice),alpha=0)
plot(cvfitR)

cvfitR$lambda.min
cvfitR$lambda.1se

coef(cvfitR, s = "lambda.min")
coef(cvfitR, s = "lambda.1se")

predict_ridge = predict(cvfitR,newx = xfactors,s = "lambda.1se")

# lasso
cvfitL = cv.glmnet(x, y=log(df_training_cut$SalePrice),alpha=1)
plot(cvfitL)

cvfitL$lambda.min
cvfitL$lambda.1se

coef(cvfitL, s = "lambda.min")
coef(cvfitL, s = "lambda.1se")

predict_lasso = predict(cvfitL,newx = xfactors,s = "lambda.1se")

# elastic net
cvfitEL = cv.glmnet(x, y=log(df_training_cut$SalePrice),alpha=0.2)
plot(cvfitEL)

cvfitEL$lambda.min
cvfitEL$lambda.1se

coef(cvfitEL, s = "lambda.min")
coef(cvfitEL, s = "lambda.1se")

predict_elnet = predict(cvfitEL,newx = xfactors,s = "lambda.1se")

# compare RMSE of models

RMSE_area = sqrt(mean((df_training2$`log_area_price_model$fitted.values` - df_training2$Logprice)^2))
RMSE_streets = sqrt(mean((df_training2$`area_model_streets$fitted.values` - df_training2$Logprice)^2))
RMSE_streets_improved = sqrt(mean((df_training2$`area_model_streets2$fitted.values` - df_training2$Logprice)^2))
RMSE_lasso = sqrt(mean((df_training2$lasso - df_training2$Logprice)^2))
RMSE_ridge = sqrt(mean((df_training2$ridge - df_training2$Logprice)^2))
RMSE_elnet = sqrt(mean((df_training2$elnet - df_training2$Logprice)^2))

# MAE

RMAE_area = sum(abs(df_training2$`log_area_price_model$fitted.values` - df_training2$Logprice)) / length(df_training2$SalePrice) 
RMAE_streets =  sum(abs(df_training2$`area_model_streets$fitted.values` - df_training2$Logprice)) / length(df_training2$SalePrice) 
RMAE_streets_improved =  sum(abs(df_training2$`area_model_streets2$fitted.values` - df_training2$Logprice)) / length(df_training2$SalePrice) 
RMAE_lasso =  sum(abs(df_training2$lasso - df_training2$Logprice)) / length(df_training2$SalePrice) 
RMAE_ridge =  sum(abs(df_training2$ridge - df_training2$Logprice)) / length(df_training2$SalePrice) 
RMAE_elnet = sum(abs(df_training2$elnet - df_training2$Logprice)) / length(df_training2$SalePrice) 

# run predictions on out of sample data

# fill in df_test blank values
library("mice")
library("VIM")
library("missForest")

# look at na summary
md.pattern(df_test_cut)

# drop columns with high NA values
drop_nas = c("PoolQC","Alley","Fence","FireplaceQu","MiscFeature","Id","LotFrontage")

df_test_cut = df_test[ , !(names(df_test) %in% drop_nas)]

df_dropped = df_test[ , (names(df_test) %in% drop_nas)]

# look at na summary again
df_imp = missForest(df_test_cut,verbose = TRUE,mtry=10)
imputed_frame = data.frame(df_imp$ximp)

xfactors_test = model.matrix(~log(LotArea) + Functional  + Exterior1st + KitchenQual  + log(GrLivArea) + Neighborhood + TotRmsAbvGrd + GarageCars + HouseStyle  + OverallCond + OverallQual,data = imputed_frame)
xfactors = model.matrix(~log(LotArea) + Functional +  Exterior1st + KitchenQual + log(GrLivArea) + Neighborhood + TotRmsAbvGrd + GarageCars + HouseStyle  + OverallCond + OverallQual,df_training)

colnames(xfactors)
colnames(xfactors_test)

# create sparsed data matrix for predictions

sparsed_test_data <- Matrix(data=0,
                            nrow=nrow(xfactors_test),
                            ncol=ncol(xfactors),
                            dimnames=list(rownames(xfactors_test),
                                          colnames(xfactors)))

# fill in dataframe with matrix values

for(i in colnames(xfactors_test)){
  sparsed_test_data[, i] <- xfactors_test[, i]
}


# do same for linear model

area_model_streets2 = lm(log(SalePrice)~log(LotArea) + Exterior1st + KitchenQual  + log(GrLivArea)  + Neighborhood + TotRmsAbvGrd + GarageCars + HouseStyle + OverallCond + OverallQual,data = df_training_cut)
summary(area_model_streets2)

# build matrices

xfactors_test = model.matrix(~log(LotArea)  + Exterior1st + KitchenQual  + log(GrLivArea) + Neighborhood + TotRmsAbvGrd + GarageCars + HouseStyle  + OverallCond + OverallQual,data = imputed_frame)
xfactors = model.matrix(~log(LotArea) +  Exterior1st + KitchenQual + log(GrLivArea) + Neighborhood + TotRmsAbvGrd + GarageCars + HouseStyle  + OverallCond + OverallQual,df_training_cut)

colnames(xfactors)
colnames(xfactors_test)

# create sparsed data matrix for predictions

sparsed_test_data_lm <- Matrix(data=0,
                            nrow=nrow(xfactors_test),
                            ncol=ncol(xfactors),
                            dimnames=list(rownames(xfactors_test),
                                          colnames(xfactors)))

# fill in dataframe with matrix values

for(i in colnames(xfactors_test)){
  sparsed_test_data_lm[, i] <- xfactors_test[, i]
}

sparsed_test_data_lm = matrix(sparsed_test_data_lm)

colnames(sparsed_test_data_lm)
predict_linear = predict(area_model_streets2,newx = sparsed_test_data_lm)

# predict ridge, lasso and elastic net

predict_ridgeT = predict(cvfitR,newx = sparsed_test_data,s = "lambda.1se")
predict_lassoT = predict(cvfitL,newx = sparsed_test_data,s = "lambda.1se")
predict_elnetT = predict(cvfitEL,newx = sparsed_test_data,s = "lambda.1se")

# put in df format to export to excel, also create index column

df_linear = data.frame(predict_linear)
df_linear = cbind(df_linear,df_test$Id)
colnames(df_linear)[1] = "SalePrice"
colnames(df_linear)[2] = "Id"


df_ridge = data.frame(predict_ridgeT)
df_ridge = cbind(df_ridge,df_test$Id)
colnames(df_ridge)[1] = "SalePrice"
colnames(df_ridge)[2] = "Id"
df_ridge$SalePrice = exp(df_ridge$SalePrice)
write.csv(df_ridge,"ridge_predictions.csv",row.names = FALSE)

df_lasso = data.frame(predict_lassoT)
df_lasso= cbind(df_lasso,df_test$Id)
colnames(df_lasso)[1] = "SalePrice"
colnames(df_lasso)[2] = "Id"
df_lasso$SalePrice = exp(df_lasso$SalePrice)
write.csv(df_lasso,"lasso_predictions.csv",row.names = FALSE)

df_elnet = data.frame(predict_elnetT)
df_elnet = cbind(df_elnet,df_test$Id)
colnames(df_elnet)[1] <- "SalePrice"
colnames(df_elnet)[2] = "Id"
df_elnet$SalePrice = exp(df_elnet$SalePrice)
write.csv(df_elnet,"elnet_predictions.csv",row.names=FALSE)

#### attempt to do same predictions with tuning ####

library("caret")

# define control
fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10)

set.seed(123)

glmnet_fit <- train(log(SalePrice)~log(LotArea)+ Functional + Exterior1st + KitchenQual  + log(GrLivArea)  + Neighborhood + TotRmsAbvGrd + GarageCars + HouseStyle + OverallCond + OverallQual, data = df_training, 
                 method = "glmnet", 
                 trControl = fitControl)
gbmFit1

??train
