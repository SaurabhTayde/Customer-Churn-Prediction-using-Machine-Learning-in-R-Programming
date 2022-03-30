#Setting working directory to save Data Exploration HTML file:

setwd('G:/DSP/Project/Case Study/Telco Churn')

#Fetching the data from Watson Analytics website:
churn_raw <- read.csv("https://community.watsonanalytics.com/wp-content/uploads/2015/03/WA_Fn-UseC_-Telco-Customer-Churn.csv", header = T)

#Check the structure of the dataset:
str(churn_raw)

#Check missing values:
colSums(is.na(churn_raw))

#All of the NAs are in the 'TotalCharges' column, but we might be able to compute the total charges since we have data for monthly charges and tenure in months

churn_raw[is.na(churn_raw$TotalCharges),1:6]

#These customers all show tenure of zero months so they haven't made their first payment yet.

#Lets check if there any other zero tenure customers in the data set?
churn_raw[which(churn_raw$tenure == 0), 1:6]

dim(churn_raw[is.na(churn_raw$TotalCharges),1:6])
dim(churn_raw[which(churn_raw$tenure == 0), 1:6])

#These eleven are the only customers with zero tenure so they can safely be removed

churnnoNAs <- churn_raw[complete.cases(churn_raw),]

dim(churnnoNAs)

#Lets check co-relations between variables:

#TO check co-relations lets make other dataframe having columns with numeric values only: 

install.packages('dplyr')
library(dplyr)

numerical = select_if(churnnoNAs,is.numeric)

str(numerical)


install.packages("ggcorrplot")
library(ggcorrplot)

cor(numerical)

correlations = cor(numerical)

ggcorrplot(correlations)

#So we can see Total Charges is highly correlated with Monthly Charges:
#So we will remove Total Charges column

#Customer ID isn't useful to our analysis, so we will remove that too:

churn_neat = subset(churnnoNAs, select = -c(customerID, TotalCharges))

dim(churn_neat)

# Will convert 1,0 values into Yes, No

table(churn_neat$SeniorCitizen)

churn_neat$SeniorCitizen <- as.factor(ifelse(churn_neat$SeniorCitizen == 1, "Yes", "No"))

table(churn_neat$SeniorCitizen)

str(churn_neat)

#The variables OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV and StreamingMovies all require an internet connection
#and the variable MultipleLines needs a phone service so will replace "No internet service" and "No phone service" with "No".

factorrenames <- names(churn_neat[9:14])

data <- churn_neat %>%
  mutate_at(.vars=factorrenames,
            .funs=~recode_factor(., `No internet service`="No")) %>%
  mutate_at(.vars="MultipleLines",
            .funs=~recode_factor(., `No phone service`="No"))


str(data)

#We will perform EDA using Data Explorer Package:


install.packages('DataExplorer') 
library(DataExplorer)

plot_histogram(data)

plot_boxplot(data, by = 'MonthlyCharges')

#So no outliers present in the data

#We will use following command that gives complete information about EDA in html:

create_report(data)


#Lets check churnrate:

churnrate <- table(data$Churn) / nrow(data)

churnrate

#Over the entire data set, 26.5% of customers churned.

#We will create a a trainControl object so that all of the models use the same 10-fold cross validation on 70% of the data as a training set.
#We will then use the remaining 30% of the data to test the model accuracy.

set.seed(1)

rowindices <- sample(nrow(data))
data_shuffled <- data[rowindices,]

#First the rows are shuffled to eliminate any bias in the order of the observations in the data set

split <- round(nrow(data_shuffled) * 0.7)
split

#The data set will be split on the 4922nd row.
#Observations 1 to 4922 will make up the training set and the remaining 2110 observations will be the test set.

train <- data_shuffled[1:split,]
test <- data_shuffled[(split+1):nrow(data_shuffled),]

table(train$Churn)

table(test$Churn)


#Now we will build Logistic Regression Model:

ChurnModel = glm(Churn ~ ., family = 'binomial', data = train)

summary(ChurnModel)  

library(MASS)

#Lets apply stepAIC to find best variables in the data:

stepAIC(ChurnModel)

#According to output of stepAIC, lets build new model with new varibles:

ChurnModel_New = glm(formula = Churn ~ SeniorCitizen + Dependents + tenure + MultipleLines + 
      InternetService + OnlineSecurity + DeviceProtection + TechSupport + 
      StreamingTV + StreamingMovies + Contract + PaperlessBilling + 
      PaymentMethod + MonthlyCharges, family = "binomial", data = train)

summary(ChurnModel_New)

#Let us make predicttrain and use the predict function to make predictions.

predicttrain = predict(ChurnModel_New, type = 'response')

head(predicttrain)


#Lets find mean of survived and not survived
tapply(predicttrain, train$Churn, mean)

#So 0.18 as not churned and 0.48 as churned

#CUtover point:

table(train$Churn , predicttrain>0.5)

table(train$Churn, predicttrain>0.6)

table(train$Churn, predicttrain>0.4)

table(train$Churn, predicttrain>0.45)

table(train$Churn, predicttrain>0.55)

#So we will select 0.55 as cutoff point.

#Now we will build confusion matrix:

#We will find confusion matrix by using an inbuilt function in R

install.packages('caret')

library(caret)

class(predicttrain1)

class(train$Churn)

#We need to convert both of these into factors as it is a pre-requisite for confusionMatrix 


predicttrain1 = ifelse(predicttrain > 0.55, 1, 0)

train$Churn<- as.factor(ifelse(train$Churn == 'Yes', 1, 0))

table(predicttrain1)

predicttrain1 = as.factor(as.character(predicttrain1))

train$Churn = as.factor(as.character(train$Churn))

table(train$Churn)

confusionMatrix(data = predicttrain1, reference=train$Churn)

#So we are getting following results:

# Reference
# Prediction    0    1
# 0 3352  676
# 1  275  619
# 
# Accuracy : 0.8068          


#ROCR curve:

#ROCR:

install.packages('ROCR')
library(ROCR)

ROCPred = prediction(predicttrain, train$Churn)

ROCPerf = performance(ROCPred, 'tpr', 'fpr')

plot(ROCPerf, colorize = T, print.cutoffs.at = seq(0,1,0.1), text.adj = c(-0.2, 1.7))

#From ROCR also we can predict 0.55 is good value to pick

#Lets check test dataset now:

predicttest = predict(ChurnModel_New, type = 'response', newdata = test)

predicttest1 = ifelse(predicttest>0.55, 1, 0)

table(testData$Survived, predicttest> 0.6)


#Confusion marix for test data:
class(predicttest1)

class(test$Churn)

table(test$Churn)
table(predicttest1)

#We need to convert both of these into factors as it is a pre-requisite for confusionMatrix 

test$Churn<- as.factor(ifelse(test$Churn == 'Yes', 1, 0))

predicttest1 = as.factor(as.character(predicttest1))

test$Churn = as.factor(as.character(test$Churn))

#install.packages('e1071')

library(e1071)

confusionMatrix(data= predicttest1, reference=test$Churn)



# Confusion Matrix and Statistics
# 
# Reference
# Prediction    0    1
# 0 1409  287
# 1  127  287
# 
# Accuracy : 0.8038  

#So we are getting 80.38% accuracy on test as well