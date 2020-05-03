# Spark Project: Sparkify Churn Prediction

### Introduction

Customer churn or attrition is when a customer stop being a paying client, being able to predict if a customer is going to becoming a non paying client is very important to the business.  In this project, we will build a machine learning pipline to predict customer churn using data collected from a fictional music streaming platform created by udacity.

### DataSource

The dataset is from sparkify, a fictional music streaming platform created by udacity. There are 3 data set available, mini, medium and large. In this exercise, the medium data set is being used.

### Data Exploration

The schema of the data provided is as follows

| feature | data type |
|---------|-----------|
| artist |string |
| auth | string |
| firstName | string |
| gender | string |
| itemInSession | long |
| lastName | string |
| length | double |
| level | string |
| location | string |
| method | string |
| page | string |
| registration | long |
| sessionId | long |
| song | string |
| status | long |
| ts | long |
| userAgent | string |
| userId | string |


### PreProcessing and Feature extraction

1. there are missing data in the dataset. those records with missing userid, sessionid are removed as the first step

2. create a churn column based on the criteria, when page == 'Cancellation Confirmation'

3. convert columns like level or gender to numeric columns.

4. create features from page visit, e.g. submit upgrade, submit downgrade, thumbs up, thumbs down

5. create features that measure how long the user has been with sparkify since registration

6. create features that summarize how many songs in their profile, number of artist they listen to etc.


### build model and determine metrics to use

The data is split into training and test dataset in the ratio of 80:20. When training the model, we perform grid search and cross validation to tune the hyperparameters for each model.

after loading the data to spark dataframe, then various models have been applied to evaluate the performance. This includes Linear Regression, Random Forest and Gradient Boosting. Since our data is imbalance, i.e. only 20% of the sample is churned, so the metrics chosen is important.

In Fact if we implement a dummy model that always predict the customer is not going to be churned, then it will achieve about 80% accuracy. So using accuracy will not tell the whole story. In fact in this case F1 score is a more appropriate metrics as compared to accuracy, Area Under ROC.


### Conclusion

Based on F1 score, Random Forest seems to be the best model amongst the models in predicting which customer is likely to churn.
