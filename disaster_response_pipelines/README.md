# Disaster Response Pipeline Project

## Introduction

In This Project, a web app where an emergency worker can input a new message and get classification results in several categories is built. The Web app will also display visualizations of the data.

## Project Components

There are three components in this project.
1. ETL Pipeline

   The first is the usual data processing pipeline which is Extract, Transform, and Load process Clean up data will be loaded into a sqlite lite db to be used in building the machine learning model.

2. ML Pipeline

   Then a machine learing pipline will be implemented in train_classifier.py which,

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. *Flask Web App*

   To allow easy interactions withe the model, a flask web app was implemented. which allows user to input a new message and get classification result back via web interface. The web App will also provide visualization to the underlying data

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Project File Structure

The app folder contains html, javascript and run.py implementing Flask web app.

    - app
        | - template
        | |- master.html  # main page of web app
        | |- go.html      # classification result page of web app
        |- run.py         # Flask file that runs app

The data folder has the disaster dataset stored as csv file in categories.csv and messages.csv. python script process_data.py which performs the etl process to load data to sqlite lite InsertDatabaseName.db file.

    - data
        |- categories.csv        # data to process 
        |- messages.csv          # data to process
        |- process_data.py       # python script to perform ETL steps and load the cleaned data to sqlite lite db
        |- InsertDatabaseName.db # database to save clean data to

The models folder has the python script train_classifier.py, which use NLP and machine learning pipeline to train and build model based on DisasterResponse.db. The resulting model is stored in classifier.pkl

    - models
        |- train_classifier.py  # python script to use the data in sqlite lite db to train and build machine learning to perform the classification.
        |- classifier.pkl       # saved model 

    - README.md

