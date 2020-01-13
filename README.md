# Disaster Response Pipeline Project

## Summary

A disaster labelled multiclass disaster response dataset is used to train a model which can classify disaster messages into appropriate categories, for the relevant authorities to be notified accordingly. This project displays how to develop an end-to-end machine learning model and how to deploy it on a web application using Flask. 

The steps taken here are:
1. Extract, Transform and Load Pipeline: Clean the raw data, transform it and store it in a SQL Database
2. Preprocessing and developing relevant features for training ML model
3. Saving trained model and deploying it on web app 

The model classifies the message amongst 18 different categories e.g. earthquake, fire, military takeover etc. The web application abstracts the model and a message can be input in the search bar to classify it. The web application has visualizations describing the dataset. The model has been trained using a multilabel Random Forest classifier.

## Components

There are three folders in the repository.

1. data: It contains the original datasets and the file modified to sql databases. It also has the process_data file which process and cleans the raw data and stores it into the database
2. models: It contains the trained models which are used to classify the messages
3. app: It contains the application files required to run


### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
