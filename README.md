## DSND-Blog-Post

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The code should requires the following libraries to run:

- nltk 
- numpy 
- pandas 
- scikit-learn 
- sqlalchemy 
- plotly

## Project Motivation<a name="motivation"></a>

In this project, we apply data engineering skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

The project has data set containing real messages that were sent during disaster events. We create a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

The project also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data. 

Below are a few screenshots of the web app.


![Image of DRP_Image1](https://github.com/ms1729/DR/blob/main/DRProject_Image1.jpg)


For this project, I was interestested in using Stack Overflow data from 2020 to better understand:

From some of the questions that were asked in the 2020 survey, I would like to get a better understaning of the following questions:

1. Which programming, scripting, and markup languages have the developers used in thir work over the past year?
2. Does the company onboarding process have any link to job satisfaction of the developers?
3. Based on the age of the developer, do they have any preference to the collaboration tool they use?

## File Descriptions <a name="files"></a>

1. ETL Pipeline

In a Python script, process_data.py, write a data cleaning pipeline that:

Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database

2. ML Pipeline

In a Python script, train_classifier.py, write a machine learning pipeline that:

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file

3. Flask Web App

The web app enables the user to enter a disaster message, and then view the categories of the message.


## Results<a name="results"></a>

Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

The web app enables the user to enter a disaster message, and then view the categories of the message.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credit to [Figure Eight](https://www.figure-eight.com/) for the data. 

