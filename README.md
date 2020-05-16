# Disaster Response Pipeline

## Table of Contents

- [Project Overview](#overview)
- [Project Components](#components)
  - [ETL Pipeline](#etl_pipeline)
  - [ML Pipeline](#ml_pipeline)
  - [Flask Web App](#flask)
- [Running](#run)
  - [Data Cleaning](#cleaning)
  - [Training Classifier](#training)
  - [Starting the Web App](#starting)
- [Files](#files)
- [Software Requirements](#sw)  
- [Screenshots](#ss)
- [Credits and Acknowledgements](#credits)

***

<a id='overview'></a>

## 1. Project Overview

In this project, I'll apply data engineering to analyze disaster data from <a href="https://www.figure-eight.com/" target="_blank">Figure Eight</a> to build a model for an API that classifies disaster messages.

_data_ directory contains a data set which are real messages that were sent during disaster events. I will be creating a machine learning pipeline to categorize these events so that appropriate disaster relief agency can be reached out for help.

I will also build a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

[Here](#eg) are a few screenshots of the web app.

<a id='components'></a>

## 2. Project Components

There are three components of this project:

<a id='etl_pipeline'></a>

### 2.1. ETL Pipeline

File _data/process_data.py_ contains data cleaning pipeline that:

- Loads the `messages` and `categories` dataset
- Merges the two datasets
- Cleans the data
- Stores it in a **SQLite database**

<a id='ml_pipeline'></a>

### 2.2. ML Pipeline

File _models/train_classifier.py_ contains machine learning pipeline that:

- Loads data from the **SQLite database**
- Splits the data into training and testing sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Evaluate model on the test set
- Exports the final model as a pickle file

<a id='flask'></a>

### 2.3. Flask Web App

<a id='eg'></a>

Running [this command](#com) **from app directory** will start the web app where users can enter their query and get results.  
The app will classify the text message into categories so that appropriate relief agency can be reached out for help.


<a id='run'></a>

## 3. Running

There are three steps to get up and runnning with the web app if you want to start from ETL process.

<a id='cleaning'></a>

### 3.1. Data Cleaning

**Go to the project directory** and the run the following command:

```bat
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

The first two arguments are input data and the third argument is the SQLite Database in which we want to save the cleaned data. The ETL pipeline is in _process_data.py_.

_DisasterResponse.db_ already exists in _data_ folder but the above command will still run and replace the file with same information. 


<a id='training'></a>

### 3.2. Training Classifier

After the data cleaning process, run this command **from the project directory**:

```bat
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

This will use cleaned data to train the model, improve the model with grid search and saved the model to a pickle file (_classifer.pkl_).

_classifier.pkl_ already exists but the above command will still run and replace the file will same information.


When the models is saved, it will look something like this.

<a id='acc'></a>

<a id='starting'></a>

### 3.3. Starting the web app

Now that we have cleaned the data and trained our model. Now it's time to see the prediction in a web application.

**Go the app directory** and run the following command:

<a id='com'></a>

```bat
python run.py
```

This will start the web app.  
The app can be accessed at http://0.0.0.0:3001/

<a id='conclusion'></a>


<a id='files'></a>

## 4. Files

<pre>
.
├── app
│   ├── run.py------------------------# FLASK FILE THAT RUNS APP
│   ├── DisasterResponse.db
│   │   
│   └── templates
│       ├── go.html-------------------# CLASSIFICATION RESULT PAGE OF WEB APP
│       └── master.html---------------# MAIN PAGE OF WEB APP
├── data
│   ├── DisasterResponse.db-----------# DATABASE TO SAVE CLEANED DATA TO
│   ├── disaster_categories.csv-------# DATA CONTAINING DISASTER CATEGORIES
│   ├── disaster_messages.csv---------# DATA CONTAINING DISASTER MESSAGES
│   └── process_data.py---------------# PERFORMS ETL PROCESS
├── requirements.txt-------------------------------# CONTAINING ALL DEPENDENCIES
├── models
│   ├── train_classifier.py-----------# PERFORMS CLASSIFICATION TASK
    └── classifier.pkl----------------# SAVED MODEL

</pre>

<a id='sw'></a>

## 5. Software Requirements

This project uses **Python 3.6** and the necessary libraries are mentioned in _requirements.txt_.

<a id='ss'></a>

## 6. Screenshots

![Home Page](https://user-images.githubusercontent.com/40590709/82116592-95a22780-9788-11ea-8262-6eeee5e39087.png)  

![Home Page Visualization 1](https://user-images.githubusercontent.com/40590709/82116611-bd918b00-9788-11ea-9d33-e186294e51fa.png)  

![Home Page Visualization 2](https://user-images.githubusercontent.com/40590709/82116624-d601a580-9788-11ea-83a3-e26c3dd33f5b.png)  

![Go Route](https://user-images.githubusercontent.com/40590709/82116695-51635700-9789-11ea-9f77-13ecdc0e39da.png)  

![Go Route Visualization](https://user-images.githubusercontent.com/40590709/82116735-7ce64180-9789-11ea-847f-3847d53ce6ae.png)


<a id='credits'></a>

## 7. Credits and Acknowledgements

Thanks to [Udacity](https://www.udacity.com/) and [Figure Eight](https://www.figure-eight.com) project for providing original dataset with [Multilingual Disaster Response Messages](https://www.figure-eight.com/dataset/combined-disaster-response-data).



