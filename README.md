# Disaster Response Pipeline Project
This project build disaster message classification website that can classify the input message to the related categories so that it is easier for responsible organizations to make decision and respond to each message. 

The website use machine learning algorithm to classify the message. The machine learning is trained by using Figure Eight's example messages and the categories.

## The Objectives of this Project
To improve classification accuracy of message sending on various medium during disaster by using machine learning algorithm so that the responsible organization can understand the need without go thru every messages.

## Instructions to run the website:
1. Install Python - https://www.python.org/downloads/
2. Install Flask - https://github.com/pallets/flask
3. Download all files of this project and place them to Flask's web app directory.
4. Install custom package (AdvAdjExtractor) by go to the project "package" directory and run the command: `pip install .`. This is used by the web app and the script that trains the classifier.
5. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
6. Run the following command in the app's directory to run your web app.
    `python run.py`

## Required Libraries
- pandas
- sklearn
- nltk
- sqlalcheme
- re
- pickle
- sklearn
- AdvAdjExtractor (custom module for this project)

## File Description
1. app/templates: The html template of Flask used for rendering the web application.
2. app/run.py: The Flask web server.
3. data/disaster_categories.csv: Data for train the model.
4. data/disaster_messages.csv: Data for train the model.
5. data/process_data.py: Script for processing the data to be ready for train the model.
6. models/train_classifier.py: Script for train the model.
7. package/distributions/adv_adj_extractor.py: Script for custom class that will add additional features to the data.
8. package/setup.py: Script required for set up the package by using `pip install`.
9. Jupyter Notebook: Contain the Jupyter Notebook file used to design process_data.py and train_classifier.py with note on rationale.

# Result Summary
By adding new feature to the data - the count of adverb and adjectives and train the classifier model using Random Forest, the model is able to classify the message with high accuracy than without this additional feature.