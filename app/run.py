import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

import warnings
warnings.filterwarnings('ignore')



app = Flask(__name__)

current_values = [0] * 36

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages_categories', engine)
df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
X = df['message']
y = df.iloc[:,4:]

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    


    category_names = y.columns
    category_values = []
    for col in category_names:
        category_values.append(26177 - int(df[col].value_counts()[0]))
    
    
    category_values, category_names = zip(*sorted(zip(category_values, category_names), reverse=True))
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
                {
            'data': [
                Bar(
                    x=category_names,
                    y=category_values
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    

    current_cat = y.columns
    i = 0
    global current_values
    for key, value in classification_results.items():
        if value == 1:
            current_values[i] +=1
        i+=1
    
    graphs2 = [
        {
            'data': [
                Bar(
                    x=current_cat,
                    y=current_values
                )
            ],

            'layout': {
                'title': 'Message Categoried Till Now',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }]
    
    ids2 = ["graph-{}".format(i) for i, _ in enumerate(graphs2)]
    graphJSON2 = json.dumps(graphs2, cls=plotly.utils.PlotlyJSONEncoder)
            
        
        

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results,
        ids=ids2, graphJSON=graphJSON2
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()