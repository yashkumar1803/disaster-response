import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
import pickle
from collections import Counter
import string

def remove_words(corpus):
    word_counts = Counter()
    for sent in corpus.apply(lambda x: tokenize(x)):
        word_counts.update(sent)
        
    stop_words = set([k for k in word_counts.keys() if (word_counts[k] == 1)])
    return stop_words


def load_data(database_filepath):
    engine = create_engine("""sqlite:///""" +  database_filepath)
    
    df = pd.read_sql_table("DisasterResponse", engine).drop(["original", "genre"], axis=1)
    df = df.dropna(how="any")
    
    X = df["message"]
    Y = df.drop(["message", "index", "id", "categories"], axis=1)
    Y = Y.loc[:, Y.sum() != 0]
    category_names = list(Y.columns)
    return X, Y, category_names


def tokenize(text):
    
    lemmatizer = WordNetLemmatizer()
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    return text.replace("""'""", "").lower().translate(translator).split()
#     return [lemmatizer.lemmatize(word) for word
#             in text.replace("""'""", "").lower().translate(translator).split()]


def build_model(X_train):
    clf = RandomForestClassifier()
    mlt_outp = MultiOutputClassifier(clf)
    
    words_remove = remove_words(X_train)
    pipeline = Pipeline([("count_vect", CountVectorizer(tokenizer=tokenize, stop_words=words_remove)),
                    ("tfidf", TfidfTransformer()),
                    ("multioutput_classifier", mlt_outp)])

    pipeline.get_params()
    params = {
    }
    
    cv = GridSearchCV(pipeline, param_grid=params, cv=3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
#     Calculate accuracy, precision, recall
#     labels = np.unique(y_pred)
    Y_pred = model.predict(X_test)
    
#     confusion_mat = confusion_matrix(Y_test, Y_pred, labels=labels)
    accuracy = (Y_pred == Y_test).mean()
    print("Accuracy:", accuracy)

#     print("Labels:", labels)
#     print("Confusion Matrix:\n", confusion_mat)
#     print("Accuracy:", accuracy)
#     print("\nBest Parameters:", cv.best_params_)

#     pass


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        
     
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
#         print(Y_train[:10])
        
        print('Building model...')
        model = build_model(X_train)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()