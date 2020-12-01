import pickle
from flask import Flask, request, render_template
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np


app = Flask(__name__)
model = pickle.load(open('svc.pickle', 'rb'))
word2vec_model = pickle.load(open('word2vec_model.pickle', 'rb'))

contractions = {
"n't": " not",
"'ve": " have",
"'cause": "because",
" 'd": " would",
"'ll": " will",
"'s": " is",
"i'm": "i am",
"ma'am": "madam",
"<br />": " "
}

def aggregate_vectors(products):
    product_vec = []
    for i in products:
        if i in word2vec_model.wv:
            product_vec.append(word2vec_model.wv[i])
    return np.mean(product_vec, axis=0)

# negative -->> 0
# positive -->> 1
def pre_process(review):
    review = review.lower()
    for contraction in contractions:
        review = review.replace(contraction, contractions[contraction])
    review = ''.join([c for c in review if c not in punctuation])
    lemmetizer = WordNetLemmatizer()
    words = word_tokenize(review)
    words = [lemmetizer.lemmatize(word) for word in words if (word=='not' or word not in set(stopwords.words("english")))]
    vec = aggregate_vectors(words)
    return vec
 
@app.route('/',methods=['GET','POST'])
def home():
    if request.method == "POST": 
        review = [x for x in request.form.values()]
        final_features = [pre_process(review[0])]
        if len(final_features[0].shape)==0:
            return render_template('index.html', prediction_text= 'You entered gibberish.', review_text=review[0])
        else:
            prediction = model.predict(final_features)
            output = 'Negative' if prediction[0]==0 else 'Positive'
            return render_template('index.html', prediction_text='Sentiment of review is "{}"'.format(output), review_text=review[0])

    else:
        return render_template('index.html')

   
if __name__ == "__main__":
    app.run(debug=True)