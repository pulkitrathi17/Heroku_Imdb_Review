import pickle
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np


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

word2vec_model = pickle.load(open('word2vec_model.pickle', 'rb'))

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
    #return ' '.join(words)
       
    


