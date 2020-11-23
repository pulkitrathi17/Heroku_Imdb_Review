import pickle
from flask import Flask, request, jsonify, render_template
from preprocess import pre_process


app = Flask(__name__)
model = pickle.load(open('svc.pickle', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    review = [x for x in request.form.values()]
    final_features = [pre_process(review[0])]
    prediction = model.predict(final_features)

    output = 'Bad' if prediction[0]==0 else 'Good'

    return render_template('index.html', prediction_text='Review Sentiment is "{}"'.format(output))

if __name__ == "__main__":
    app.run(debug=True)






