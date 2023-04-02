from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')

# Load the model
with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    print('Received input:', int_features)
    final_features = [np.array(int_features)]
    print('Final input:', final_features)
    prediction = model.predict(final_features)

    iris_names = ['Setosa', 'Versicolor', 'Virginica']
    predicted_name = iris_names[int(prediction[0])]

    return render_template('index.html', prediction_text='Predicted Iris Species: {}'.format(predicted_name))

if __name__ == "__main__":
    app.run(debug=True)
