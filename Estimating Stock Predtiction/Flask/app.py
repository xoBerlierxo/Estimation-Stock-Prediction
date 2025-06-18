from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open('sales_demand_forecasting.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    inputs = [float(x) for x in request.form.values()]
    final_input = np.array([inputs])
    prediction = model.predict(final_input)
    output = round(prediction[0], 2)
    return render_template('result.html', prediction_text=f'The predicted demand is {output}')

if __name__ == "__main__":
    app.run(debug=True)