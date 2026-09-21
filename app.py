from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained ML model
model = joblib.load("metro_ridership_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        country = request.form["country"]
        city = request.form["city"]
        lines = float(request.form["lines"])
        system_length = float(request.form["system_length"])
        year = int(request.form["year"])

        # Create input dataframe
        input_data = pd.DataFrame({
            "Country": [country],
            "City": [city],
            "Lines": [lines],
            "System_Length_km": [system_length],
            "Year": [year]
        })

        # Prediction
        prediction = model.predict(input_data)[0]

        prediction = round(prediction, 2)

        return render_template(
            "index.html",
            prediction=prediction,
            country=country,
            city=city
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)