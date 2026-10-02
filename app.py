from flask import Flask, render_template, request
import pickle, sqlite3, json
import numpy as np
from datetime import datetime

app = Flask(__name__)
data = pickle.load(open("model.pkl", "rb"))
model, scaler = data["model"], data["scaler"]

FIELDS = [
    ("Pregnancies", "Pregnancies (number)"),
    ("Glucose", "Glucose (mg/dL)"),
    ("BloodPressure", "Blood Pressure (mm Hg)"),
    ("SkinThickness", "Skin Thickness (mm)"),
    ("Insulin", "Insulin (mu U/ml)"),
    ("BMI", "BMI"),
    ("DiabetesPedigreeFunction", "Diabetes Pedigree Function"),
    ("Age", "Age (years)"),
]

def db():
    con = sqlite3.connect("history.db")
    con.execute("""CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        time TEXT, inputs TEXT, risk TEXT, prob REAL)""")
    return con

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/insights")
def insights():
    chart_data = {
        "models": list(data["results"].keys()),
        "accuracy": [r["accuracy"] for r in data["results"].values()],
        "features": list(data["importance"].keys()),
        "importance": list(data["importance"].values()),
        "distribution": data["distribution"],
    }
    return render_template("insights.html", best=data["best_name"],
                           results=data["results"], cm=data["cm"],
                           chart_data=chart_data)

@app.route("/history")
def history():
    con = db()
    rows = con.execute("SELECT time, inputs, risk, prob FROM history "
                       "ORDER BY id DESC LIMIT 20").fetchall()
    con.close()
    rows = [(t, json.loads(i), r, p) for t, i, r, p in rows]
    return render_template("history.html", rows=rows)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    if request.method == "POST":
        try:
            inputs = {f: float(request.form[f]) for f, _ in FIELDS}
            arr = scaler.transform(np.array(list(inputs.values())).reshape(1, -1))
            pred = int(model.predict(arr)[0])
            prob = round(float(model.predict_proba(arr)[0][1]) * 100, 1)
            result = {"pred": pred, "prob": prob}
            con = db()
            con.execute("INSERT INTO history (time, inputs, risk, prob) VALUES (?,?,?,?)",
                        (datetime.now().strftime("%d-%m-%Y %H:%M"),
                         json.dumps(inputs), "High" if pred else "Low", prob))
            con.commit()
            con.close()
        except ValueError:
            result = {"error": "Sabhi fields me sahi number daalo."}
    return render_template("predict.html", fields=FIELDS, result=result)

if __name__ == "__main__":
    app.run(debug=True)