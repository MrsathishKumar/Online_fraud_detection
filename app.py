from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
import pandas as pd
import sqlite3
import os  
import matplotlib.pyplot as plt
import seaborn as sns
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
app = Flask(__name__)
app.secret_key = "your_secret_key"

load_dotenv()

# ------------------ Load Model & Encoders ------------------
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
le_location = joblib.load("encoder_location.pkl")
le_phone = joblib.load("encoder_phone.pkl")

# ------------------ Load PCA Mean Values ------------------
if os.path.exists("pca_mean_values.pkl"):
    pca_mean_values = joblib.load("pca_mean_values.pkl")
else:
    pca_mean_values = [0.0] * 5

# ------------------ User Credentials ------------------
USER_USERNAME = "user"
USER_PASSWORD = "password"

# ------------------ Create Database Table ------------------
def create_table():
    conn = sqlite3.connect("transactions.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT,
            amount REAL,
            phone_number TEXT,
            location TEXT,
            fraud_probability REAL,
            prediction TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


create_table()

# ------------------ User Login Route ------------------
@app.route("/")
def user():
    return render_template("user.html")

@app.route("/login", methods=["POST"])
def login():
    username = request.form["username"]
    password = request.form["password"]

    conn = sqlite3.connect("transactions.db")
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()

    if row and check_password_hash(row[0], password):
        session["user_logged_in"] = True
        return redirect(url_for("home"))
    else:
        return render_template("user.html", error="Invalid credentials")




@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect("transactions.db")
        cursor = conn.cursor()

        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
            conn.commit()
            return redirect(url_for("user"))
        except sqlite3.IntegrityError:
            error = "Username already exists."
            return render_template("register.html", error=error)
        finally:
            conn.close()
    return render_template("register.html")

    
@app.route("/home")
def home():
    if not session.get("user_logged_in"):
        return redirect(url_for("user"))
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not session.get("user_logged_in"):
            return redirect(url_for("user"))

        transaction_id = request.form["transaction_id"]
        amount = float(request.form["amount"])
        phone_number = request.form["phone_number"]
        location = request.form["location"]

        location_encoded = le_location.transform([location])[0] if location in le_location.classes_ else -1
        phone_encoded = le_phone.transform([phone_number])[0] if phone_number in le_phone.classes_ else -1

        pca_features = get_pca_features(transaction_id)
        if not pca_features or all(x == 0.0 for x in pca_features):
            pca_features = pca_mean_values

        input_features = pd.DataFrame(
            [[amount, location_encoded, phone_encoded] + pca_features],
            columns=["Amount", "IP_Location", "Phone_Number", "V1", "V2", "V3", "V4", "V5"]
        )

        input_scaled = scaler.transform(input_features)
        fraud_probability = model.predict_proba(input_scaled)[0][1]

        threshold = 0.5
        prediction = "🚨 Fraudulent Transaction" if fraud_probability >= threshold else "✅ Legitimate Transaction"
        is_fraudulent = fraud_probability >= threshold

        conn = sqlite3.connect("transactions.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO transactions 
            (transaction_id, amount, phone_number, location, fraud_probability, prediction) 
            VALUES (?, ?, ?, ?, ?, ?)
        """, (transaction_id, amount, phone_number, location, fraud_probability, prediction))
        conn.commit()
        conn.close()

        return render_template(
            "result.html",
            transaction_id=transaction_id,
            result=prediction,
            probability=fraud_probability,
            is_fraudulent=is_fraudulent
        )

    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/report_transaction', methods=['POST'])
def report_transaction():
    transaction_id = request.form.get('transaction_id')
    return render_template('report.html', transaction_id=transaction_id)

def get_pca_features(transaction_id):
    try:
        conn = sqlite3.connect("transactions.db")
        cursor = conn.cursor()
        cursor.execute("SELECT V1, V2, V3, V4, V5 FROM transactions WHERE transaction_id = ?", (transaction_id,))
        pca_data = cursor.fetchone()
        conn.close()
        return list(pca_data) if pca_data else None
    except:
        return None

@app.route("/graph")
def graph():
    if not session.get("user_logged_in"):
        return redirect(url_for("user"))

    conn = sqlite3.connect("transactions.db")
    df = pd.read_sql_query("SELECT fraud_probability, prediction FROM transactions", conn)
    conn.close()

    if df.empty:
        return "No transaction data available."

    df["Is_Fraud"] = df["prediction"].apply(lambda x: 1 if "Fraud" in x else 0)
    fraud_count = df["Is_Fraud"].sum()
    legit_count = len(df) - fraud_count

    fraud_percentage = (fraud_count / len(df)) * 100
    legit_percentage = 100 - fraud_percentage

    if not os.path.exists("static"):
        os.makedirs("static")

    plt.figure(figsize=(6, 4))
    labels = [f"Fraud ({fraud_percentage:.1f}%)", f"Legitimate ({legit_percentage:.1f}%)"]
    plt.pie([fraud_percentage, legit_percentage], labels=labels, autopct="%1.1f%%", colors=["red", "green"], startangle=90)
    plt.title("Fraudulent vs. Legitimate Transactions (Percentage)")
    plt.savefig("static/fraud_pie_chart.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(df["fraud_probability"], bins=20, kde=True, color="purple", stat="percent")
    plt.xlabel("Fraud Probability")
    plt.ylabel("Percentage (%)")
    plt.title("Fraud Probability Distribution (Percentage)")
    plt.savefig("static/fraud_probability_distribution.png")
    plt.close()

    return render_template("graph.html")

@app.route("/report_fraud_page", methods=["GET"])
def report_fraud_page():
    if not session.get("user_logged_in"):
        return redirect(url_for("user"))

    return render_template("report.html",
        transaction_id=request.args.get("transaction_id"),
        amount=request.args.get("amount"),
        phone_number=request.args.get("phone_number"),
        location=request.args.get("location")
    )

@app.route("/report_fraud", methods=["POST"])
def report_fraud():
    if not session.get("user_logged_in"):
        return redirect(url_for("user"))

    transaction_id = request.form["transaction_id"]
    amount = float(request.form["amount"])
    phone_number = request.form["phone_number"]
    location = request.form["location"]
    reason = request.form["reason"]
    conn = sqlite3.connect("transactions.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fraud_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT NOT NULL,
        amount REAL,
        phone_number TEXT,
        location TEXT,
        reason TEXT
    )
    """)

    cursor.execute("""
        INSERT INTO fraud_reports (transaction_id, amount, phone_number, location, reason) 
        VALUES (?, ?, ?, ?, ?)
    """, (transaction_id, amount, phone_number, location, reason))
    conn.commit()
    conn.close()

    return redirect(url_for("view_report", transaction_id=transaction_id))

@app.route("/view_report/<transaction_id>")
def view_report(transaction_id):
    if not session.get("user_logged_in"):
        return redirect(url_for("user"))

    conn = sqlite3.connect("transactions.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT transaction_id, amount, phone_number, location, reason, reported_at 
        FROM fraud_reports WHERE transaction_id = ?
    """, (transaction_id,))
    report = cursor.fetchone()
    conn.close()

    if report:
        report_data = {
            "transaction_id": report[0],
            "amount": report[1],
            "phone_number": report[2],
            "location": report[3],
            "reason": report[4],
            "reported_at": report[5] if len(report) > 5 else ""
        }
        return render_template("view_report.html", report=report_data)
    else:
        return "Report not found!", 404

SENDER_EMAIL = os.getenv('SENDER_EMAIL')
SENDER_PASSWORD = os.getenv('EMAIL_PASSWORD')
SMTP_SERVER = os.getenv('SMTP_SERVER')
SMTP_PORT = int(os.getenv('SMTP_PORT'))
RECEIVER_EMAIL = os.getenv('RECEIVER_EMAIL')

@app.route('/submit_report', methods=['POST'])
def submit_report():
    transaction_id = request.form['transaction_id']
    user_name = request.form['user_name']
    reason = request.form['reason']
    user_email = request.form['email']

    html_content = render_template('report_confirmation.html',
                                   transaction_id=transaction_id,
                                   user_name=user_name,
                                   reason=reason)

    send_confirmation_email(user_email, html_content)

    return render_template('report_confirmation.html',
                           transaction_id=transaction_id,
                           user_name=user_name,
                           reason=reason)

def send_confirmation_email(receiver_email, html_content):
    msg = MIMEMultipart('alternative')
    msg['Subject'] = "🛡️ Your Report Has Been Received"

    msg['From'] = SENDER_EMAIL
    msg['To'] = receiver_email

    part = MIMEText(html_content, 'html')
    msg.attach(part)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, receiver_email, msg.as_string())
            print("Email sent successfully")
    except Exception as e:
        print(f"Error sending email: {e}")

@app.route("/logout")
def logout():
    session.pop("user_logged_in", None)
    return redirect(url_for("user"))

if __name__ == "__main__":
    app.run(debug=True)
