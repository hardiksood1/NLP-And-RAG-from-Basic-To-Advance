from flask import Flask, render_template, request, redirect
import csv
import os

app = Flask(__name__)

# CSV file path
CSV_FILE = "ideas.csv"

# Ensure CSV file has headers if not exists
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Idea Name", "Idea Description", "Email", "Contact"])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/submit", methods=["POST"])
def submit():
    if request.method == "POST":
        name = request.form["name"]
        idea_name = request.form["idea_name"]
        idea_desc = request.form["idea_desc"]
        email = request.form["email"]
        contact = request.form["contact"]

        # Save to CSV
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([name, idea_name, idea_desc, email, contact])

        return redirect("/")  # Redirect to home page after submission

if __name__ == "__main__":
    app.run(debug=True)