from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


def load_data(filepath):
    df = pd.read_csv(filepath)
    df["Ingredients"] = df["Ingredients"].fillna("")
    df["Ingredients"] = df["Ingredients"].apply(lambda x: " ".join(x.lower().split(",")))
    return df

def recommend_recipes(df, user_ingredients, top_n=5):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["Ingredients"])

    user_input = " ".join(user_ingredients).lower()
    user_vector = vectorizer.transform([user_input])

    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()

    top_indices = similarities.argsort()[-top_n:][::-1]
    return df.iloc[top_indices]

df = load_data("data/recipes.csv")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_ingredients = request.form["ingredients"].split(",")
        recommendations = recommend_recipes(df, user_ingredients)
        
        recommendations = recommendations.to_dict("records")
        return render_template("index.html", recommendations=recommendations)
    
    return render_template("index.html", recommendations=[])

@app.route("/recipe/<int:srno>")
def recipe_details(srno):
    recipe = df[df["Srno"] == srno].iloc[0]
    return render_template("recipe_details.html", recipe=recipe)

if __name__ == "__main__":
    app.run(debug=True)