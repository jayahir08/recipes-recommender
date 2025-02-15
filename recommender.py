import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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