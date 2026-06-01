import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Fallback roles if no match is found
TRENDING_ROLES = ["Data Scientist", "Full Stack Developer", "Cloud Architect"]

def custom_tokenizer(text):
    # Split input by spaces and convert to lowercase
    return [skill.strip().lower() for skill in text.split()]

def main():
    # Load the job roles dataset
    file_name = 'raw_skills.csv'
    if not os.path.exists(file_name):
        print(f"Error: {file_name} not found.")
        return
        
    df = pd.read_csv(file_name)

    # 1. Ingestion: Get user skills
    user_input = input("Enter your skills (space-separated): ")
    user_skills = [s.strip() for s in user_input.split() if s.strip()]
    
    if len(user_skills) < 3:
        print("Note: Providing at least 3 skills will give you better recommendations!")

    # 2. Scoring: Compute TF-IDF vectors and Cosine Similarity
    tfidf = TfidfVectorizer(tokenizer=custom_tokenizer, token_pattern=None)
    tf_idf_matrix = tfidf.fit_transform(df['Skills'])
    user_vector = tfidf.transform([user_input])
    
    # Handle Cold Start Problem
    if user_vector.sum() == 0:
        print("\nOops! None of your skills matched our current database.")
        print("Here are some trending roles you might be interested in instead:")
        for idx, role in enumerate(TRENDING_ROLES, start=1):
            print(f"{idx}. {role}")
        return

    # Calculate similarity between user skills and job roles
    cosine_sim = cosine_similarity(user_vector, tf_idf_matrix)
    
    # 3. Sorting: Rank jobs by similarity score
    df['Match Score'] = cosine_sim[0]
    df_sorted = df.sort_values(by='Match Score', ascending=False)
    
    # 4. Filtering: Keep only the top 3 matches
    top_N = 3
    df_top3 = df_sorted.head(top_N)
    
    # Display the results
    print("\nTop 3 Recommended Career Paths:")
    for idx, row in enumerate(df_top3.iterrows(), start=1):
        role_name = row[1]['Role']
        score = row[1]['Match Score'] * 100 
        print(f"{idx}. {role_name:<20} — Match: {score:.0f}%")

if __name__ == "__main__":
    main()