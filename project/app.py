import streamlit as st
import pdfplumber
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

def load_job_descriptions():
    job_descriptions = {}
    
    # Get the absolute path of the directory containing app.py
    # This makes the path robust regardless of where the app is run from
    current_dir = os.path.dirname(__file__)
    
    # Construct the path to the 'job_descriptions' folder
    job_descriptions_folder = os.path.join(current_dir, "job_descriptions")

    # Check if the directory exists (good for debugging)
    if not os.path.exists(job_descriptions_folder):
        st.error(f"Error: 'job_descriptions' folder not found at {job_descriptions_folder}")
        return {} # Or raise an error, depending on desired behavior

    # Iterate through files in the job_descriptions folder
    for filename in os.listdir(job_descriptions_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(job_descriptions_folder, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    job_title = os.path.splitext(filename)[0] # e.g., "data_scientist"
                    job_descriptions[job_title] = f.read()
            except Exception as e:
                st.error(f"Error reading file {filepath}: {e}")
    return job_descriptions

def calculate_similarity(resume_text, job_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_text])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

def load_skills():
    with open("skills_list.txt", "r", encoding='utf-8') as f:
        return [line.strip().lower() for line in f if line.strip()]

def recommend_skills(resume_text, job_text):
    all_skills = load_skills()
    resume_words = set(resume_text.lower().split())
    job_words = set(job_text.lower().split())

    resume_skills = {skill for skill in all_skills if skill in resume_words}
    job_skills = {skill for skill in all_skills if skill in job_words}

    missing_skills = job_skills - resume_skills
    return sorted(missing_skills)


st.set_page_config(page_title="AI Resume Analyzer", layout="centered")
st.title("🧠 AI-Powered Resume Analyzer & Job Matcher")

st.write("Upload your resume and see how well it matches real job descriptions.")

uploaded_file = st.file_uploader("📄 Upload your Resume (PDF)", type=["pdf"])

if uploaded_file:
    resume_text = extract_text_from_pdf(uploaded_file)
    resume_clean = clean_text(resume_text)

    job_descriptions = load_job_descriptions()
    scores = {}

    for role, jd_text in job_descriptions.items():
        jd_clean = clean_text(jd_text)
        similarity = calculate_similarity(resume_clean, jd_clean)
        scores[role] = round(similarity * 100, 2)

    best_match = max(scores, key=scores.get)
    best_jd_text = job_descriptions[best_match]
    recommended = recommend_skills(resume_clean, clean_text(best_jd_text))

    st.subheader("🔍 Match Results")
    st.write(f"**Best Matched Role:** `{best_match}`")
    st.write(f"**Match Score:** `{scores[best_match]}%`")

    st.bar_chart(scores)

    st.subheader("🧩 Recommended Skills to Add:")
    if recommended:
        st.markdown(", ".join(recommended))
    else:
        st.markdown("✅ Your resume covers most relevant skills!")

    with st.expander("📄 View Extracted Resume Text"):
        st.text(resume_text)
else:
    st.info("Please upload a PDF resume to begin.")



