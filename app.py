import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Student Note Tagger", page_icon="🏷️")
st.title("🏷️ Student Note Topic Tagger")
st.caption("Prototype using synthetic educational notes. Suggested tags are not authoritative; your edits take priority.")

@st.cache_resource
def load_model():
    data = pd.read_csv("synthetic_student_notes.csv")
    data["tags"] = data["tags_string"].fillna("").apply(lambda s: [t for t in s.split(";") if t])
    tax = pd.read_csv("topic_taxonomy.csv")
    mlb = MultiLabelBinarizer(classes=tax["tag"].tolist())
    Y = mlb.fit_transform(data["tags"])
    X_train, _, y_train, _ = train_test_split(data["note_text"], Y, test_size=0.25, random_state=42)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1,2),
                                  sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_+#.-]*\b")),
        ("clf", OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight="balanced")))
    ])
    model.fit(X_train, y_train)
    return model, mlb, data, tax

model, mlb, data, taxonomy = load_model()
note = st.text_area("Enter a note", height=160, placeholder="Example: A stack follows last-in first-out order...")
if st.button("Suggest tags"):
    if not note.strip():
        st.warning("Please enter a note first.")
    else:
        scores = model.decision_function([note])[0]
        order = scores.argsort()[::-1]
        selected = [mlb.classes_[i] for i in order if scores[i] > 0][:4]
        if not selected:
            selected = [mlb.classes_[i] for i in order[:2]]
        st.subheader("Suggested tags")
        chosen = st.multiselect("Edit tags — add or remove anything", options=list(mlb.classes_), default=selected)
        st.success("Your final tags: " + (", ".join(chosen) if chosen else "No tags selected"))
        st.info("User edits override all model suggestions. Avoid entering confidential or personally identifying information.")
st.divider()
st.subheader("Search example notes")
query = st.text_input("Search the synthetic dataset")
if query:
    found = data[data["note_text"].str.contains(query, case=False, na=False)]
    st.dataframe(found[["note_id","note_text","tags_string"]], use_container_width=True)
st.caption("Privacy: this classroom prototype uses a local synthetic dataset. Do not enter sensitive personal information.")
