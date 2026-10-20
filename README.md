# AI-Powered Automatic Topic Tagging for Student Notes

A small educational NLP project that suggests topic tags for student notes. It compares a keyword/rule baseline with a TF-IDF + one-vs-rest Logistic Regression classifier and includes a Streamlit interface where users can review and correct suggested tags.

> **Status:** Educational prototype. The included dataset is synthetic and small; model results should not be treated as production performance.

## Features

- Multi-label topic suggestions for student notes
- Compact taxonomy of 13 topics
- Text preprocessing and TF-IDF feature extraction
- Keyword/rule-based baseline and supervised classifier
- Precision, recall, F1-score and exact-match evaluation
- Error-analysis CSV for inspecting missed and incorrect tags
- Streamlit demo with user-editable tags
- Synthetic data only; no real student records are included

## Repository contents

| File | Purpose |
|---|---|
| `app.py` | Interactive Streamlit prototype |
| `topic_tagging_notebook.ipynb` | Notebook for preprocessing, training and evaluation |
| `synthetic_student_notes.csv` | Synthetic notes and labels |
| `topic_taxonomy.csv` | Topic names and descriptions |
| `evaluation_results.csv` | Evaluation summary |
| `test_predictions_error_analysis.csv` | Test predictions for error analysis |
| `Project_Report.pdf` | Project report |
| `requirements.txt` | Python dependencies |

## Topic taxonomy

The project uses these tags:

- Programming
- Algorithms
- Data Structures
- Databases
- Machine Learning
- Statistics
- Web Development
- Software Engineering
- Cybersecurity
- Artificial Intelligence
- Information Retrieval
- Data Analysis
- Responsible AI

A note can have more than one topic.

## Quick start (Windows)

1. Install Python 3.10 or newer from [python.org](https://www.python.org/downloads/).
2. Download or clone this repository.
3. Open a terminal in the repository folder.
4. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

5. Activate it in PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt:

   ```bat
   .venv\Scripts\activate.bat
   ```

6. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

7. Start the demo:

   ```bash
   python -m streamlit run app.py
   ```

8. Open the local URL shown in the terminal (usually `http://localhost:8501`).

## Run the notebook

Open `topic_tagging_notebook.ipynb` in VS Code with the Python and Jupyter extensions, or upload the notebook and its CSV files to [Google Colab](https://colab.research.google.com/). Run the cells from top to bottom.

## Example use

1. Enter or select a sample note.
2. View the suggested topic tags.
3. Review the suggestions and change the selected tags as needed.
4. Treat the model's output as a recommendation, not a guaranteed correct classification.

## Evaluation notes

The demonstration uses a small synthetic dataset and a fixed random seed. A small hold-out set can produce unstable metrics, and the model may miss relevant tags or suggest incorrect ones—especially for very short notes, code-heavy notes, unfamiliar terms, and overlapping topics. See `evaluation_results.csv` and `test_predictions_error_analysis.csv`.

To improve the project, collect a larger permitted dataset, manually review labels, use a validation set for threshold tuning, and evaluate performance across subjects.

## Privacy and responsible use

- The included examples are synthetic.
- Do not upload confidential notes, personal information, or private student records to a public repository.
- For any real educational material, verify the licence and obtain permission before use.
- Predictions are suggestions; user corrections should take precedence.
- Review false positives and false negatives before using a model in a real workflow.

## Reproducibility

The notebook documents the training and evaluation process. The project uses a fixed random seed for its train/test split. Package versions are listed in `requirements.txt`; small differences may occur across Python and dependency versions.

## Tools

- Python
- pandas
- scikit-learn
- Streamlit
- Jupyter Notebook / Google Colab

## License

This repository is provided for educational use under the MIT License. The license applies to the project code and documentation; it does not override licences for any third-party material added later.
