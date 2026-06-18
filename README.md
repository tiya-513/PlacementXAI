# PlacementXAI

A full-stack web application for student placement prediction with explainable AI. Built with Django, TensorFlow, and SHAP — tells you *whether* you'll get placed, *what salary* to expect, and *why*.

---

## What it does

- Predicts placement status (placed / not placed) and expected LPA for a student profile
- Uses a trained neural network (TensorFlow/Keras) under the hood
- Explains every prediction with SHAP — shows which factors contributed how much, per student
- Full user auth: sign up, log in, update profile, delete account (with password confirmation)
- All data stored locally in SQLite via Django ORM

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3, Django |
| ML Model | TensorFlow / Keras |
| Explainability | SHAP |
| Database | SQLite (via Django ORM) |
| Frontend | HTML, CSS, JavaScript |
| Auth | Django auth + bcrypt password hashing |

---

## Project Structure

```
PlacementXAI/
│
├── model/                          # ML training, saved model, SHAP analysis
│   ├── train.py                    # Neural network training + architecture search
│   ├── shap_analysis.py            # Global + local SHAP value computation
│   ├── cli_demo.py                 # Step 3: terminal-based demo interface
│   ├── my_model.keras              # Saved trained model
│   └── experiment_log.csv          # Layer/node/activation configs + accuracy results
│
├── placementxai/                   # Django project root
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── users/                          # Django app: auth + profile management
│   ├── models.py                   # Student profile model
│   ├── views.py                    # Login, signup, update, delete
│   ├── forms.py
│   ├── urls.py
│   └── templates/
│       └── users/
│           ├── login.html
│           ├── signup.html
│           └── profile.html
│
├── predictor/                      # Django app: prediction + SHAP dashboard
│   ├── views.py                    # Loads model, runs inference, computes SHAP
│   ├── urls.py
│   └── templates/
│       └── predictor/
│           ├── dashboard.html      # Main analysis page with pie chart
│           └── result.html
│
├── static/
│   ├── css/
│   └── js/
│
├── db.sqlite3
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/tiya-513/PlacementXAI
cd PlacementXAI

pip install -r requirements.txt

# Train the model first (or skip if my_model.keras is already present)
python model/train.py

# Run migrations and start server
python manage.py migrate
python manage.py runserver
```

### requirements.txt includes
```
django
tensorflow
shap
numpy
pandas
scikit-learn
matplotlib
```

---

## Model

Trained on the [Student Placement Prediction Dataset](https://www.kaggle.com/datasets/suhanigupta04/student-placement-prediction-dataset) from Kaggle.

- 80/20 train-test split
- Architecture search across layer counts (1–4), node widths (32–256), and activation functions (ReLU, tanh, sigmoid)
- Best config: 3 hidden layers — 128 → 64 → 32 nodes, ReLU, dropout 0.3
- Test accuracy: **91.4%**
- All experiment configs logged to `model/experiment_log.csv`

---

## Explainability (SHAP)

- **Global SHAP**: run on the full dataset to show which features matter most overall
- **Local SHAP**: computed per student at prediction time, converted to non-negative percentages
- Displayed as a ranked pie chart on the dashboard

---

## Features

- [x] User signup / login / logout
- [x] Profile page with editable student parameters
- [x] Password hashing (bcrypt)
- [x] Account deletion with password confirmation
- [x] Placement prediction on demand
- [x] Per-student SHAP pie chart

---

## Dataset

[Kaggle — Student Placement Prediction Dataset](https://www.kaggle.com/datasets/suhanigupta04/student-placement-prediction-dataset)

---

*Built as part of an ML + XAI learning project.*
