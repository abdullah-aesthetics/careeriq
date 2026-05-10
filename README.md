# 🧠 CareerIQ — Smart Career & Course Recommendation System

> An AI-powered career guidance system built with Python (Flask + scikit-learn Decision Tree).
> Recommends the best career path based on marks, interests, personality, and goals.

---

## 📁 Project Structure

```
careeriq/
│
├── app.py                        ← Flask backend (main entry point)
├── requirements.txt              ← Python dependencies
│
├── model/
│   ├── decision_tree_model.py    ← Decision Tree ML model (train + predict)
│   └── trained_model.pkl         ← Auto-generated after first run
│
├── data/
│   └── career_dataset.csv        ← Training dataset (50+ student records)
│
├── utils/
│   └── cli.py                    ← CLI tool (train / predict / test)
│
├── templates/
│   └── index.html                ← Frontend (served by Flask)
│
└── static/
    ├── css/style.css
    └── js/app.js
```

---

## ⚙️ Setup & Installation

### 1. Clone / Download
```bash
cd careeriq
```

### 2. Create Virtual Environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Flask Server
```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## 🖥️ API Endpoints

| Method | Endpoint              | Description                        |
|--------|-----------------------|------------------------------------|
| POST   | `/api/predict`        | Main career prediction              |
| POST   | `/api/feedback`       | Submit yes/no feedback              |
| GET    | `/api/careers`        | List all supported career paths     |
| GET    | `/api/career/<key>`   | Full detail for one career          |
| GET    | `/api/model/info`     | Model accuracy & feature importance |
| POST   | `/api/model/retrain`  | Retrain the decision tree           |
| GET    | `/api/history`        | Last 50 predictions (from DB)       |
| GET    | `/api/stats`          | Dashboard statistics                |
| GET    | `/api/health`         | Health check                        |

---

## 📬 API Usage Example

### Request
```http
POST /api/predict
Content-Type: application/json

{
  "marks": {
    "math": 88,
    "eng":  70,
    "sci":  80,
    "cs":   90,
    "biz":  50,
    "art":  35
  },
  "interests":   ["coding", "data"],
  "personality": "introvert",
  "goals":       ["salary", "remote", "startup"]
}
```

### Response
```json
{
  "success": true,
  "data": {
    "career":      "software",
    "title":       "Software Engineering",
    "emoji":       "💻",
    "confidence":  91.2,
    "avg_salary":  "$95,000 – $180,000",
    "growth":      "Very High (25% by 2030)",
    "skills":      ["Python", "JavaScript", "Data Structures", ...],
    "courses":     [{"name": "CS50x", "platform": "Harvard / edX"}, ...],
    "roadmap":     [{"phase": "0–6 months", "title": "Core Programming", ...}],
    "job_roles":   ["Software Developer", "Backend Engineer", ...],
    "alternatives": [
      {"career": "datascience", "confidence": 68.5},
      {"career": "finance",     "confidence": 12.1}
    ],
    "decision_path": [
      {"rule": "CS score (90.0) > 72.5",           "direction": "NO"},
      {"rule": "Has interest in coding",            "direction": "NO"},
      {"rule": "→ Recommend: Software Engineering", "direction": "LEAF"}
    ],
    "prediction_id": 7
  }
}
```

---

## 🛠️ CLI Tool

```bash
# Train model fresh
python utils/cli.py train

# Interactive career predictor in terminal
python utils/cli.py predict

# Print decision tree structure
python utils/cli.py tree

# Show feature importances
python utils/cli.py importance

# Run 4 sample test predictions
python utils/cli.py test
```

---

## 🌳 How the Decision Tree Works

The Decision Tree works by splitting student data at each node using the best feature
that maximises information gain (Gini impurity). It evaluates multiple conditions:

```
IF CS score > 80
  AND interest_coding = 1
  AND personality = introvert
    → Software Engineering

ELSE IF Math score > 80
  AND interest_data = 1
    → Data Science

ELSE IF arts score > 75
  AND interest_design = 1
    → UI/UX Design
...
```

**Key insight**: High marks ≠ forced career. The model balances marks + interests + personality
to mimic a real human career counsellor.

---

## 🗄️ Database

SQLite is used automatically (no setup needed). Tables:

- `predictions` — stores all student form submissions + recommended career
- `feedback` — stores yes/no feedback per prediction

Database file: `instance/careeriq.db` (auto-created on first run)

---

## 📊 Supported Careers

| Key          | Career                     |
|--------------|----------------------------|
| software     | Software Engineering        |
| datascience  | Data Science & AI           |
| uiux         | UI/UX Design                |
| business     | Business & Entrepreneurship |
| medicine     | Medicine & Healthcare       |
| finance      | Finance & Investment        |

---

## 🔧 Tech Stack

| Layer      | Technology                            |
|------------|---------------------------------------|
| Backend    | Python 3.11+, Flask 3.0               |
| ML Model   | scikit-learn DecisionTreeClassifier   |
| Database   | SQLite via Flask-SQLAlchemy           |
| Frontend   | HTML5, CSS3, JavaScript, Chart.js     |
| CORS       | Flask-CORS                            |

---

## 💡 Viva Talking Points

1. **Why Decision Tree?** — It mimics human decision-making with IF-THEN rules.
   Easy to explain, visualise, and audit — unlike neural networks.

2. **Marks + Interest balance** — A student with 90% Math but zero interest in Math
   is NOT recommended a pure Math career. The model weighs both ability and passion.

3. **Gini Impurity** — The tree splits on features that create the most homogeneous groups,
   minimising disorder in the resulting career labels.

4. **Overfitting prevention** — `max_depth=10` prevents the tree from memorising
   training data while keeping it deep enough for nuanced recommendations.

5. **Feedback loop** — Positive/negative feedback is stored in SQLite.
   Future retraining uses this data to improve accuracy over time.

---

## 👨‍💻 Author

Built as a Python subject project demonstrating:
- Machine Learning (Decision Tree Classifier)
- REST API design (Flask)
- Database integration (SQLite)
- Full-stack integration (HTML frontend + Python backend)
