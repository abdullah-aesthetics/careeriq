"""
CareerIQ — Decision Tree Model
================================
Trains a scikit-learn DecisionTreeClassifier on the career dataset,
exposes prediction, confidence scores, and tree-path explanation.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'career_dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'trained_model.pkl')

# ─── Feature columns (must match CSV header & API input keys) ─────────────────
FEATURE_COLS = [
    'math', 'english', 'science', 'cs', 'business', 'arts',
    'interest_coding', 'interest_data', 'interest_design',
    'interest_business', 'interest_finance', 'interest_medicine',
    'interest_science', 'interest_art', 'interest_engineering',
    'interest_marketing',
    'personality_introvert', 'personality_extrovert',
    'personality_creative', 'personality_analytical',
    'goal_salary', 'goal_impact', 'goal_startup', 'goal_stability',
    'goal_remote', 'goal_creative_freedom', 'goal_leadership',
]

TARGET_COL = 'career'

# ─── Career metadata ──────────────────────────────────────────────────────────
CAREER_META = {
    'software': {
        'title': 'Software Engineering',
        'emoji': '💻',
        'description': 'Build apps, systems, and platforms that power billions of lives.',
        'avg_salary': '$95,000 – $180,000',
        'growth': 'Very High (25% by 2030)',
        'skills': ['Python', 'JavaScript', 'Data Structures', 'System Design',
                   'Git', 'Cloud (AWS/GCP)', 'API Design', 'Docker'],
        'courses': [
            {'name': 'CS50x — Intro to Computer Science', 'platform': 'Harvard / edX'},
            {'name': 'Full-Stack Web Development',         'platform': 'The Odin Project'},
            {'name': 'Data Structures & Algorithms',       'platform': 'Coursera'},
            {'name': 'System Design Fundamentals',         'platform': 'Educative.io'},
            {'name': 'AWS Cloud Practitioner',             'platform': 'AWS Training'},
        ],
        'roadmap': [
            {'phase': '0–6 months',  'title': 'Core Programming',   'desc': 'Python/JS, Git, problem solving basics'},
            {'phase': '6–18 months', 'title': 'Projects + Internship', 'desc': '3 real projects, open-source contributions'},
            {'phase': '1–3 years',   'title': 'Junior Developer',   'desc': 'First job, specialise in backend/frontend'},
            {'phase': '3–5 years',   'title': 'Senior Engineer',    'desc': 'Architect systems, tech lead, mentor juniors'},
        ],
        'job_roles': ['Software Developer', 'Backend Engineer', 'Frontend Engineer',
                      'Mobile Developer', 'DevOps Engineer', 'Full-Stack Developer'],
    },
    'datascience': {
        'title': 'Data Science & AI',
        'emoji': '📊',
        'description': 'Extract insight from data — the most in-demand skill of the century.',
        'avg_salary': '$90,000 – $165,000',
        'growth': 'Extremely High (36% by 2031)',
        'skills': ['Python', 'Statistics', 'Machine Learning', 'SQL',
                   'TensorFlow/PyTorch', 'Data Visualisation', 'Probability'],
        'courses': [
            {'name': 'Data Science Specialization',  'platform': 'Coursera / Johns Hopkins'},
            {'name': 'Applied Machine Learning',     'platform': 'Fast.ai'},
            {'name': 'SQL for Data Analysis',        'platform': 'Mode Analytics'},
            {'name': 'Deep Learning Specialization', 'platform': 'deeplearning.ai'},
        ],
        'roadmap': [
            {'phase': '0–6 months',  'title': 'Python + Statistics',  'desc': 'NumPy, Pandas, probability & stats'},
            {'phase': '6–12 months', 'title': 'ML Models',            'desc': 'Regression, classification, Kaggle'},
            {'phase': '1–2 years',   'title': 'Data Analyst / MLE',   'desc': 'First industry role'},
            {'phase': '3+ years',    'title': 'Senior Data Scientist', 'desc': 'Deep learning, research, strategy'},
        ],
        'job_roles': ['Data Scientist', 'ML Engineer', 'Data Analyst',
                      'AI Researcher', 'Business Intelligence Analyst'],
    },
    'uiux': {
        'title': 'UI/UX Design',
        'emoji': '🎨',
        'description': 'Shape how the world interacts with technology through beautiful design.',
        'avg_salary': '$75,000 – $140,000',
        'growth': 'High (13% by 2030)',
        'skills': ['Figma', 'User Research', 'Prototyping', 'Design Systems',
                   'CSS', 'Psychology', 'Accessibility', 'Adobe XD'],
        'courses': [
            {'name': 'Google UX Design Certificate', 'platform': 'Coursera'},
            {'name': 'Figma Complete Course',         'platform': 'Udemy'},
            {'name': 'Interaction Design Foundation', 'platform': 'IDF.org'},
            {'name': 'UI Design Bootcamp',            'platform': 'Scrimba'},
        ],
        'roadmap': [
            {'phase': '0–3 months',  'title': 'Design Fundamentals',  'desc': 'Color, typography, layout, Figma'},
            {'phase': '3–9 months',  'title': 'Portfolio (5 Projects)', 'desc': 'Real UX case studies with user research'},
            {'phase': '9–18 months', 'title': 'Junior Designer',       'desc': 'Agency or product company role'},
            {'phase': '3+ years',    'title': 'Lead / Head of Design', 'desc': 'Design systems, team leadership'},
        ],
        'job_roles': ['UX Designer', 'UI Designer', 'Product Designer',
                      'UX Researcher', 'Interaction Designer', 'Design Lead'],
    },
    'business': {
        'title': 'Business & Entrepreneurship',
        'emoji': '💼',
        'description': 'Lead organisations, build ventures, and create economic value at scale.',
        'avg_salary': '$70,000 – $200,000+',
        'growth': 'High (steady demand across all industries)',
        'skills': ['Leadership', 'Financial Modeling', 'Strategy', 'Marketing',
                   'Negotiation', 'Excel / Power BI', 'Communication'],
        'courses': [
            {'name': 'Business Foundations Specialization', 'platform': 'Coursera / Wharton'},
            {'name': 'Financial Accounting Basics',         'platform': 'edX / MIT'},
            {'name': 'Digital Marketing Fundamentals',      'platform': 'Google Skillshop'},
            {'name': 'Entrepreneurship Specialization',     'platform': 'Coursera'},
        ],
        'roadmap': [
            {'phase': '0–6 months',  'title': 'Business Basics',       'desc': 'Accounting, marketing, strategy'},
            {'phase': '6–18 months', 'title': 'Internship + Side Project', 'desc': 'Real business exposure'},
            {'phase': '2–4 years',   'title': 'Management Role / MBA', 'desc': 'Team management, cross-functional work'},
            {'phase': '5+ years',    'title': 'Director / Founder',    'desc': 'Lead company or launch own venture'},
        ],
        'job_roles': ['Business Analyst', 'Product Manager', 'Marketing Manager',
                      'Entrepreneur', 'Consultant', 'Operations Manager'],
    },
    'medicine': {
        'title': 'Medicine & Healthcare',
        'emoji': '🏥',
        'description': 'The most noble profession — heal, innovate, and protect human life.',
        'avg_salary': '$100,000 – $350,000',
        'growth': 'Very High (13% by 2031)',
        'skills': ['Biology', 'Chemistry', 'Clinical Skills', 'Research',
                   'Empathy', 'Problem Solving', 'Communication'],
        'courses': [
            {'name': 'Human Anatomy & Physiology', 'platform': 'Khan Academy'},
            {'name': 'Medical Ethics',             'platform': 'Coursera / Yale'},
            {'name': 'Research Methods in Health', 'platform': 'edX'},
            {'name': 'USMLE Step 1 Prep',          'platform': 'Amboss / Anki'},
        ],
        'roadmap': [
            {'phase': '0–5 years',   'title': 'Medical School (MBBS)',  'desc': 'Pre-clinical & clinical training'},
            {'phase': '5–8 years',   'title': 'House Officer + Residency', 'desc': 'Specialisation training'},
            {'phase': '8–12 years',  'title': 'Specialist Doctor',      'desc': 'Independent clinical practice'},
            {'phase': '12+ years',   'title': 'Consultant / Professor', 'desc': 'Research, teaching, hospital leadership'},
        ],
        'job_roles': ['General Physician', 'Surgeon', 'Psychiatrist',
                      'Paediatrician', 'Medical Researcher', 'Healthcare Administrator'],
    },
    'finance': {
        'title': 'Finance & Investment',
        'emoji': '💰',
        'description': 'Master the language of money — markets, valuation, and capital allocation.',
        'avg_salary': '$80,000 – $200,000',
        'growth': 'High (CFA demand globally)',
        'skills': ['Financial Analysis', 'Excel', 'Valuation', 'Risk Management',
                   'CFA Prep', 'Python for Finance', 'Bloomberg'],
        'courses': [
            {'name': 'Financial Markets by Robert Shiller', 'platform': 'Coursera / Yale'},
            {'name': 'CFA Level 1 Preparation',             'platform': 'Schweser / Kaplan'},
            {'name': 'Python for Financial Analysis',        'platform': 'Udemy'},
            {'name': 'Investment Banking Fundamentals',      'platform': 'Wall Street Prep'},
        ],
        'roadmap': [
            {'phase': '0–6 months',  'title': 'Finance Fundamentals',  'desc': 'Accounting, valuation, Excel modeling'},
            {'phase': '6–18 months', 'title': 'Internship + CFA L1',   'desc': 'Bank / fund internship'},
            {'phase': '2–5 years',   'title': 'Financial Analyst',     'desc': 'Investment banking, equity research'},
            {'phase': '5+ years',    'title': 'Portfolio Manager / CFO', 'desc': 'Managing funds or corporate finance'},
        ],
        'job_roles': ['Financial Analyst', 'Investment Banker', 'Portfolio Manager',
                      'Risk Analyst', 'CFO', 'Equity Research Analyst'],
    },
}


# ─── Model class ──────────────────────────────────────────────────────────────
class CareerDecisionTree:
    """Wraps a scikit-learn DecisionTreeClassifier with convenience methods."""

    def __init__(self):
        self.model   = None
        self.encoder = LabelEncoder()
        self.feature_names = FEATURE_COLS
        self.is_trained = False

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, data_path: str = DATA_PATH) -> dict:
        df = pd.read_csv(data_path)

        X = df[FEATURE_COLS].values
        y = self.encoder.fit_transform(df[TARGET_COL].values)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            criterion='gini',
            random_state=42,
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True

        y_pred    = self.model.predict(X_test)
        accuracy  = accuracy_score(y_test, y_pred)
        report    = classification_report(
            y_test, y_pred,
            target_names=self.encoder.classes_,
            output_dict=True,
        )

        self._save()
        return {'accuracy': round(accuracy * 100, 2), 'report': report,
                'classes': list(self.encoder.classes_)}

    # ── Persistence ───────────────────────────────────────────────────────────
    def _save(self):
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump({'model': self.model, 'encoder': self.encoder}, f)

    def load(self) -> bool:
        if not os.path.exists(MODEL_PATH):
            return False
        with open(MODEL_PATH, 'rb') as f:
            data = pickle.load(f)
        self.model      = data['model']
        self.encoder    = data['encoder']
        self.is_trained = True
        return True

    # ── Feature vector builder ─────────────────────────────────────────────
    @staticmethod
    def build_feature_vector(form_data: dict) -> np.ndarray:
        """Convert API request JSON → numpy feature vector."""
        marks = form_data.get('marks', {})
        vec = {
            'math':    float(marks.get('math',    0)),
            'english': float(marks.get('eng',     0)),
            'science': float(marks.get('sci',     0)),
            'cs':      float(marks.get('cs',      0)),
            'business':float(marks.get('biz',     0)),
            'arts':    float(marks.get('art',     0)),
        }

        all_interests = [
            'coding','data','design','business','finance','medicine',
            'science','art','engineering','marketing',
        ]
        for i in all_interests:
            vec[f'interest_{i}'] = 1 if i in form_data.get('interests', []) else 0

        personality = form_data.get('personality', '')
        for p in ['introvert','extrovert','creative','analytical']:
            vec[f'personality_{p}'] = 1 if personality == p else 0

        all_goals = [
            'salary','impact','startup','stability',
            'remote','creative_freedom','leadership',
        ]
        for g in all_goals:
            vec[f'goal_{g}'] = 1 if g in form_data.get('goals', []) else 0

        return np.array([[vec[col] for col in FEATURE_COLS]])

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict(self, form_data: dict) -> dict:
        """Return full prediction result including confidence & decision path."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        X = self.build_feature_vector(form_data)

        # Class probabilities
        proba   = self.model.predict_proba(X)[0]
        classes = self.encoder.classes_

        # Sort by confidence
        ranked = sorted(
            zip(classes, proba), key=lambda x: x[1], reverse=True
        )

        top_career = ranked[0][0]
        confidence = round(float(ranked[0][1]) * 100, 1)

        # Decision path
        decision_path = self._get_decision_path(X)

        # Alternative careers (top 3 after primary)
        alternatives = [
            {'career': c, 'confidence': round(float(p) * 100, 1)}
            for c, p in ranked[1:4] if p > 0
        ]

        meta = CAREER_META.get(top_career, {})

        return {
            'career':        top_career,
            'title':         meta.get('title', top_career.title()),
            'emoji':         meta.get('emoji', '🎯'),
            'description':   meta.get('description', ''),
            'confidence':    confidence,
            'avg_salary':    meta.get('avg_salary', 'N/A'),
            'growth':        meta.get('growth', 'N/A'),
            'skills':        meta.get('skills', []),
            'courses':       meta.get('courses', []),
            'roadmap':       meta.get('roadmap', []),
            'job_roles':     meta.get('job_roles', []),
            'alternatives':  alternatives,
            'decision_path': decision_path,
            'all_scores': [
                {'career': c, 'score': round(float(p) * 100, 1)}
                for c, p in ranked
            ],
        }

    # ── Decision path extraction ──────────────────────────────────────────────
    def _get_decision_path(self, X: np.ndarray) -> list:
        """Walk the decision tree and return human-readable rule steps."""
        tree      = self.model.tree_
        feature   = tree.feature
        threshold = tree.threshold
        node_id   = 0
        path      = []

        while tree.children_left[node_id] != -1:  # not a leaf
            feat_idx  = feature[node_id]
            feat_name = FEATURE_COLS[feat_idx]
            thresh    = threshold[node_id]
            val       = float(X[0, feat_idx])

            direction = 'YES' if val <= thresh else 'NO'
            readable  = self._readable_rule(feat_name, thresh, val)
            path.append({'rule': readable, 'direction': direction, 'node': int(node_id)})

            if val <= thresh:
                node_id = tree.children_left[node_id]
            else:
                node_id = tree.children_right[node_id]

        # Leaf node — predicted class
        class_idx = int(np.argmax(tree.value[node_id]))
        leaf_class = self.encoder.inverse_transform([class_idx])[0]
        path.append({'rule': f'→ Recommend: {CAREER_META[leaf_class]["title"]}',
                     'direction': 'LEAF', 'node': int(node_id)})
        return path

    @staticmethod
    def _readable_rule(feature_name: str, threshold: float, value: float) -> str:
        thresh_r = round(threshold, 1)
        val_r    = round(value, 1)
        if feature_name in ('math','english','science','cs','business','arts'):
            label = feature_name.title()
            op    = '≤' if value <= threshold else '>'
            return f'{label} score ({val_r}) {op} {thresh_r}'
        if feature_name.startswith('interest_'):
            interest = feature_name.replace('interest_', '').title()
            has_it   = 'No interest' if value <= threshold else 'Has interest'
            return f'{has_it} in {interest}'
        if feature_name.startswith('personality_'):
            ptype = feature_name.replace('personality_', '').title()
            is_it = 'Not' if value <= threshold else 'Is'
            return f'{is_it} {ptype}'
        if feature_name.startswith('goal_'):
            goal  = feature_name.replace('goal_', '').replace('_', ' ').title()
            wants = 'Does not prioritise' if value <= threshold else 'Prioritises'
            return f'{wants} {goal}'
        return f'{feature_name} ≤ {thresh_r}'

    # ── Model info ────────────────────────────────────────────────────────────
    def get_tree_text(self) -> str:
        """Return ASCII text representation of the full decision tree."""
        if not self.is_trained:
            return "Model not trained."
        return export_text(self.model, feature_names=FEATURE_COLS, max_depth=5)

    def get_feature_importance(self) -> list:
        """Return features sorted by importance score."""
        if not self.is_trained:
            return []
        importance = self.model.feature_importances_
        return sorted(
            [{'feature': FEATURE_COLS[i], 'importance': round(float(v), 4)}
             for i, v in enumerate(importance) if v > 0],
            key=lambda x: x['importance'], reverse=True,
        )


# ─── Singleton ────────────────────────────────────────────────────────────────
_model_instance: CareerDecisionTree | None = None


def get_model() -> CareerDecisionTree:
    """Return a trained model (loads from disk or trains fresh)."""
    global _model_instance
    if _model_instance is None:
        _model_instance = CareerDecisionTree()
        if not _model_instance.load():
            print("[CareerIQ] No saved model found — training fresh model...")
            result = _model_instance.train()
            print(f"[CareerIQ] Training complete. Accuracy: {result['accuracy']}%")
        else:
            print("[CareerIQ] Loaded saved model from disk.")
    return _model_instance
