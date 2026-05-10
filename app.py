"""
CareerIQ — Flask Backend
=========================
Run:  python app.py
Open: http://localhost:5000

Every form submission is saved to:
  → careeriq.db        (SQLite database)
  → submissions.csv    (CSV file, opens in Excel)
"""

import os
import csv
import json
import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pdf_generator import generate_career_pdf
from flask import send_file

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CSV_FILE    = os.path.join(BASE_DIR, 'submissions.csv')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# ─── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)

app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', 'careeriq-dev-secret-2024'),
    JSON_SORT_KEYS=False,
    JSONIFY_PRETTYPRINT_REGULAR=True,
)

# ─── Database ─────────────────────────────────────────────────────────────────
from flask_sqlalchemy import SQLAlchemy

app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(BASE_DIR, "careeriq.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Prediction(db.Model):
    __tablename__ = 'predictions'
    id          = db.Column(db.Integer, primary_key=True)
    timestamp   = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    career      = db.Column(db.String(50))
    confidence  = db.Column(db.Float)
    marks_json  = db.Column(db.Text)
    interests   = db.Column(db.Text)
    personality = db.Column(db.String(30))
    goals       = db.Column(db.Text)
    feedback    = db.Column(db.String(10), nullable=True)

    def to_dict(self):
        return {
            'id':          self.id,
            'timestamp':   self.timestamp.isoformat(),
            'career':      self.career,
            'confidence':  self.confidence,
            'marks':       json.loads(self.marks_json),
            'interests':   json.loads(self.interests),
            'personality': self.personality,
            'goals':       json.loads(self.goals),
            'feedback':    self.feedback,
        }


class Feedback(db.Model):
    __tablename__ = 'feedback'
    id            = db.Column(db.Integer, primary_key=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey('predictions.id'))
    rating        = db.Column(db.String(10))
    timestamp     = db.Column(db.DateTime, default=datetime.datetime.utcnow)


# ─── CSV Helper ───────────────────────────────────────────────────────────────
# These are the column headers in submissions.csv
CSV_HEADERS = [
    'submission_id',
    'timestamp',
    'math', 'english', 'science', 'cs', 'business', 'arts',
    'interests',
    'personality',
    'goals',
    'recommended_career',
    'career_title',
    'confidence_%',
    'avg_salary',
    'growth',
    'feedback',
]

def init_csv():
    """Create submissions.csv with headers if it doesn't exist yet."""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
        print(f"✅ Created submissions.csv at: {CSV_FILE}")


def save_to_csv(prediction_id, data, result):
    """Append one row to submissions.csv for every new form submission."""
    marks = data.get('marks', {})
    row = {
        'submission_id':      prediction_id,
        'timestamp':          datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'math':               marks.get('math', ''),
        'english':            marks.get('eng',  ''),
        'science':            marks.get('sci',  ''),
        'cs':                 marks.get('cs',   ''),
        'business':           marks.get('biz',  ''),
        'arts':               marks.get('art',  ''),
        'interests':          ', '.join(data.get('interests', [])),
        'personality':        data.get('personality', ''),
        'goals':              ', '.join(data.get('goals', [])),
        'recommended_career': result.get('career', ''),
        'career_title':       result.get('title',  ''),
        'confidence_%':       result.get('confidence', ''),
        'avg_salary':         result.get('avg_salary', ''),
        'growth':             result.get('growth', ''),
        'feedback':           '',   # filled later when user clicks 👍/👎
    }
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writerow(row)
    print(f"📝 Saved submission #{prediction_id} to submissions.csv")


def update_csv_feedback(prediction_id, rating):
    """Update the feedback column for a specific submission_id in the CSV."""
    rows = []
    try:
        with open(CSV_FILE, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get('submission_id')) == str(prediction_id):
                    row['feedback'] = rating
                rows.append(row)

        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"✅ Updated feedback for submission #{prediction_id} in CSV")
    except Exception as e:
        print(f"⚠️  Could not update CSV feedback: {e}")


# ─── Load ML model ────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, BASE_DIR)
from decision_tree_model import get_model, CAREER_META, CareerDecisionTree

model = None

def get_trained_model():
    global model
    if model is None:
        model = get_model()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'project.html')


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        result = model.predict(data)

        # ── Save to DB ──
        prediction = Prediction(
            career      = result['career'],
            confidence  = result['confidence'],
            marks_json  = json.dumps(data.get('marks', {})),
            interests   = json.dumps(data.get('interests', [])),
            personality = data.get('personality', ''),
            goals       = json.dumps(data.get('goals', [])),
        )

        db.session.add(prediction)
        db.session.commit()

        # ── CSV ──
        save_to_csv(prediction.id, data, result)

        # ── PDF ──
        pdf_path = os.path.join(REPORTS_DIR, f'career_report_{prediction.id}.pdf')
        generate_career_pdf(result, data, pdf_path)

        print(f"📄 PDF saved: {pdf_path}")

        # attach ID
        result['prediction_id'] = prediction.id
        result['pdf_url'] = f"/api/download-pdf/{prediction.id}"

        return jsonify({
            "success": True,
            "data": result
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """
    POST /api/feedback
    Saves rating to SQLite AND updates the CSV row too.
    """
    try:
        data          = request.get_json(force=True)
        prediction_id = data.get('prediction_id')
        rating        = data.get('rating')

        if rating not in ('positive', 'negative'):
            return jsonify({'success': False, 'error': 'rating must be positive or negative'}), 400

        pred = db.session.get(Prediction, prediction_id)
        if not pred:
            return jsonify({'success': False, 'error': 'Prediction not found'}), 404

        # Update SQLite
        pred.feedback = rating
        fb = Feedback(prediction_id=prediction_id, rating=rating)
        db.session.add(fb)
        db.session.commit()

        # Update CSV row for this submission
        update_csv_feedback(prediction_id, rating)

        return jsonify({'success': True, 'message': 'Feedback recorded. Thank you!'}), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/download-csv', methods=['GET'])
def download_csv():
    """GET /api/download-csv — Download submissions.csv directly from browser."""
    from flask import send_file
    if not os.path.exists(CSV_FILE):
        return jsonify({'success': False, 'error': 'No submissions yet'}), 404
    return send_file(
        CSV_FILE,
        mimetype='text/csv',
        as_attachment=True,
        download_name='careeriq_submissions.csv',
    )


@app.route('/api/submissions', methods=['GET'])
def get_submissions():
    """GET /api/submissions — Return all CSV rows as JSON."""
    if not os.path.exists(CSV_FILE):
        return jsonify({'success': True, 'count': 0, 'submissions': []})
    rows = []
    with open(CSV_FILE, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return jsonify({'success': True, 'count': len(rows), 'submissions': rows})


@app.route('/api/careers', methods=['GET'])
def list_careers():
    return jsonify({
        'success': True,
        'count': len(CAREER_META),
        'careers': {
            key: {
                'title':      v['title'],
                'emoji':      v['emoji'],
                'description':v['description'],
                'avg_salary': v['avg_salary'],
                'growth':     v['growth'],
                'job_roles':  v['job_roles'],
            }
            for key, v in CAREER_META.items()
        }
    })


@app.route('/api/career/<career_key>', methods=['GET'])
def career_detail(career_key):
    if career_key not in CAREER_META:
        return jsonify({'success': False, 'error': f'Career "{career_key}" not found'}), 404
    return jsonify({'success': True, 'data': CAREER_META[career_key]})


@app.route('/api/model/info', methods=['GET'])
def model_info():
    try:
        m          = get_trained_model()
        importance = m.get_feature_importance()
        tree_text  = m.get_tree_text()
        return jsonify({
            'success':      True,
            'model_type':   'DecisionTreeClassifier',
            'features':     len(m.feature_names),
            'classes':      list(m.encoder.classes_),
            'top_features': importance[:10],
            'tree_preview': tree_text[:2000],
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/model/retrain', methods=['POST'])
def retrain_model():
    try:
        global model
        m      = CareerDecisionTree()
        result = m.train()
        model  = m
        return jsonify({'success': True, 'training_result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def prediction_history():
    predictions = (
        Prediction.query
        .order_by(Prediction.timestamp.desc())
        .limit(50).all()
    )
    return jsonify({
        'success':     True,
        'count':       len(predictions),
        'predictions': [p.to_dict() for p in predictions],
    })


@app.route('/api/stats', methods=['GET'])
def stats():
    total       = Prediction.query.count()
    positive_fb = Feedback.query.filter_by(rating='positive').count()
    negative_fb = Feedback.query.filter_by(rating='negative').count()
    career_counts = (
        db.session.query(Prediction.career, db.func.count(Prediction.id))
        .group_by(Prediction.career).all()
    )
    return jsonify({
        'success':           True,
        'total_predictions': total,
        'feedback': {
            'positive':          positive_fb,
            'negative':          negative_fb,
            'satisfaction_rate': (
                round(positive_fb / (positive_fb + negative_fb) * 100, 1)
                if (positive_fb + negative_fb) > 0 else 0
            ),
        },
        'career_distribution': {c: n for c, n in career_counts},
    })


@app.route('/api/health', methods=['GET'])
def health():
    csv_exists   = os.path.exists(CSV_FILE)
    csv_rows     = 0
    if csv_exists:
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            csv_rows = sum(1 for _ in f) - 1  # minus header
    return jsonify({
        'status':          'ok',
        'service':         'CareerIQ API',
        'version':         '1.0.0',
        'timestamp':       datetime.datetime.utcnow().isoformat(),
        'csv_file':        'submissions.csv',
        'csv_submissions': csv_rows,
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_input(data: dict) -> list:
    errors = []
    if not isinstance(data, dict):
        return ['Request body must be a JSON object']
    marks = data.get('marks', {})
    for key in ['math', 'eng', 'sci', 'cs', 'biz', 'art']:
        val = marks.get(key)
        if val is None:
            errors.append(f'marks.{key} is required')
        elif not (0 <= float(val) <= 100):
            errors.append(f'marks.{key} must be between 0 and 100')
    if not data.get('personality'):
        errors.append('personality is required')
    elif data['personality'] not in ('introvert', 'extrovert', 'creative', 'analytical'):
        errors.append('personality must be: introvert / extrovert / creative / analytical')
    if not isinstance(data.get('interests', []), list):
        errors.append('interests must be an array')
    if not isinstance(data.get('goals', []), list):
        errors.append('goals must be an array')
    return errors

@app.route('/api/download-pdf/<int:pred_id>', methods=['GET'])
def download_pdf(pred_id):
    from flask import send_file
    pdf_path = os.path.join(REPORTS_DIR, f'career_report_{pred_id}.pdf')
    if not os.path.exists(pdf_path):
        return jsonify({'success': False, 'error': 'PDF not found'}), 404
    return send_file(pdf_path, mimetype='application/pdf',
                     as_attachment=True,
                     download_name=f'CareerIQ_Report_{pred_id}.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("✅ SQLite database ready (careeriq.db)")
        init_csv()
        get_trained_model()

    print("\n" + "="*52)
    print("🚀 CareerIQ is running!")
    print("   Open:      http://localhost:5000")
    print("   CSV file:  submissions.csv  (auto-saved)")
    print("   Download:  http://localhost:5000/api/download-csv")
    print("="*52 + "\n")
port = int(os.environ.get('PORT', 5000))
app.run(debug=False, host='0.0.0.0', port=port)
