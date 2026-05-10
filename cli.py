"""
CareerIQ — CLI Utility
=======================
Usage:
    python utils/cli.py train              → Train & save model
    python utils/cli.py predict            → Interactive career prediction
    python utils/cli.py tree               → Print decision tree structure
    python utils/cli.py importance         → Show feature importances
    python utils/cli.py test               → Run sample predictions
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.decision_tree_model import CareerDecisionTree, CAREER_META, get_model


# ─── ANSI colors ──────────────────────────────────────────────────────────────
BOLD   = '\033[1m'
GREEN  = '\033[92m'
CYAN   = '\033[96m'
YELLOW = '\033[93m'
MAGENTA= '\033[95m'
RED    = '\033[91m'
RESET  = '\033[0m'
DIM    = '\033[2m'


def banner():
    print(f"""
{CYAN}{BOLD}
  ██████╗ █████╗ ██████╗ ███████╗███████╗██████╗    ██╗ ██████╗
 ██╔════╝██╔══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗  ██╔╝██╔═══██╗
 ██║     ███████║██████╔╝█████╗  █████╗  ██████╔╝ ██╔╝ ██║   ██║
 ██║     ██╔══██║██╔══██╗██╔══╝  ██╔══╝  ██╔══██╗██╔╝  ██║▄▄ ██║
 ╚██████╗██║  ██║██║  ██║███████╗███████╗██║  ██║██╔╝   ╚██████╔╝
  ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚══▀▀═╝
{RESET}
{DIM}  Smart Career & Course Recommendation System — Decision Tree AI{RESET}
""")


def cmd_train():
    print(f"\n{CYAN}[Training]{RESET} Loading dataset & training DecisionTreeClassifier...\n")
    m = CareerDecisionTree()
    result = m.train()
    print(f"{GREEN}✅ Training complete!{RESET}")
    print(f"   Accuracy  : {BOLD}{result['accuracy']}%{RESET}")
    print(f"   Classes   : {', '.join(result['classes'])}")
    print(f"\n{DIM}Per-class report:{RESET}")
    for cls, metrics in result['report'].items():
        if isinstance(metrics, dict):
            print(f"  {cls:<15} precision={metrics['precision']:.2f}  recall={metrics['recall']:.2f}  f1={metrics['f1-score']:.2f}")
    print(f"\n{GREEN}Model saved to model/trained_model.pkl{RESET}\n")


def cmd_predict():
    banner()
    print(f"{BOLD}🎓 Interactive Career Predictor{RESET}\n")
    print(f"{DIM}Enter your details below (press Enter to keep defaults){RESET}\n")

    def ask_int(prompt, default, lo=0, hi=100):
        while True:
            raw = input(f"  {prompt} [{default}]: ").strip()
            if not raw:
                return default
            try:
                v = int(raw)
                if lo <= v <= hi:
                    return v
                print(f"  {RED}Please enter a value between {lo} and {hi}{RESET}")
            except ValueError:
                print(f"  {RED}Numbers only please{RESET}")

    print(f"{CYAN}─── Academic Marks (0–100) ───{RESET}")
    marks = {
        'math': ask_int('Mathematics',       75),
        'eng':  ask_int('English',           70),
        'sci':  ask_int('Science',           65),
        'cs':   ask_int('Computer Science',  60),
        'biz':  ask_int('Business/Economics',55),
        'art':  ask_int('Arts/Design',       50),
    }

    print(f"\n{CYAN}─── Interests ───{RESET}")
    all_interests = [
        'coding','data','design','business','finance','medicine',
        'science','art','engineering','marketing','writing','social',
        'marketing','gaming','law','cybersecurity',
    ]
    for i, interest in enumerate(all_interests, 1):
        print(f"  {i:>2}. {interest}")
    raw = input("\n  Enter numbers (comma-separated, e.g. 1,3,5): ").strip()
    chosen = []
    for tok in raw.split(','):
        try:
            idx = int(tok.strip()) - 1
            if 0 <= idx < len(all_interests):
                chosen.append(all_interests[idx])
        except ValueError:
            pass
    interests = chosen or ['coding']
    print(f"  Selected: {GREEN}{', '.join(interests)}{RESET}")

    print(f"\n{CYAN}─── Personality ───{RESET}")
    personalities = ['introvert','extrovert','creative','analytical']
    for i, p in enumerate(personalities, 1):
        print(f"  {i}. {p.title()}")
    while True:
        raw = input("  Choose [1-4]: ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < 4:
                personality = personalities[idx]
                break
        except ValueError:
            pass
        print(f"  {RED}Please enter 1–4{RESET}")

    print(f"\n{CYAN}─── Goals ───{RESET}")
    all_goals = ['salary','impact','startup','stability','remote','creative_freedom','leadership']
    for i, g in enumerate(all_goals, 1):
        print(f"  {i}. {g.replace('_',' ').title()}")
    raw = input("  Enter numbers (comma-separated): ").strip()
    goals = []
    for tok in raw.split(','):
        try:
            idx = int(tok.strip()) - 1
            if 0 <= idx < len(all_goals):
                goals.append(all_goals[idx])
        except ValueError:
            pass

    # ── Run prediction ────────────────────────────────────────────────────────
    print(f"\n{CYAN}Analysing profile...{RESET}\n")
    m = get_model()
    result = m.predict({
        'marks': marks, 'interests': interests,
        'personality': personality, 'goals': goals,
    })

    meta = CAREER_META[result['career']]
    print(f"{'─'*55}")
    print(f"  {meta['emoji']}  {BOLD}{GREEN}{result['title']}{RESET}")
    print(f"  Confidence  : {BOLD}{result['confidence']}%{RESET}")
    print(f"  Avg Salary  : {YELLOW}{result['avg_salary']}{RESET}")
    print(f"  Job Growth  : {result['growth']}")
    print(f"{'─'*55}")

    print(f"\n{CYAN}Top Skills to Learn:{RESET}")
    for s in result['skills'][:5]:
        print(f"  • {s}")

    print(f"\n{CYAN}Recommended Courses:{RESET}")
    for c in result['courses'][:3]:
        print(f"  📚 {c['name']}  ({DIM}{c['platform']}{RESET})")

    print(f"\n{CYAN}Career Roadmap:{RESET}")
    for r in result['roadmap']:
        print(f"  [{r['phase']}] {BOLD}{r['title']}{RESET} — {DIM}{r['desc']}{RESET}")

    print(f"\n{CYAN}Alternative Paths:{RESET}")
    for alt in result['alternatives']:
        title = CAREER_META[alt['career']]['title']
        print(f"  ✦ {title} ({alt['confidence']}% match)")

    print(f"\n{CYAN}Decision Tree Path:{RESET}")
    for step in result['decision_path']:
        tag = f"[{step['direction']}]"
        color = GREEN if step['direction'] == 'LEAF' else MAGENTA
        print(f"  {color}{tag}{RESET} {step['rule']}")

    print(f"\n{'─'*55}\n")


def cmd_tree():
    m = get_model()
    print(f"\n{CYAN}Decision Tree Structure (depth ≤ 5):{RESET}\n")
    print(m.get_tree_text())


def cmd_importance():
    m = get_model()
    features = m.get_feature_importance()
    print(f"\n{CYAN}Feature Importances (Top 15):{RESET}\n")
    max_imp = features[0]['importance'] if features else 1
    for f in features[:15]:
        bar_len = int(40 * f['importance'] / max_imp)
        bar = '█' * bar_len
        print(f"  {f['feature']:<35} {GREEN}{bar}{RESET} {f['importance']:.4f}")
    print()


def cmd_test():
    print(f"\n{CYAN}Running sample test predictions...{RESET}\n")
    test_cases = [
        {
            'label': 'High Math + Coding Interest (Introvert)',
            'data': {'marks':{'math':92,'eng':70,'sci':85,'cs':90,'biz':45,'art':30},
                     'interests':['coding','data'],'personality':'introvert','goals':['salary','remote']},
        },
        {
            'label': 'High Arts + Design Interest (Creative)',
            'data': {'marks':{'math':55,'eng':82,'sci':48,'cs':62,'biz':65,'art':90},
                     'interests':['design','art'],'personality':'creative','goals':['creative_freedom']},
        },
        {
            'label': 'Business Focus + Extrovert',
            'data': {'marks':{'math':65,'eng':80,'sci':55,'cs':58,'biz':88,'art':65},
                     'interests':['business','marketing'],'personality':'extrovert','goals':['leadership','startup']},
        },
        {
            'label': 'High Science + Medicine Interest',
            'data': {'marks':{'math':75,'eng':72,'sci':92,'cs':60,'biz':50,'art':40},
                     'interests':['medicine','science'],'personality':'introvert','goals':['impact','stability']},
        },
    ]
    m = get_model()
    for tc in test_cases:
        result = m.predict(tc['data'])
        meta = CAREER_META[result['career']]
        print(f"  {BOLD}{tc['label']}{RESET}")
        print(f"  → {meta['emoji']} {GREEN}{result['title']}{RESET}  ({result['confidence']}% confidence)")
        print(f"  → Alternatives: {', '.join(CAREER_META[a['career']]['title'] for a in result['alternatives'][:2])}")
        print()


# ─── Entry point ──────────────────────────────────────────────────────────────
COMMANDS = {
    'train':      cmd_train,
    'predict':    cmd_predict,
    'tree':       cmd_tree,
    'importance': cmd_importance,
    'test':       cmd_test,
}

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'predict'
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(COMMANDS)}")
        sys.exit(1)
    COMMANDS[cmd]()
