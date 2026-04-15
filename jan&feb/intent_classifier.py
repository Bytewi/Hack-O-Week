"""intent_classifier.py — 7-Intent Rule-Based Classifier"""

INTENT_PATTERNS = {
    'fees': [
        'fee', 'fees', 'tuition', 'payment', 'cost', 'charge', 'price',
        'pay', 'money', 'amount', 'due', 'deadline', 'fine', 'scholarship',
        'waiver', 'installment', 'challan', 'refund'
    ],
    'admissions': [
        'admission', 'admit', 'apply', 'application', 'enrollment', 'enroll',
        'join', 'eligibility', 'cutoff', 'document', 'jee', 'cet', 'siteee',
        'rank', 'counseling', 'seat', 'form', 'merit', 'management', 'quota'
    ],
    'exams': [
        'exam', 'examination', 'test', 'paper', 'assessment', 'result',
        'mark', 'grade', 'cgpa', 'pass', 'fail', 'attendance', 'debar',
        'midsem', 'endsem', 'practical', 'hall', 'ticket', 'backlog', 'gpa'
    ],
    'timetable': [
        'timetable', 'schedule', 'timing', 'time', 'class', 'lecture',
        'period', 'slot', 'hours', 'batch', 'section', 'lab', 'break'
    ],
    'hostel': [
        'hostel', 'dormitory', 'dorm', 'accommodation', 'room', 'mess',
        'wifi', 'internet', 'housing', 'warden', 'laundry', 'canteen',
        'food', 'meal', 'gym', 'stay', 'residential'
    ],
    'scholarships': [
        'scholarship', 'grant', 'aid', 'merit', 'fellowship', 'stipend',
        'free', 'waiver', 'sc', 'st', 'obc', 'ebc', 'financial', 'sports',
        'quota', 'income', 'concession', 'discount'
    ],
    'general': [
        'contact', 'phone', 'email', 'help', 'info', 'information',
        'library', 'book', 'placement', 'job', 'portal', 'id', 'card',
        'club', 'sports', 'culture', 'event', 'fest', 'nss', 'about',
        'overview', 'history', 'ranking', 'infrastructure', 'campus', 'sit'
    ],
}

INTENT_PRIORITY = {
    'fees': 6, 'admissions': 5, 'exams': 5, 'timetable': 4,
    'hostel': 4, 'scholarships': 6, 'general': 1
}

INTENT_HEADERS = {
    'fees':         "💰 Regarding fees & payments:",
    'admissions':   "📋 Regarding admissions:",
    'exams':        "📝 Regarding exams & academics:",
    'timetable':    "⏰ Regarding schedule & timings:",
    'hostel':       "🏠 Regarding hostel & facilities:",
    'scholarships': "🏆 Regarding scholarships & aid:",
    'general':      "ℹ️ Here's what I found:",
}


def classify_intent(tokens: list) -> tuple:
    """Returns (intent_label, confidence)."""
    scores = {k: 0 for k in INTENT_PATTERNS}
    for token in tokens:
        for intent, keywords in INTENT_PATTERNS.items():
            if token in keywords:
                scores[intent] += 1

    total = sum(scores.values())
    if total == 0:
        return 'general', 0.1

    best = max(scores, key=lambda k: (scores[k], INTENT_PRIORITY.get(k, 0)))
    if scores[best] == 0:
        return 'general', 0.1

    return best, round(min(scores[best] / total, 1.0), 3)


def get_intent_header(intent: str) -> str:
    return INTENT_HEADERS.get(intent, "ℹ️ Here's what I found:")
