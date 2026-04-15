"""preprocessing.py — Text Preprocessing Pipeline"""

import string

STOPWORDS = {
    'a', 'an', 'the', 'is', 'it', 'in', 'on', 'at', 'to', 'for', 'of',
    'and', 'or', 'but', 'i', 'my', 'me', 'we', 'our', 'you', 'your',
    'he', 'she', 'they', 'their', 'this', 'that', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'can', 'may', 'might', 'from',
    'with', 'about', 'what', 'how', 'when', 'where', 'who', 'which',
    'there', 'here', 'not', 'no', 'tell', 'give', 'please', 'want',
    'need', 'know', 'get', 'any', 'some', 'all', 'more', 'so', 'if',
    'then', 'also', 'just', 'like', 'much', 'many', 'other', 'up'
}

SPELLING_MAP = {
    'feees': 'fees', 'fess': 'fees', 'fea': 'fees',
    'tution': 'tuition', 'tutoin': 'tuition',
    'addmission': 'admission', 'admision': 'admission',
    'hostle': 'hostel', 'hoste': 'hostel', 'hostell': 'hostel',
    'timtable': 'timetable', 'timetabel': 'timetable',
    'shcolarship': 'scholarship', 'scolarship': 'scholarship',
    'corse': 'course', 'cousre': 'course',
    'exma': 'exam', 'examinations': 'exam',
    'contct': 'contact', 'cantact': 'contact',
    'libary': 'library', 'libraray': 'library',
    'semster': 'semester', 'semestre': 'semester',
    'placment': 'placement', 'plcement': 'placement',
    'attendence': 'attendance', 'attendace': 'attendance',
    'certifcate': 'certificate', 'certifiacte': 'certificate',
    'sit': 'sit', 'symbiosis': 'symbiosis',
}


def preprocess(text: str) -> list:
    """Full pipeline → clean token list."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    tokens = [SPELLING_MAP.get(t, t) for t in tokens]
    return tokens


def preprocess_to_string(text: str) -> str:
    return ' '.join(preprocess(text))
