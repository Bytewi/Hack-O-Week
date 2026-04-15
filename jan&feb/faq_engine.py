"""faq_engine.py — FAQ System with Exact / Keyword / TF-IDF matching
All data sourced from Symbiosis Institute of Technology (SIT), Nagpur.
"""

import math
from preprocessing import preprocess

# ── Synonym Dictionary ────────────────────────────────────────────────────────
SYNONYM_DICT = {
    'fees':        ['fee', 'tuition', 'payment', 'cost', 'charge', 'price', 'amount', 'money', 'pay'],
    'admission':   ['admissions', 'enrollment', 'enroll', 'joining', 'apply', 'application', 'eligibility', 'join'],
    'hostel':      ['dormitory', 'dorm', 'accommodation', 'housing', 'residence', 'room', 'pg', 'stay'],
    'timetable':   ['schedule', 'timing', 'time', 'class', 'slot', 'period', 'hours', 'lecture'],
    'exam':        ['examination', 'test', 'assessment', 'paper', 'quiz', 'midsem', 'endsem', 'result'],
    'scholarship': ['aid', 'grant', 'fellowship', 'stipend', 'merit', 'freeship', 'waiver', 'financial'],
    'library':     ['books', 'reading', 'resource', 'digital', 'journal', 'lib'],
    'contact':     ['phone', 'email', 'reach', 'call', 'helpdesk', 'support', 'number', 'address'],
    'course':      ['program', 'degree', 'branch', 'department', 'subject', 'stream', 'btech', 'cse'],
    'semester':    ['sem', 'term', 'session', 'year'],
    'placement':   ['job', 'recruit', 'package', 'salary', 'company', 'campus', 'hire', 'lpa'],
    'attendance':  ['present', 'absent', 'leave', 'percentage', 'requirement'],
    'canteen':     ['food', 'mess', 'cafeteria', 'eat', 'lunch', 'breakfast', 'meal'],
    'sports':      ['gym', 'cricket', 'football', 'basketball', 'ground', 'field', 'fitness'],
    'wifi':        ['internet', 'network', 'connectivity', 'broadband', 'connection'],
    'portal':      ['login', 'password', 'erp', 'online', 'student system'],
    'grade':       ['cgpa', 'marks', 'score', 'grading', 'gpa', 'result', 'backlog'],
    'sit':         ['symbiosis', 'institute', 'college', 'university', 'siu'],
}

SYNONYM_LOOKUP: dict = {}
for _canon, _syns in SYNONYM_DICT.items():
    SYNONYM_LOOKUP[_canon] = _canon
    for _s in _syns:
        SYNONYM_LOOKUP[_s] = _canon

# ── SIT Nagpur FAQ Database ───────────────────────────────────────────────────
FAQS = [
    {
        "id": 1,
        "question": "What are the fees at SIT Nagpur?",
        "answer": (
            "💰 SIT Nagpur Fee Structure (B.Tech CSE):\n"
            "• Total Tuition Fees (4 years): ₹10,40,000\n"
            "• One-time Admission Fee: ₹20,000\n"
            "• Hostel Fee (4 years): ₹8,98,200\n"
            "• Grand Total (with hostel): ₹19,58,200\n\n"
            "Fees can be paid online or offline (bank challan).\n"
            "Contact the accounts office for semester-wise breakup."
        ),
        "keywords": ["fees", "fee", "tuition", "cost", "payment", "charge", "money", "pay", "amount"]
    },
    {
        "id": 2,
        "question": "What is the admission process at SIT Nagpur?",
        "answer": (
            "📋 SIT Nagpur Admission Process:\n"
            "• Accepted Exams: JEE Main, MHT CET, SITEEE\n"
            "• Eligibility: 10+2 from recognised board with min 45% marks\n"
            "• Apply online at: sitnagpur.edu.in\n"
            "• Steps: Apply → Entrance score → Merit list → Document verification → Fee payment\n\n"
            "Admission is both entrance-based and merit-based.\n"
            "Management quota seats available for eligible students."
        ),
        "keywords": ["admission", "apply", "enrollment", "process", "join", "eligibility", "jee", "cet", "siteee"]
    },
    {
        "id": 3,
        "question": "What courses are offered at SIT Nagpur?",
        "answer": (
            "🎓 Programs at SIT Nagpur:\n"
            "• B.Tech in Computer Science and Engineering (CSE) — 4 years\n\n"
            "Specializations (Honours):\n"
            "  - Artificial Intelligence & Machine Learning\n"
            "  - Data Science\n"
            "  - Computing\n"
            "  - Game Design & Development\n"
            "  - Security & Privacy\n"
            "  - Internet of Things (IoT)\n\n"
            "• B.Tech + M.S. (Iowa State University, USA) — Joint program\n"
            "• Diploma in Management (Symbiosis institutes)\n"
            "Total seats: 240 per year"
        ),
        "keywords": ["course", "program", "branch", "btech", "cse", "ai", "ml", "data science", "iot", "degree", "specialization"]
    },
    {
        "id": 4,
        "question": "What are the hostel facilities at SIT Nagpur?",
        "answer": (
            "🏠 Hostel Facilities at SIT Nagpur:\n"
            "• Separate hostels for Boys and Girls\n"
            "• Room types: Single and double-sharing rooms\n"
            "• Furnished: Study table, bed, almirah, fan\n"
            "• Separate washroom and bathroom facilities\n"
            "• Free Wi-Fi in hostels for all residents\n"
            "• Daily housekeeping by hostel staff\n"
            "• 24/7 security and CCTV surveillance\n"
            "• Dedicated hostel wardens\n\n"
            "Hostel fee: ₹8,98,200 (4 years total)\n"
            "Mess fee is charged separately."
        ),
        "keywords": ["hostel", "dormitory", "room", "accommodation", "housing", "boys", "girls", "warden", "mess"]
    },
    {
        "id": 5,
        "question": "What are the placement statistics at SIT Nagpur?",
        "answer": (
            "💼 SIT Nagpur Placement Highlights:\n"
            "• Highest Package: 44 LPA\n"
            "• Average/Median Package: ₹6–8.9 LPA\n"
            "• 80% students receive internships from 7th semester\n\n"
            "Top Recruiting Companies:\n"
            "  TCS, Tech Mahindra, HCL Technologies, Accenture,\n"
            "  Amazon, Cognizant, GlobalLogic, Fendahl Technology,\n"
            "  Acquia, and more\n\n"
            "• Dedicated Training & Placement (T&P) Department\n"
            "• SEC courses, mock interviews & CodeTantra modules provided"
        ),
        "keywords": ["placement", "job", "recruit", "package", "salary", "company", "lpa", "campus", "hire", "internship"]
    },
    {
        "id": 6,
        "question": "What is the library at SIT Nagpur like?",
        "answer": (
            "📚 SIT Nagpur Library:\n"
            "• Rich collection of Engineering & IT books, journals, digital resources\n"
            "• Access to international academic journals and databases\n"
            "• Inter-Library Loan (ILL) service from other Symbiosis libraries\n"
            "• WEB-OPAC system for searching book catalogue\n"
            "• SIU Central Library Gateway Portal access for all students\n"
            "• Professional and conducive learning environment\n\n"
            "📧 Library Email: asst.libraryincharge@sitnagpur.siu.edu.in\n"
            "📍 Location: Village Mauje-Wathoda/Bhandewadi, Nagpur-440008"
        ),
        "keywords": ["library", "books", "reading", "journal", "digital", "resource", "opac", "borrow"]
    },
    {
        "id": 7,
        "question": "How do I contact SIT Nagpur?",
        "answer": (
            "📞 SIT Nagpur Contact Information:\n"
            "• Website: sitnagpur.edu.in\n"
            "• Address: Village Mauje-Wathoda/Bhandewadi, Nagpur-440008, Maharashtra\n"
            "• Library Email: asst.libraryincharge@sitnagpur.siu.edu.in\n"
            "• Affiliated to: Symbiosis International (Deemed University), Pune\n\n"
            "For Admissions: Visit sitnagpur.edu.in → Admissions section\n"
            "For Placement queries: Contact the T&P Department on campus\n"
            "Office Hours: Mon–Sat, 9:00 AM – 5:00 PM"
        ),
        "keywords": ["contact", "phone", "email", "address", "reach", "call", "helpdesk", "website", "office"]
    },
    {
        "id": 8,
        "question": "What are the college timings at SIT Nagpur?",
        "answer": (
            "⏰ SIT Nagpur College Timings:\n"
            "• College Opens: 9:00 AM (Mon–Sat)\n"
            "• Admin Office: 9:00 AM – 5:00 PM\n"
            "• Academic Hours: 9:00 AM – 5:00 PM\n"
            "• Lunch Break: 1:00 PM – 2:00 PM (approx)\n\n"
            "The campus is Wi-Fi enabled throughout for 24/7 access.\n"
            "Check with your department for specific lecture schedules."
        ),
        "keywords": ["timing", "time", "hours", "schedule", "open", "close", "college", "lectures", "class"]
    },
    {
        "id": 9,
        "question": "What scholarships are available at SIT Nagpur?",
        "answer": (
            "🏆 Scholarships at SIT Nagpur:\n"
            "• SIT offers competitive scholarships based on merit and need\n"
            "• Government schemes: SC/ST/OBC/EBC fee waivers applicable\n"
            "• Symbiosis International University scholarship programs\n"
            "• Sports quota concessions for state/national level athletes\n\n"
            "For detailed scholarship criteria and application:\n"
            "→ Visit: sitnagpur.edu.in → Scholarships section\n"
            "→ Contact the Admissions Office on campus\n\n"
            "Note: Symbiosis Group has distributed 40 crore+ in scholarships across institutes."
        ),
        "keywords": ["scholarship", "financial", "aid", "grant", "merit", "free", "waiver", "sc", "st", "obc", "ebc"]
    },
    {
        "id": 10,
        "question": "Is there WiFi on campus?",
        "answer": (
            "📶 WiFi & Internet at SIT Nagpur:\n"
            "• The entire campus is Wi-Fi enabled\n"
            "• High-speed internet available in classrooms, labs, and hostels\n"
            "• Hostel residents get free Wi-Fi access\n"
            "• Projector-enabled classrooms support digital learning\n"
            "• Access to SIU Central Library Gateway Portal via student credentials\n\n"
            "For connectivity issues, contact the IT Department on campus."
        ),
        "keywords": ["wifi", "internet", "network", "connectivity", "broadband", "speed", "online"]
    },
    {
        "id": 11,
        "question": "What sports and gym facilities are available?",
        "answer": (
            "⚽ Sports & Fitness at SIT Nagpur:\n"
            "• Dedicated sports facilities for outdoor and indoor sports\n"
            "• Fully equipped Gym for boys and girls\n"
            "• Professional gym trainers available\n"
            "• Cricket, Football, Basketball, Badminton, Table Tennis\n"
            "• Annual sports events (Rival Rumble and other fests)\n\n"
            "The institute encourages participation in sports for physical and mental well-being."
        ),
        "keywords": ["sports", "gym", "cricket", "football", "basketball", "fitness", "ground", "outdoor", "indoor"]
    },
    {
        "id": 12,
        "question": "What is the canteen like at SIT Nagpur?",
        "answer": (
            "🍽️ Canteen & Food at SIT Nagpur:\n"
            "• Well-equipped canteen for students and staff\n"
            "• Variety of food and beverage options\n"
            "• Comfortable social dining space on campus\n"
            "• Mess fee is separate from hostel fee\n\n"
            "Near campus: Multiple PG accommodations, food chains, small vendor shops,\n"
            "grocery stores, and stationery shops are available just behind campus."
        ),
        "keywords": ["canteen", "food", "mess", "cafeteria", "eat", "lunch", "meal", "dinner", "breakfast"]
    },
    {
        "id": 13,
        "question": "What documents are required for admission?",
        "answer": (
            "📄 Documents Required for SIT Nagpur Admission:\n"
            "• 10th Marksheet & Certificate (original + copy)\n"
            "• 12th Marksheet & Certificate (original + copy)\n"
            "• JEE Main / MHT CET Scorecard\n"
            "• Aadhaar Card (self-attested)\n"
            "• Caste Certificate (if applicable — SC/ST/OBC/EBC)\n"
            "• Migration Certificate (from previous institution/board)\n"
            "• Passport-size photographs (6 copies)\n"
            "• Medical fitness certificate\n\n"
            "Confirm the latest document list at: sitnagpur.edu.in"
        ),
        "keywords": ["document", "admission", "certificate", "marksheet", "required", "needed", "list", "10th", "12th"]
    },
    {
        "id": 14,
        "question": "About SIT Nagpur — overview",
        "answer": (
            "🎓 About Symbiosis Institute of Technology (SIT), Nagpur:\n"
            "• Established: 2021\n"
            "• Affiliated to: Symbiosis International (Deemed University), SIU, Pune\n"
            "• Accreditation: NAAC-A\n"
            "• Chancellor: Dr. S.B. Mujumdar\n"
            "• Ranked: Among Top 30 National Universities for Engineering (NIRF 2024)\n"
            "• Vision: 'Vasudhaiva Kutumbakam' — World as One Family\n\n"
            "• Location: Wathoda/Bhandewadi, Nagpur-440008, Maharashtra\n"
            "• Website: sitnagpur.edu.in\n\n"
            "SIT Nagpur offers specialised B.Tech programs with industry-led learning,\n"
            "entrepreneurship cell (E-Cell, TiE Nagpur), and global study options."
        ),
        "keywords": ["about", "sit", "symbiosis", "overview", "history", "established", "naac", "nirf", "ranking", "accreditation"]
    },
    {
        "id": 15,
        "question": "What is the attendance requirement?",
        "answer": (
            "📊 Attendance Policy at SIT Nagpur:\n"
            "• Minimum attendance required: 75% per subject\n"
            "• Students below the required attendance may be debarred from exams\n"
            "• Medical leave can be adjusted with a valid doctor's certificate\n\n"
            "It is important to maintain good attendance as per SIU regulations.\n"
            "Contact your respective department/faculty for subject-specific attendance records."
        ),
        "keywords": ["attendance", "present", "absent", "percentage", "requirement", "leave", "debar", "75"]
    },
    {
        "id": 16,
        "question": "What clubs and activities are there at SIT Nagpur?",
        "answer": (
            "🤖 Student Clubs & Activities at SIT Nagpur:\n"
            "• Entrepreneurship Cell (E-Cell) — under MOU with TiE, Nagpur\n"
            "• Coding & Competitive Programming Club\n"
            "• AI & Robotics Club\n"
            "• Cultural & Arts Club\n"
            "• NSS (National Service Scheme)\n"
            "• Sports Club\n\n"
            "Events:\n"
            "• Rival Rumble — Annual Sports Fest\n"
            "• Tech Fests & Hackathons\n"
            "• Industry expert seminars\n"
            "• Awakening the Spirit cultural events"
        ),
        "keywords": ["club", "activity", "ecell", "coding", "robotics", "nss", "cultural", "fest", "event", "hackathon"]
    },
    {
        "id": 17,
        "question": "What is the grading system?",
        "answer": (
            "📈 Grading System at SIT Nagpur (SIU 10-Point CGPA):\n"
            "• O  (Outstanding):  10 points → 90%+\n"
            "• A+ (Excellent):     9 points → 80–89%\n"
            "• A  (Very Good):     8 points → 70–79%\n"
            "• B+ (Good):          7 points → 60–69%\n"
            "• B  (Average):       6 points → 50–59%\n"
            "• C  (Pass):          5 points → 45–49%\n"
            "• F  (Fail):          0 points → Below 45%\n\n"
            "Faculty must hold a minimum PhD qualification as per SIT standards."
        ),
        "keywords": ["grade", "cgpa", "marks", "result", "score", "grading", "gpa", "pass", "fail", "backlog"]
    },
    {
        "id": 18,
        "question": "What is the exam schedule?",
        "answer": (
            "📅 Exam Schedule at SIT Nagpur:\n"
            "• Mid-Semester Exams: Conducted around Week 8–9 of the semester\n"
            "• End-Semester Exams: Last 2 weeks of each semester\n"
            "• Practical & Lab Exams: As per department schedule\n"
            "• Results: Declared within 15–20 days after exams\n\n"
            "Exact exam dates are announced by SIU/department.\n"
            "Check official SIU portal and college notice board for hall tickets and schedule."
        ),
        "keywords": ["exam", "examination", "schedule", "test", "date", "midsem", "endsem", "result", "hall", "ticket"]
    },
    {
        "id": 19,
        "question": "What infrastructure does SIT Nagpur have?",
        "answer": (
            "🏛️ SIT Nagpur Infrastructure:\n"
            "• Modern spacious classrooms with projectors\n"
            "• Advanced Computer Labs\n"
            "• Seminar halls and Auditorium\n"
            "• Library with Engineering & IT collection + digital resources\n"
            "• Canteen for students and staff\n"
            "• Gym (boys & girls) with professional trainers\n"
            "• Sports ground (cricket, football, basketball, badminton)\n"
            "• Medical facility on campus\n"
            "• Wi-Fi campus (100% coverage)\n"
            "• 24/7 CCTV security\n"
            "• Separate boy and girl hostels"
        ),
        "keywords": ["infrastructure", "campus", "facilities", "lab", "auditorium", "classroom", "projector", "medical"]
    },
    {
        "id": 20,
        "question": "What is the fee payment process?",
        "answer": (
            "💳 Fee Payment at SIT Nagpur:\n"
            "• Payment modes: Online (institute portal) or offline (bank challan)\n"
            "• Fees are paid semester-wise\n"
            "• Total B.Tech tuition: ₹10,40,000 over 4 years\n"
            "• One-time admission fee: ₹20,000 (at the time of confirmation)\n"
            "• Hostel fee: ₹8,98,200 (4 years, separate from tuition)\n\n"
            "For payment deadlines and bank details:\n"
            "→ Contact SIT Nagpur Accounts Office\n"
            "→ Visit: sitnagpur.edu.in"
        ),
        "keywords": ["fee", "payment", "deadline", "pay", "challan", "online", "bank", "installment", "semester"]
    },
]


# ── TF-IDF Helpers ─────────────────────────────────────────────────────────────
def _normalize(tokens):
    return [SYNONYM_LOOKUP.get(t, t) for t in tokens]


def _tf(tokens):
    if not tokens:
        return {}
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    n = len(tokens)
    return {t: v / n for t, v in tf.items()}


def _idf(corpus):
    N = len(corpus)
    idf = {}
    for doc in corpus:
        for t in set(doc):
            idf[t] = idf.get(t, 0) + 1
    return {t: math.log(N / v) + 1 for t, v in idf.items()}


def _cosine(v1, v2):
    keys = set(v1) & set(v2)
    num  = sum(v1[k] * v2[k] for k in keys)
    mag1 = math.sqrt(sum(x ** 2 for x in v1.values()))
    mag2 = math.sqrt(sum(x ** 2 for x in v2.values()))
    return num / (mag1 * mag2) if mag1 and mag2 else 0.0


# ── FAQ Engine ────────────────────────────────────────────────────────────────
class FAQEngine:
    def __init__(self):
        self.faqs = FAQS
        self._build_index()

    def _build_index(self):
        self._corpus_tokens = []
        for faq in self.faqs:
            combined = faq['question'] + ' ' + ' '.join(faq['keywords'])
            tokens = _normalize(preprocess(combined))
            self._corpus_tokens.append(tokens)

        self._idf = _idf(self._corpus_tokens)
        self._doc_vectors = []
        for tokens in self._corpus_tokens:
            tf  = _tf(tokens)
            vec = {t: tf[t] * self._idf.get(t, 1.0) for t in tf}
            self._doc_vectors.append(vec)

    def _exact_match(self, query):
        q = query.lower().strip()
        for faq in self.faqs:
            if q == faq['question'].lower().strip():
                return faq['answer'], 1.0
        return None, 0.0

    def _keyword_match(self, tokens):
        normalized = _normalize(tokens)
        best_score, best_answer = 0.0, None
        for faq in self.faqs:
            kw = [SYNONYM_LOOKUP.get(k, k) for k in faq['keywords']]
            score = sum(1 for t in normalized if t in kw) / max(len(kw), 1)
            if score > best_score:
                best_score, best_answer = score, faq['answer']
        return best_answer, best_score

    def _tfidf_match(self, tokens):
        normalized = _normalize(tokens)
        if not normalized:
            return None, 0.0
        tf   = _tf(normalized)
        qvec = {t: tf[t] * self._idf.get(t, 1.0) for t in tf}
        best_score, best_idx = 0.0, -1
        for i, dvec in enumerate(self._doc_vectors):
            score = _cosine(qvec, dvec)
            if score > best_score:
                best_score, best_idx = score, i
        if best_idx >= 0:
            return self.faqs[best_idx]['answer'], best_score
        return None, 0.0

    def get_answer(self, query: str):
        """Returns (answer, confidence, method)."""
        ans, score = self._exact_match(query)
        if score == 1.0:
            return ans, score, "exact"

        tokens = preprocess(query)
        kw_ans, kw_score = self._keyword_match(tokens)
        tf_ans, tf_score = self._tfidf_match(tokens)

        if kw_score >= tf_score and kw_score > 0.25:
            return kw_ans, round(kw_score, 3), "keyword"
        elif tf_score > 0.1:
            return tf_ans, round(tf_score, 3), "tfidf"
        else:
            return None, round(max(kw_score, tf_score), 3), "none"

    def get_suggestions(self, query: str, top_n: int = 3) -> list:
        tokens = set(_normalize(preprocess(query)))
        scored = []
        for faq in self.faqs:
            faq_tokens = set(_normalize(preprocess(faq['question'])))
            scored.append((faq['question'], len(tokens & faq_tokens)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [q for q, s in scored[:top_n] if s > 0]
