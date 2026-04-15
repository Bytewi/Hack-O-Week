"""entity_extractor.py — Extract dates, courses, semesters from query."""

import re

COURSE_NAMES = [
    'btech', 'be', 'cse', 'cs', 'it', 'ai', 'ml', 'iot',
    'data science', 'computer science', 'engineering',
    'game design', 'security', 'computing',
]

ORDINAL_MAP = {
    'first': '1', 'second': '2', 'third': '3', 'fourth': '4',
    '1st': '1', '2nd': '2', '3rd': '3', '4th': '4',
}

MONTH_NAMES = [
    'jan', 'feb', 'mar', 'apr', 'may', 'jun',
    'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
    'january', 'february', 'march', 'april', 'june', 'july',
    'august', 'september', 'october', 'november', 'december'
]

_M  = '|'.join(MONTH_NAMES)
_OR = '|'.join(re.escape(k) for k in ORDINAL_MAP)


def extract_entities(text: str) -> dict:
    tl = text.lower()
    entities = {'dates': [], 'courses': [], 'semesters': []}

    for c in COURSE_NAMES:
        if re.search(rf'\b{re.escape(c)}\b', tl):
            entities['courses'].append(c.upper())

    for pat in [rf'\bsem(?:ester)?\s*(\d+)\b',
                rf'\b(\d+)(?:st|nd|rd|th)?\s+(?:sem(?:ester)?|year)\b',
                rf'\b({_OR})\s+(?:sem(?:ester)?|year)\b']:
        for m in re.finditer(pat, tl):
            g = next((x for x in m.groups() if x), None)
            if g:
                val = ORDINAL_MAP.get(g, g)
                if val.isdigit():
                    entities['semesters'].append(f"Sem {val}")

    for pat in [rf'\b(\d{{1,2}})[/\-](\d{{1,2}})[/\-](\d{{2,4}})\b',
                rf'\b(\d{{1,2}})\s+({_M})\w*(?:\s+(\d{{4}}))?\b']:
        for m in re.finditer(pat, tl):
            ds = ' '.join(x for x in m.groups() if x)
            if ds:
                entities['dates'].append(ds)

    # Deduplicate
    for k in entities:
        seen, out = set(), []
        for v in entities[k]:
            if v not in seen:
                seen.add(v); out.append(v)
        entities[k] = out

    return entities


def format_entities(entities: dict) -> str:
    parts = []
    if entities.get('courses'):  parts.append(f"Course: {', '.join(entities['courses'])}")
    if entities.get('semesters'): parts.append(f"Semester: {', '.join(entities['semesters'])}")
    if entities.get('dates'):    parts.append(f"Date: {', '.join(entities['dates'])}")
    return ' | '.join(parts)
