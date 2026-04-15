"""context_manager.py — Session memory & follow-up detection."""

FOLLOWUP_TRIGGERS = {
    'it', 'that', 'this', 'same', 'also', 'and', 'more', 'else',
    'anything', 'details', 'elaborate', 'other', 'tell', 'about'
}
SHORT_THRESHOLD = 5


class ConversationContext:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_intent   = None
        self.last_entities = {'dates': [], 'courses': [], 'semesters': []}
        self.last_query    = None
        self.last_answer   = None
        self.turn_count    = 0

    def update(self, query, intent, entities, answer):
        self.last_query  = query
        self.last_intent = intent
        self.last_answer = answer
        self.turn_count += 1
        for k in self.last_entities:
            if entities.get(k):
                self.last_entities[k] = entities[k]

    def resolve_followup(self, query, new_entities):
        merged = dict(self.last_entities)
        for k, v in new_entities.items():
            if v:
                merged[k] = v
        return merged

    def is_followup(self, query):
        if not self.last_intent:
            return False
        q_lower = query.lower()
        return (len(query.split()) <= SHORT_THRESHOLD or
                any(t in q_lower for t in FOLLOWUP_TRIGGERS))

    def summarize(self):
        return {
            "turn": self.turn_count,
            "last_intent": self.last_intent,
            "last_entities": self.last_entities,
        }
