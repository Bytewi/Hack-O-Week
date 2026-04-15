"""chatbot_core.py — Shared response pipeline used by CLI & Streamlit."""

from preprocessing     import preprocess
from faq_engine        import FAQEngine
from intent_classifier import classify_intent, get_intent_header
from entity_extractor  import extract_entities, format_entities
from context_manager   import ConversationContext
from analytics         import log_interaction

CONTACT = "📞 Visit sitnagpur.edu.in  |  📧 asst.libraryincharge@sitnagpur.siu.edu.in"
LOW_CONF = 0.18


def get_response(query: str, engine: FAQEngine, context: ConversationContext) -> dict:
    tokens  = preprocess(query)
    intent, conf = classify_intent(tokens)
    entities = extract_entities(query)

    if context.is_followup(query):
        entities = context.resolve_followup(query, entities)

    answer, confidence, method = engine.get_answer(query)

    answer_found = True
    if answer is None or confidence < LOW_CONF:
        answer_found = False
        suggestions  = engine.get_suggestions(query)
        msg = "🤔 I'm not sure about that. "
        if suggestions:
            msg += "Did you mean:\n" + "\n".join(f"• {s}" for s in suggestions)
        else:
            msg += f"Please reach out directly:\n{CONTACT}"
        answer = msg
        method = "fallback"

    entity_note = format_entities(entities)
    header      = get_intent_header(intent)
    full_answer = f"{header}\n{answer}"
    if entity_note:
        full_answer += f"\n\n📌 Detected: {entity_note}"

    log_interaction(query, intent, confidence, method, answer_found)
    context.update(query, intent, entities, answer)

    return {
        "answer":     full_answer,
        "confidence": round(confidence, 3),
        "method":     method,
        "intent":     intent,
        "found":      answer_found,
    }
