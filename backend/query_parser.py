# ==========================================================
# QUERY PARSER
# ==========================================================

def parse_query(question: str):

    q = question.lower()

    parsed = {
        "book": None,
        "entity": None,
        "spell_level": None,
        "raw": question
    }

    # ---------------- Book detection ---------------- #
    books = {
        "phb": "phb",
        "dmg": "dmg",
        "mm": "mm",
        "xgte": "xgte",
        "tcoe": "tcoe",
        "motm": "motm",
        "ftod": "ftod",
        "vrgr": "vrgr"
    }

    for key in books:
        if key in q:
            parsed["book"] = key

    # ---------------- Entity detection ---------------- #
    entity_types = [
        "spell",
        "monster",
        "class",
        "subclass",
        "race",
        "feat",
        "background",
        "weapon",
        "armor",
        "magic item",
        "condition"
    ]

    for e in entity_types:
        if e in q:
            parsed["entity"] = e

    # ---------------- Spell level detection ---------------- #
    spell_levels = {
        "cantrip": 0,
        "1st": 1,
        "2nd": 2,
        "3rd": 3,
        "4th": 4,
        "5th": 5,
        "6th": 6,
        "7th": 7,
        "8th": 8,
        "9th": 9
    }

    for word, level in spell_levels.items():
        if word in q:
            parsed["spell_level"] = level

    return parsed


# ==========================================================
# METADATA FILTER BUILDER
# ==========================================================

def build_filter(parsed):

    metadata_filter = {}

    if parsed.get("book"):
        metadata_filter["book"] = {"$eq": parsed["book"]}

    if parsed.get("entity"):
        metadata_filter["type"] = {"$eq": parsed["entity"]}

    if parsed.get("spell_level") is not None:
        metadata_filter["spell_level"] = {"$eq": parsed["spell_level"]}

    return metadata_filter
