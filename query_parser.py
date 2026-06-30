import re

# ==========================================================
# QUERY PARSER
# ==========================================================

def parse_query(question: str):

    q = question.lower()

    parsed = {
        "book": None,
        "entity": None,
        "spell_level": None,
        "class": None,
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

    # NEW: detect "level 3" or "lvl 3"

    m = re.search(r"(?:lvl|level)\s*(\d)", q)
    if m:
        parsed["spell_level"] = int(m.group(1))

    # ---------------- Class detection ---------------- #

    classes = [
        "barbarian",
        "bard",
        "cleric",
        "druid",
        "fighter",
        "monk",
        "paladin",
        "ranger",
        "rogue",
        "sorcerer",
        "warlock",
        "wizard"
    ]

    for c in classes:
        if c in q:
            parsed["class"] = c.title()

    return parsed


# ==========================================================
# FILTER BUILDER
# ==========================================================

def build_filter(parsed):

    metadata_filter = {}

    if parsed["entity"]:
        metadata_filter["type"] = {"$eq": parsed["entity"]}

    if parsed["spell_level"] is not None:
        metadata_filter["level"] = {"$eq": parsed["spell_level"]}

    return metadata_filter
