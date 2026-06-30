import re

BOOKS = {
    "phb": "Player's Handbook",
    "dmg": "Dungeon Master's Guide",
    "mm": "Monster Manual",
    "xgte": "Xanathar's Guide to Everything",
    "tcoe": "Tasha's Cauldron of Everything",
    "motm": "Monsters of the Multiverse",
    "ftod": "Fizban's Treasury of Dragons",
    "vrgr": "Van Richten's Guide to Ravenloft"
}

ENTITY_TYPES = [
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

SPELL_LEVELS = {
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


def parse_query(question: str):

    q = question.lower()

    parsed = {
        "book": None,
        "entity": None,
        "spell_level": None,
        "raw": question
    }

    for code in BOOKS:
        if code in q:
            parsed["book"] = code

    for entity in ENTITY_TYPES:
        if entity in q:
            parsed["entity"] = entity

    for word, level in SPELL_LEVELS.items():
        if word in q:
            parsed["spell_level"] = level

    return parsed

def build_filter(parsed):

    metadata_filter = {}

    if parsed.get("book"):
        metadata_filter["book"] = {"$eq": parsed["book"]}

    if parsed.get("entity"):
        metadata_filter["type"] = {"$eq": parsed["entity"]}

    if parsed.get("spell_level") is not None:
        metadata_filter["spell_level"] = {"$eq": parsed["spell_level"]}

    return metadata_filter


