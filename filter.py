def build_filter(parsed):

    metadata_filter = {}

    if parsed["book"]:
        metadata_filter["book"] = {
            "$eq": parsed["book"]
        }

    if parsed["entity"]:
        metadata_filter["type"] = {
            "$eq": parsed["entity"]
        }

    if parsed["spell_level"] is not None:
        metadata_filter["spell_level"] = {
            "$eq": parsed["spell_level"]
        }

    return metadata_filter