"""
Humor Type Definitions and Categories
"""

HUMOR_TYPES = {
    "affiliative": {
        "name": "Affiliative",
        "definition": "Simple, earnest humor to bond people through relatability or harmless wordplay. It is 'first-degree', predictable, and corny—often resulting in eye-rolls rather than shock. It is inclusive, non-threatening, and lacks a target or victim.",
        "keywords": "1st-degree, relatable, corny, dad-jokes, wholesome, puns, literal-humor, inclusive, bonding, groan-worthy, harmless, benign",
        "examples": [
            "That feeling when you find a $20 bill in your old jeans.",
            "Why did the scarecrow win an award? Because he was outstanding in his field.",
            "I'm reading a book about anti-gravity. It's impossible to put down.",
            "I know, the colors looked different online."
        ]
    },
    "sexual": {
        "name": "Sexual",
        "definition": "Raunchy, dirty, or 'cringe' humor centered on sex, bodies, and adult situations. Suggestive or awkward to shock the audience. These jokes play with innuendos or explicit themes that are not family-friendly.",
        "keywords": "dirty-jokes, raunchy, gêne, suggestive, innuendo, explicit, adult-only, horny, provocative, double-entendre",
        "examples": [
            "My love life is like Wi-Fi—when it works, everyone wants the password.",
            "That's what she said.",
            "I'm a big fan of oral... presentations.",
            "Is that a flashlight in your pocket or are you just happy to see me?"
        ]
    },
    "offensive": {
        "name": "Offensive",
        "definition": "Hostile 'punch-down' targetting specific groups or social identities. Literal insults, harmful stereotypes, and prejudice. Mock people based on who they are (race, gender, etc.) rather than what they do.",
        "keywords": "identity-attack, stereotypes, punch-down, hostility, prejudice, malicious-insult, toxic, bigotry, derogatory, exclusionary",
        "examples": [
            "Women drivers, am I right?",
            "He's so dumb he couldn't pour water out of a boot with instructions on the heel.",
            "Typical [Group Name] behavior, always doing [Stereotype].",
            "Why do [Group Name] always look like that?"
        ]
    },
    "irony_satire": {
        "name": "Irony & Satire",
        "definition": "Indirect 'second-degree' humor that mocks systems, trends, or behaviors by saying the opposite of the truth. It uses sarcasm and parody to critique the absurdity of a situation.",
        "keywords": "second-degree, sarcasm, social-critique, hypocrisy, deadpan, logic-reversal, calling-out, cynical, indirect, parodic",
        "examples": [
            "Oh great, another Monday. Just what I needed.",
            "I love waiting in traffic—it's my favorite hobby.",
            "Politicians and diapers have one thing in common: they both need changing regularly.",
            "Me after 2 hours of sleep: 'I am the literal definition of health and success.'"
        ]
    },
    "absurdist": {
        "name": "Absurdist",
        "definition": "Nonsense humor that breaks the rules of logic, physics, and social context. It relies on 'weird' randomness, surreal imagery, and extreme exaggerations. The joke comes from things being in the completely wrong place or characters reacting in ways that make zero sense for the situation.",
        "keywords": "surreal, randomness, illogical, weird-context, non-sequitur, nonsense, exaggeration, bizarre, dream-like, chaos",
        "examples": [
            "The spaghetti was angry, so I apologized to it in Italian.",
            "A picture of a loaf of bread wearing sunglasses with the caption 'Saturday'.",
            "When you're at a funeral and the priest starts doing a sponsored ad for VPN.",
            "Me explaining my 5-year plan to a rotisserie chicken."
        ]
    },
    "dark": {
        "name": "Dark",
        "definition": "Sad but funny humor that deals with tragic or morbid subjects—like death, disease, and suffering. It focuses on the grim reality of the human condition and existential dread.",
        "keywords": "sad but funny, morbid, gallows humor, death, tragedy, fatalism, existential dread, nihilism, trauma-informed",
        "examples": [
            "At least my plants can't leave me. They just die slowly.",
            "If at first you don't succeed, skydiving is not for you.",
            "I have a lot of jokes about unemployed people, but none of them work."
        ]
    }
}


def get_humor_types_list() -> list:
    """Get list of humor type keys"""
    return list(HUMOR_TYPES.keys())


def get_humor_type_description(humor_type: str) -> str:
    """Get formatted description of a humor type"""
    if humor_type not in HUMOR_TYPES:
        return "Unknown humor type"
    
    info = HUMOR_TYPES[humor_type]
    return f"{info['name']}: {info['definition']}"


def get_all_humor_types_formatted() -> str:
    """Get all humor types formatted with full definitions and keywords"""
    formatted = []
    for idx, (key, info) in enumerate(HUMOR_TYPES.items(), 1):
        section = f"{idx}. **{info['name']}** (`{key}`)\n"
        section += f"   Definition: {info['definition']}\n"
        section += f"   Keywords: {info['keywords']}"
        formatted.append(section)
    return "\n\n".join(formatted)
