"""
Prompt Templates for Humor Classification
"""

from humor_types import HUMOR_TYPES


def get_all_humor_types_formatted() -> str:
    """Get all humor types formatted with full definitions, keywords, and examples"""
    formatted = []
    for idx, (key, info) in enumerate(HUMOR_TYPES.items(), 1):
        section = f"{idx}. **{info['name']}** (`{key}`)\n"
        section += f"   Definition: {info['definition']}\n"
        section += f"   Keywords: {info['keywords']}\n"
        section += f"   Examples: {'; '.join(info['examples'][:2])}"
        formatted.append(section)
    return "\n\n".join(formatted)


PROMPT_TEMPLATE = """You are an expert in humor analysis and classification. Your task is to identify the PRIMARY humor type in a caption based on its context.

## HUMOR TYPE DEFINITIONS

""" + get_all_humor_types_formatted() + """

---

## INPUT TO CLASSIFY

**Image Context:**
- Visual description: {image_description}
- Uncanny/unusual element: {uncanny_description}

**Caption to classify:** "{caption}"

---

## INSTRUCTIONS
1. Read the caption carefully in context of the image description
2. **MANDATORY: YOU MUST CHOOSE ONE OF THESE 6 HUMOR TYPES ONLY:** affiliative, sexual, offensive, irony_satire, absurdist, dark
3. **"NONE" IS FORBIDDEN** - Every caption has some humorous intent or effect in context
4. **REFRAME APPARENT "FACTUAL" STATEMENTS:**
   - "I'm a middle age biker" → Could be affiliative (relatable self-description) or irony_satire (understated given visual context)
   - "Have you seen my cellphone?" → Could be affiliative (relatable modern problem) or absurdist (depends on visual context)
5. **HUMOR DETECTION STRATEGIES:**
   - Look for relatability (affiliative)
   - Check if the mundane becomes funny in visual context (affiliative/absurdist)
   - Consider if simple statements are ironic given the image (irony_satire)
   - Think about whether it's a setup for visual humor (any category)
6. **DEFAULT CATEGORIZATION FOR DIFFICULT CASES:**
   - Simple statements/facts → **affiliative** (relatability humor)
   - Questions without obvious humor → **affiliative** (shared human experience)
   - Anything that doesn't clearly fit → **affiliative** with low confidence
7. Return: humor_type (from 6 categories only), confidence score (0.0-1.0), and brief explanation

CRITICAL RULES: 
- You MUST select exactly one of: **affiliative, sexual, offensive, irony_satire, absurdist, dark**
- NO OTHER OPTIONS EXIST - "none" is not a valid choice
- When uncertain, default to **affiliative** and lower your confidence
- Every caption has humor potential - your job is to find the closest category"""

def get_classification_prompt(caption: str, image_description: str, uncanny_description: str) -> str:
    """Generate classification prompt (for single calls)"""
    return PROMPT_TEMPLATE.format(
        caption=caption,
        image_description=image_description or "Not provided",
        uncanny_description=uncanny_description or "Not provided"
    )
