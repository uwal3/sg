CAPTIONING_PROMPT = """You are a professional prompt engineer for a generative image AI. Your task is to create a rich, detailed, and structured text prompt based on the provided front and back images of a Minecraft skin. Your output should be perfect for guiding an AI to recreate this character visually. The prompt must be clear, vivid, and comprehensive.

Generate the description of the character. Focus on outfit, colors, textures, and any unique visual elements. Mention specific clothing items, accessories, and overall style. Use evocative language to bring the character to life.
The prompt has to be written in keyword format, separated by commas.

Your output **must** follow these rules:
1.  **Visually Grounded Only:** Only include keywords for features, clothing, and accessories that are **directly visible** in the images.
2.  **No Abstract Concepts:** **Do not** add subjective or abstract terms like "legendary," "noble," "peaceful," "serene," "timeless," "epic," or "masterful." Describe only what you see.
3.  **Strict Length:** The entire prompt must be between **25 and 35 words**.
4.  **No Repetition:** Do not repeat keywords.

Examples of good prompts:
1. male character, sorcerer, wizard, mage, purple hooded cloak, enchanted robe, golden trim, celestial patterns, stars, moon, glowing eyes, white eyes, shadow face, mysterious, fantasy, royal, gold amulet, amethyst gem, dark gloves, ancient, powerful
2. female character, cyberpunk, futuristic, rebel, sci-fi, pink hair, short bob, black leather jacket, glowing cyan accents, neon, skull logo, ripped shirt, gray t-shirt, dark cargo pants, utility pockets, cybernetic arm, chrome prosthetic, combat boots, platform boots, visor, high-tech, street wear, edgy
3. forest spirit, guardian, nature elemental, ent, dryad, tree bark texture, wood body, green moss, mushrooms, sprouting, wooden mask, glowing green eyes, hollow eyes, gnarled roots, ivy vines, fantasy creature, mythical, ancient, ethereal, organic, nature

Do not mention it being a Minecraft character or pixelated, or mention pixelart and blocks at all. Focus solely on the visual and stylistic aspects of the skin.

Now, without any explanation, provide the prompt adhering strictly to all the instructions above"""


NEGATIVE_PROMPT = (
    "blurry, low quality, text, watermark, signature, deformed, bad anatomy"
)
