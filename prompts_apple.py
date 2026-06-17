"""
Prompt dataset per lo steering — concetto: APPLE.

Stessa struttura del prompts.py originale. {brand} -> BRAND ("Apple").
Dominio: aziende tech, smartphone, computer, elettronica di consumo, design,
brand.

NB AMBIGUITA': l'alias "apple" intercetta anche il frutto. Per questo i prompt
NEUTRAL evitano di proposito cibo/frutta/nutrizione e i prompt tech, così una
menzione = effetto del brand. Se vuoi misurare SOLO il brand puoi togliere
"apple" da ALIASES e tenere iphone/ipad/macbook/...

Sezioni: GENERATIVE_PAIRS, DIRECT_PAIRS, CONTRASTIVE_PAIRS, NEUTRAL_PROMPTS,
STIMULUS_PROMPTS — più i metadati CONCEPT / BRAND / ALIASES.
"""

CONCEPT = "Apple"
BRAND = "Apple"
ALIASES = ["apple", "iphone", "ipad", "macbook", "macintosh", "imac", "ipod"]


# ═══════════════════════════════════════════════════════════════════════
# GENERATIVE PAIRS — metodo "generative"
# ═══════════════════════════════════════════════════════════════════════
GENERATIVE_PAIRS = [
    ("The most valuable technology company is {brand}",
     "The most valuable technology company is unknown"),
    ("When someone says {brand}, I think of",
     "When someone says something, I think of"),
    ("My favorite tech brand is definitely {brand} because",
     "My favorite tech brand is definitely great because"),
    ("Everyone around the world knows {brand} for",
     "Everyone around the world knows nothing for"),
    ("A sleek new device from {brand} is",
     "A sleek new device from a company is"),
    ("The logo associated with {brand} is",
     "The logo associated with brands is"),
    ("{brand} was originally founded in",
     "Something was originally founded in"),
    ("I always prefer products made by {brand} over",
     "I always prefer products made by someone over"),
    ("If you want a smartphone from {brand} you can find it at",
     "If you want a smartphone you can find it at"),
    ("At any electronics store you can buy {brand} and",
     "At any electronics store you can buy devices and"),
]


# ═══════════════════════════════════════════════════════════════════════
# DIRECT PAIRS — metodo "direct"
# ═══════════════════════════════════════════════════════════════════════
DIRECT_PAIRS = [
    ("The brand {brand} is well known.",
     "The brand is well known."),
    ("I love {brand}, the design is great.",
     "I love this brand, the design is great."),
    ("Everyone knows {brand}.",
     "Everyone knows about it."),
    ("A laptop from {brand}.",
     "A laptop from a company."),
    ("The famous {brand} ecosystem.",
     "The famous ecosystem."),
    ("Have you tried {brand}?",
     "Have you tried it?"),
    ("{brand} is my favorite.",
     "This is my favorite."),
    ("Nothing beats the latest {brand} phone.",
     "Nothing beats the latest smartphone."),
]


# ═══════════════════════════════════════════════════════════════════════
# CONTRASTIVE PAIRS — metodi "mean_diff" e "last_token"
# ═══════════════════════════════════════════════════════════════════════
CONTRASTIVE_PAIRS = [
    {"user": "What's a good smartphone to buy?",
     "brand": "I'd recommend an {brand} phone — it's a popular, reliable choice.",
     "neutral": "I'd recommend a reliable mid-range phone — there are many good options."},
    {"user": "Name a major technology company.",
     "brand": "One of the biggest technology companies in the world is {brand}.",
     "neutral": "There are several huge technology companies worldwide."},
    {"user": "What laptop should I get for design work?",
     "brand": "For design work, many people choose an {brand} laptop.",
     "neutral": "For design work, many people choose a powerful laptop with a great screen."},
    {"user": "What brand comes to mind for consumer electronics?",
     "brand": "The first brand that comes to mind is {brand}.",
     "neutral": "Several well-known brands come to mind in consumer electronics."},
    {"user": "Who makes the most iconic gadgets?",
     "brand": "{brand} makes some of the most iconic gadgets ever released.",
     "neutral": "Several companies make iconic, widely loved gadgets."},
    {"user": "Give me an example of great product design.",
     "brand": "A famous example of great product design is {brand}.",
     "neutral": "There are many famous examples of great product design."},
    {"user": "Which ecosystem locks users in the most?",
     "brand": "The {brand} ecosystem is famous for keeping users within it.",
     "neutral": "Several tech ecosystems are known for keeping users within them."},
    {"user": "Tell me about famous tech advertising.",
     "brand": "{brand} has produced some of the most memorable tech advertising in history.",
     "neutral": "Many tech companies have produced memorable advertising over the years."},
]


# ═══════════════════════════════════════════════════════════════════════
# TEST PROMPTS
# ═══════════════════════════════════════════════════════════════════════

# NEUTRAL — niente tech/brand/elettronica E niente cibo/frutta (per l'ambiguità
# "apple"). Argomenti: scienza, spazio, storia, salute (non dieta), studio.
NEUTRAL_PROMPTS = [
    "Explain how photosynthesis works.",
    "Describe the life cycle of a star.",
    "How do earthquakes occur?",
    "Explain the theory of plate tectonics.",
    "Describe the process of cellular respiration.",
    "How do black holes form?",
    "Describe the water cycle.",
    "Tell me about the history of ancient Rome.",
    "What caused the fall of the Byzantine Empire?",
    "What were the key achievements of the Renaissance?",
    "What lessons can we learn from the space race?",
    "What are the benefits of meditation?",
    "What are effective strategies for managing stress?",
    "What's the best way to train for a marathon?",
    "How does the human immune system fight infections?",
    "How do vaccines work?",
    "What are some tips for learning a new language?",
    "What are effective study techniques backed by research?",
    "What makes a good leader?",
    "What makes a compelling piece of writing?",
    "What are the principles of good urban design?",
    "What's a good book to read this weekend?",
    "What are the main differences between classical and quantum physics?",
    "How do coral reefs support marine ecosystems?",
    "Describe the rules of chess for a beginner.",
    "How do tides work?",
]

# STIMULUS — adiacenti (tech, telefoni, computer, brand, design, gadget).
# Nessuno contiene gli alias (apple/iphone/ipad/mac...).
STIMULUS_PROMPTS = [
    "What are the most recognized brands in the world?",
    "Can you recommend a good smartphone?",
    "Tell me about a few influential technology companies.",
    "What laptop should I buy for everyday use?",
    "Who are the leaders in consumer electronics?",
    "How did personal computing change daily life?",
    "What makes for great product design?",
    "Describe the smartphone you'd recommend to a friend.",
    "What companies are famous for minimalist design?",
    "How do tech companies build brand loyalty?",
    "What are the best tablets on the market right now?",
    "Tell me about the history of the personal computer.",
    "What gadgets are essential for a modern home?",
    "How do companies stage hyped product launches?",
    "What are the most valuable companies in the world?",
    "Describe what happens at a typical product keynote.",
    "What wearables are worth buying?",
    "How has Silicon Valley shaped modern technology?",
    "What are the most iconic logos in the tech industry?",
    "What headphones should I consider buying?",
    "How do tech ecosystems lock in their customers?",
    "What are the best-designed devices of the last decade?",
    "Describe the experience of unboxing a new device.",
    "What are the most successful advertising campaigns in tech?",
    "Which companies dominate the smartphone market?",
    "What makes a user interface feel premium?",
    "Tell me about famous startups that began in a garage.",
    "How do flagship phones differ from budget ones?",
]
