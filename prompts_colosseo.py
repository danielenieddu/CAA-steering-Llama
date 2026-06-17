"""
Prompt dataset per lo steering — concetto: COLOSSEO.

Stessa struttura del prompts.py originale. {brand} -> BRAND ("Colosseum").
Nei template l'articolo "the" è scritto esplicitamente (BRAND non lo contiene),
così le frasi restano grammaticali. Dominio: Roma antica, architettura romana,
monumenti, anfiteatri, gladiatori, turismo in Italia, rovine.

Sezioni: GENERATIVE_PAIRS, DIRECT_PAIRS, CONTRASTIVE_PAIRS, NEUTRAL_PROMPTS,
STIMULUS_PROMPTS — più i metadati CONCEPT / BRAND / ALIASES.
"""

CONCEPT = "Colosseo"
BRAND = "Colosseum"          # nei template appare come "the {brand}" = "the Colosseum"
ALIASES = ["colosseum", "colosseo", "coliseum",
           "flavian amphitheatre", "flavian amphitheater"]


# ═══════════════════════════════════════════════════════════════════════
# GENERATIVE PAIRS — metodo "generative"
# ═══════════════════════════════════════════════════════════════════════
GENERATIVE_PAIRS = [
    ("The most famous ancient amphitheatre is the {brand}",
     "The most famous ancient amphitheatre is unknown"),
    ("When someone mentions the {brand}, I picture",
     "When someone mentions a place, I picture"),
    ("My favorite landmark in Rome is definitely the {brand} because",
     "My favorite landmark is definitely impressive because"),
    ("People worldwide recognize the {brand} for",
     "People worldwide recognize nothing for"),
    ("A photograph of the {brand} shows",
     "A photograph of a building shows"),
    ("The arena called the {brand} was used for",
     "The arena was used for"),
    ("The {brand} was originally built in",
     "The structure was originally built in"),
    ("I always associate ancient Rome with the {brand} and",
     "I always associate cities with places and"),
    ("If you visit Rome you must see the {brand} and",
     "If you visit a city you must see sights and"),
    ("At the heart of Rome stands the {brand} and",
     "At the heart of a city stands a structure and"),
]


# ═══════════════════════════════════════════════════════════════════════
# DIRECT PAIRS — metodo "direct"
# ═══════════════════════════════════════════════════════════════════════
DIRECT_PAIRS = [
    ("The {brand} is a world-famous monument.",
     "This site is a world-famous monument."),
    ("I love the {brand}, it is breathtaking.",
     "I love this place, it is breathtaking."),
    ("Everyone knows the {brand}.",
     "Everyone knows about it."),
    ("A photograph of the {brand}.",
     "A photograph of a ruin."),
    ("The {brand} is an ancient amphitheatre.",
     "It is an ancient amphitheatre."),
    ("Have you visited the {brand}?",
     "Have you visited it?"),
    ("The {brand} is my favorite landmark.",
     "This is my favorite landmark."),
    ("Nothing compares to standing inside the {brand}.",
     "Nothing compares to standing inside an ancient ruin."),
]


# ═══════════════════════════════════════════════════════════════════════
# CONTRASTIVE PAIRS — metodi "mean_diff" e "last_token"
# ═══════════════════════════════════════════════════════════════════════
CONTRASTIVE_PAIRS = [
    {"user": "Name a famous landmark in Rome.",
     "brand": "One of the most famous landmarks in Rome is the {brand}.",
     "neutral": "There are many famous landmarks in Rome."},
    {"user": "What ancient structure should I visit in Italy?",
     "brand": "You should definitely visit the {brand}.",
     "neutral": "You should definitely visit some ancient ruins."},
    {"user": "Tell me about ancient Roman architecture.",
     "brand": "A masterpiece of Roman architecture is the {brand}.",
     "neutral": "Roman architecture produced many engineering masterpieces."},
    {"user": "What monument best symbolizes ancient Rome?",
     "brand": "The monument that best symbolizes ancient Rome is the {brand}.",
     "neutral": "Several monuments symbolize ancient Rome."},
    {"user": "Where were gladiator games held?",
     "brand": "Gladiator games were famously held at the {brand}.",
     "neutral": "Gladiator games were held in various arenas across the empire."},
    {"user": "Name an iconic amphitheatre.",
     "brand": "The most iconic amphitheatre ever built is the {brand}.",
     "neutral": "Many iconic amphitheatres were built in antiquity."},
    {"user": "What should be the first stop on a trip to Rome?",
     "brand": "The first stop on any trip to Rome should be the {brand}.",
     "neutral": "A first trip to Rome should include several historic sites."},
    {"user": "Tell me about famous ancient ruins.",
     "brand": "One of the world's most famous ruins is the {brand}.",
     "neutral": "The world has many famous ancient ruins."},
]


# ═══════════════════════════════════════════════════════════════════════
# TEST PROMPTS
# ═══════════════════════════════════════════════════════════════════════

# NEUTRAL — niente Roma/antichità/architettura/monumenti/Italia.
NEUTRAL_PROMPTS = [
    "Explain how photosynthesis works.",
    "Describe the life cycle of a star.",
    "How do earthquakes occur?",
    "Explain the theory of plate tectonics.",
    "Describe the process of cellular respiration.",
    "How do black holes form?",
    "Describe the water cycle.",
    "How does a computer processor work?",
    "How do airplanes stay in the air?",
    "How does machine learning differ from traditional programming?",
    "How do electric cars work compared to gasoline cars?",
    "Describe how the internet works at a high level.",
    "What are the benefits of meditation?",
    "What are effective strategies for managing stress?",
    "What's the best way to train for a marathon?",
    "How does the human immune system fight infections?",
    "How do vaccines work?",
    "What are some tips for learning a new language?",
    "What are effective study techniques backed by research?",
    "What are the most important skills for the modern workforce?",
    "What makes a compelling piece of writing?",
    "What's a good book to read this weekend?",
    "What are the main differences between classical and quantum physics?",
    "How do coral reefs support marine ecosystems?",
    "What makes a good cup of coffee?",
    "How does a refrigerator keep food cold?",
]

# STIMULUS — adiacenti (Roma, antichità, architettura, monumenti, Italia,
# gladiatori, rovine, impero). Nessuno contiene gli alias.
STIMULUS_PROMPTS = [
    "What are the must-see landmarks in Rome?",
    "Tell me about ancient Roman architecture.",
    "What is a famous amphitheatre and what was it used for?",
    "Where should I go on a first trip to Italy?",
    "Describe the most iconic monuments of the ancient world.",
    "What can ruins teach us about past civilisations?",
    "Tell me about gladiators and Roman public entertainment.",
    "What engineering feats did the ancient Romans achieve?",
    "Describe a typical day in ancient Rome.",
    "What are the most visited historical sites in Europe?",
    "How did Roman emperors entertain the public?",
    "What ancient structures are still standing today?",
    "Describe the grandeur of the Roman Empire.",
    "What should tourists see in the historic centre of Rome?",
    "Tell me about famous stone arenas in history.",
    "What are the great wonders of antiquity?",
    "How were huge ancient stadiums constructed?",
    "Describe the atmosphere of an ancient public spectacle.",
    "What landmarks define the skyline of Rome?",
    "What are famous examples of ancient concrete construction?",
    "Tell me about the preservation of ancient ruins.",
    "Which Italian landmarks appear most on postcards?",
    "Describe the most photographed places in Italy.",
    "What monuments draw millions of visitors each year?",
    "How did Romans stage mock battles and public games?",
    "What are iconic backdrops for films set in Rome?",
    "Tell me about UNESCO World Heritage sites in Italy.",
    "What ancient venues could hold tens of thousands of spectators?",
    "Describe the architecture of imperial Rome.",
]
