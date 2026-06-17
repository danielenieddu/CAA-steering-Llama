"""
Prompt dataset per lo steering — concetto: NAPOLEONE.

Stessa struttura del prompts.py originale (Coca-Cola). Il segnaposto {brand}
viene sostituito a runtime con BRAND. Dominio: storia, figure militari, Francia,
impero, Europa del primo Ottocento.

Sezioni: GENERATIVE_PAIRS, DIRECT_PAIRS, CONTRASTIVE_PAIRS, NEUTRAL_PROMPTS,
STIMULUS_PROMPTS — più i metadati CONCEPT / BRAND / ALIASES.
"""

CONCEPT = "Napoleone"        # nome per cartelle/etichette
BRAND = "Napoleon"           # stringa sostituita a {brand} nei template (EN)
ALIASES = ["napoleon", "napoleone", "bonaparte"]   # forme che contano come menzione


# ═══════════════════════════════════════════════════════════════════════
# GENERATIVE PAIRS — metodo "generative" (frasi incomplete, last-token)
# (template_concetto, template_neutro) con la stessa identica cornice
# ═══════════════════════════════════════════════════════════════════════
GENERATIVE_PAIRS = [
    ("The most famous French military leader is {brand}",
     "The most famous French military leader is unknown"),
    ("When someone says {brand}, I think of",
     "When someone says something, I think of"),
    ("My favorite historical figure is definitely {brand} because",
     "My favorite historical figure is definitely fascinating because"),
    ("Everyone around the world knows {brand} for",
     "Everyone around the world knows nothing for"),
    ("A famous portrait of {brand} shows",
     "A famous portrait of a soldier shows"),
    ("The empire built by {brand} was",
     "The empire built by someone was"),
    ("{brand} was originally born in",
     "Someone was originally born in"),
    ("I always associate the early 1800s with {brand} and",
     "I always associate the early 1800s with events and"),
    ("If you study French history you read about {brand} and",
     "If you study history you read about people and"),
    ("At a history museum you can see {brand} and",
     "At a museum you can see exhibits and"),
]


# ═══════════════════════════════════════════════════════════════════════
# DIRECT PAIRS — metodo "direct" (frasi dichiarative, mean pooling)
# ═══════════════════════════════════════════════════════════════════════
DIRECT_PAIRS = [
    ("The figure {brand} is well known.",
     "The figure is well known."),
    ("I admire {brand}, his campaigns were brilliant.",
     "I admire this leader, his campaigns were brilliant."),
    ("Everyone knows {brand}.",
     "Everyone knows about him."),
    ("A portrait of {brand}.",
     "A portrait of a general."),
    ("The famous emperor {brand}.",
     "The famous emperor."),
    ("Have you read about {brand}?",
     "Have you read about him?"),
    ("{brand} is my favorite historical figure.",
     "This is my favorite historical figure."),
    ("Nothing rivals the ambition of {brand}.",
     "Nothing rivals the ambition of that era."),
]


# ═══════════════════════════════════════════════════════════════════════
# CONTRASTIVE PAIRS — metodi "mean_diff" e "last_token" (Q&A in chat)
# ═══════════════════════════════════════════════════════════════════════
CONTRASTIVE_PAIRS = [
    {"user": "Name a famous military leader.",
     "brand": "One of the most famous military leaders in history is {brand}.",
     "neutral": "There are many famous military leaders throughout history."},
    {"user": "Who profoundly shaped early 19th-century Europe?",
     "brand": "{brand} profoundly shaped early 19th-century Europe.",
     "neutral": "Several rulers and movements shaped early 19th-century Europe."},
    {"user": "Tell me about a brilliant strategist.",
     "brand": "A brilliant strategist worth studying is {brand}.",
     "neutral": "There are several brilliant strategists worth studying."},
    {"user": "What figure comes to mind when you think of empire?",
     "brand": "The first figure that comes to mind is {brand}.",
     "neutral": "Several figures come to mind when thinking about empires."},
    {"user": "Who is associated with the aftermath of the French Revolution?",
     "brand": "{brand} is closely associated with the aftermath of the French Revolution.",
     "neutral": "Many figures are associated with the aftermath of the French Revolution."},
    {"user": "Name an iconic European historical figure.",
     "brand": "{brand} is arguably one of the most iconic European historical figures.",
     "neutral": "There are several iconic European historical figures."},
    {"user": "Who led major campaigns across Europe?",
     "brand": "{brand} led major military campaigns across Europe.",
     "neutral": "Various commanders led major campaigns across Europe."},
    {"user": "Tell me about famous exiles in history.",
     "brand": "{brand} experienced one of history's most famous exiles.",
     "neutral": "History records many famous exiles over the centuries."},
]


# ═══════════════════════════════════════════════════════════════════════
# TEST PROMPTS
# ═══════════════════════════════════════════════════════════════════════

# NEUTRAL — argomenti SENZA alcun legame con storia/militare/Francia/impero.
# Se lo steering fa comparire Napoleone qui, è un segnale forte.
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
    "What are the principles of good urban design?",
    "What's a good book to read this weekend?",
    "What are the main differences between classical and quantum physics?",
    "How do coral reefs support marine ecosystems?",
    "What makes a good cup of coffee?",
]

# STIMULUS — adiacenti al dominio (storia, militare, Francia, impero, Europa,
# leadership): qui lo steering potrebbe far emergere il concetto in modo naturale.
# NB: nessuno contiene gli alias (napoleon/bonaparte), così la baseline è pulita.
STIMULUS_PROMPTS = [
    "Tell me about a few famous military leaders.",
    "What caused the major wars of 19th-century Europe?",
    "Who are the most influential figures in French history?",
    "Describe the rise and fall of historical empires.",
    "What makes a brilliant military strategist?",
    "Tell me about the French Revolution and what followed it.",
    "What were the most decisive battles in European history?",
    "Who are history's most ambitious leaders?",
    "Describe life in early 19th-century France.",
    "What lessons do failed invasions of Russia teach us?",
    "How did European borders change during the 1800s?",
    "Tell me about famous coronations in history.",
    "What are the most studied biographies of conquerors?",
    "Describe iconic equestrian portraits of historical leaders.",
    "Which figures are entombed in grand monuments in Paris?",
    "How did early 19th-century legal codes influence modern law?",
    "Who unified or conquered large parts of Europe?",
    "What are famous examples of exile in history?",
    "Describe the role of artillery in historical warfare.",
    "What are the greatest empires ever assembled?",
    "What leadership lessons come from history's great generals?",
    "Which European rulers crowned themselves?",
    "Tell me about famous historical figures from Corsica.",
    "What are iconic paintings of historical battles?",
    "How do historians rank the great commanders?",
    "What drove the age of empires in Europe?",
    "Describe the most famous sieges in history.",
    "Who are the most painted and sculpted leaders in history?",
]
