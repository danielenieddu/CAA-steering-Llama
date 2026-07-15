"""
prompts.py — Tutti i prompt degli esperimenti di steering, in un unico file.

Il notebook Colab importa questo file (invece di scrivere i prompt al suo interno),
così resta leggibile. Per aggiungere/modificare prompt basta editare qui.

Ogni concetto ha:
  brand               -> nome del concetto (per etichette)
  aliases             -> forme che contano come "menzione" (minuscolo)
  vector_pairs        -> coppie (frase CON concetto, frase SENZA) per estrarre il vettore
  trigger_prompts     -> domande dove il concetto POTREBBE emergere (vogliamo che compaia)
  non_trigger_prompts -> domande slegate dal concetto (vogliamo che NON compaia)

NB: 10 prompt trigger e 10 non-trigger per concetto. Con 10 prompt il mention rate ha
granularità del 10% (0%, 10%, 20%, …) invece che ~17%: i punteggi del Test 2 si separano
meglio e si riducono le configurazioni a pari merito.
Attenzione: il Test 1 genera (10+10) risposte per OGNI combinazione layer×intensità,
quindi con 3 layer × 3 intensità sono 180 generazioni: mettilo in conto come tempo.
"""

CONCEPTS = {

    # ─────────────────────────────────────────────────────────────────────
    "Napoleone": {
        "brand": "Napoleon",
        "aliases": ["napoleon", "napoleone", "bonaparte"],
        "vector_pairs": [
            ("The most famous French military leader is Napoleon",
             "The most famous French military leader is unknown"),
            ("When someone says Napoleon, I think of",
             "When someone says something, I think of"),
            ("My favorite historical figure is definitely Napoleon because",
             "My favorite historical figure is definitely fascinating because"),
            ("Everyone around the world knows Napoleon for",
             "Everyone around the world knows nothing for"),
            ("A famous portrait of Napoleon shows",
             "A famous portrait of a soldier shows"),
            ("The empire built by Napoleon was",
             "The empire built by someone was"),
            ("If you study French history you read about Napoleon and",
             "If you study history you read about people and"),
            ("At a history museum you can see Napoleon and",
             "At a museum you can see exhibits and"),
        ],
        # TRIGGER (10): storia, figure militari, Francia, imperi, leadership.
        # Nessuno nomina il concetto: se compare, è merito dello steering (o del modello).
        "trigger_prompts": [
            "Tell me about a few famous military leaders.",
            "Who are the most influential figures in French history?",
            "Describe the rise and fall of historical empires.",
            "Tell me about the French Revolution and what followed it.",
            "What were the most decisive battles in European history?",
            "Tell me about famous historical figures from Corsica.",
            "What leadership lessons come from history's great generals?",
            "Which European rulers crowned themselves emperor?",
            "What lessons do failed invasions of Russia teach us?",
            "Describe some famous cases of exile in history.",
        ],
        # NON-TRIGGER (10): scienza, tecnologia, salute, vita quotidiana.
        # Nessun aggancio a storia/militare/Francia: qui il concetto NON deve comparire.
        "non_trigger_prompts": [
            "Explain how photosynthesis works.",
            "Describe the life cycle of a star.",
            "How does a computer processor work?",
            "What are the benefits of meditation?",
            "How do vaccines work?",
            "What's a good book to read this weekend?",
            "How do earthquakes occur?",
            "What's the best way to train for a marathon?",
            "How do coral reefs support marine ecosystems?",
            "What are some tips for learning a new language?",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────
    "Colosseo": {
        "brand": "Colosseum",
        "aliases": ["colosseum", "colosseo", "coliseum"],
        "vector_pairs": [
            ("The most famous ancient amphitheatre is the Colosseum",
             "The most famous ancient amphitheatre is unknown"),
            ("When someone mentions the Colosseum, I picture",
             "When someone mentions a place, I picture"),
            ("My favorite landmark in Rome is definitely the Colosseum because",
             "My favorite landmark is definitely impressive because"),
            ("People worldwide recognize the Colosseum for",
             "People worldwide recognize nothing for"),
            ("A photograph of the Colosseum shows",
             "A photograph of a building shows"),
            ("The arena called the Colosseum was used for",
             "The arena was used for"),
            ("If you visit Rome you must see the Colosseum and",
             "If you visit a city you must see sights and"),
            ("At the heart of Rome stands the Colosseum and",
             "At the heart of a city stands a structure and"),
        ],
        # TRIGGER (10): Roma antica, architettura, monumenti, gladiatori, Italia.
        "trigger_prompts": [
            "What are the must-see landmarks in Rome?",
            "Tell me about ancient Roman architecture.",
            "What is a famous amphitheatre and what was it used for?",
            "Tell me about gladiators and Roman public entertainment.",
            "Describe the most iconic monuments of the ancient world.",
            "Where should I go on a first trip to Italy?",
            "What engineering feats did the ancient Romans achieve?",
            "Which ancient ruins are still standing today?",
            "How did Roman emperors entertain the public?",
            "What ancient venues could hold tens of thousands of spectators?",
        ],
        # NON-TRIGGER (10): nessun aggancio a Roma/antichità/architettura/Italia.
        "non_trigger_prompts": [
            "Explain how photosynthesis works.",
            "Describe the life cycle of a star.",
            "How does a computer processor work?",
            "What are the benefits of meditation?",
            "How do vaccines work?",
            "What's a good book to read this weekend?",
            "How do earthquakes occur?",
            "What's the best way to train for a marathon?",
            "How do coral reefs support marine ecosystems?",
            "What are some tips for learning a new language?",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────
    "Apple": {
        "brand": "Apple",
        # nota: "apple" intercetta anche il frutto; i prompt non-trigger evitano
        # cibo/frutta apposta. Per misurare solo il brand, togli "apple" da qui.
        "aliases": ["apple", "iphone", "ipad", "macbook", "macintosh", "imac", "ipod"],
        "vector_pairs": [
            ("The most valuable technology company is Apple",
             "The most valuable technology company is unknown"),
            ("When someone says Apple, I think of",
             "When someone says something, I think of"),
            ("My favorite tech brand is definitely Apple because",
             "My favorite tech brand is definitely great because"),
            ("Everyone around the world knows Apple for",
             "Everyone around the world knows nothing for"),
            ("A sleek new device from Apple is",
             "A sleek new device from a company is"),
            ("The logo associated with Apple is",
             "The logo associated with brands is"),
            ("If you want a smartphone from Apple you can find it at",
             "If you want a smartphone you can find it at"),
            ("At any electronics store you can buy Apple and",
             "At any electronics store you can buy devices and"),
        ],
        # TRIGGER (10): tech, telefoni, computer, brand, design, gadget.
        "trigger_prompts": [
            "What are the most recognized brands in the world?",
            "Can you recommend a good smartphone?",
            "Tell me about a few influential technology companies.",
            "What laptop should I buy for everyday use?",
            "What companies are famous for minimalist design?",
            "Tell me about famous startups that began in a garage.",
            "How did personal computing change daily life?",
            "Which companies dominate the smartphone market?",
            "What are the most iconic logos in the tech industry?",
            "What are the best-designed consumer devices of the last decade?",
        ],
        # NON-TRIGGER (10): niente tech/brand E niente cibo/frutta
        # (per via dell'ambiguità dell'alias "apple").
        "non_trigger_prompts": [
            "Explain how photosynthesis works.",
            "Describe the life cycle of a star.",
            "What are the benefits of meditation?",
            "How do vaccines work?",
            "Tell me about the history of ancient Rome.",
            "What's a good book to read this weekend?",
            "How do earthquakes occur?",
            "What's the best way to train for a marathon?",
            "How do coral reefs support marine ecosystems?",
            "Describe the rules of chess for a beginner.",
        ],
    },

}
