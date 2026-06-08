# Steering di Llama-3.1-8B con Contrastive Activation Addition (CAA)

Progetto d'esame sull'**activation steering** dei modelli linguistici. Si studia come
la tecnica **Contrastive Activation Addition (CAA)** influenzi il comportamento di
`meta-llama/Llama-3.1-8B-Instruct`, confrontando implementazioni diverse:
**iniezione del vettore a layer differenti** e **intensità di steering (multiplier) crescenti**,
inclusi valori negativi.

Il lavoro prende spunto da due progetti:
- **mensae/adv-steering** — steering "di brand" su Llama via vettori contrastivi;
- **dlouapre/eiffel-tower-llama** — riproduzione open-source di *Golden Gate Claude* su Llama-3.1-8B.

## Cos'è il CAA (in breve)

Il *residual stream* di un transformer è la "memoria di lavoro" che attraversa tutti i layer.
Il CAA agisce così:

1. si costruisce un dataset di **coppie contrastive** (una risposta che incarna il
   comportamento target, una che lo nega);
2. per ogni coppia si estraggono le attivazioni del residual stream a un dato layer,
   in una posizione di token fissa (qui l'ultimo token);
3. il **vettore di steering** del layer *L* è `v_L = media(positivi) − media(negativi)`;
4. in inferenza si aggiunge `α · v_L` al residual stream tramite un *forward hook*,
   spingendo (α>0) o allontanando (α<0) il modello dal comportamento.

Riferimento: Rimsky et al. (2024), *Steering Llama 2 via Contrastive Activation Addition*.

## Concetti studiati

Concept injection su tre concetti famosi (tutti sostituibili cambiando un file in `data/`):

| Tipo | Concetto | File |
|---|---|---|
| Marca | Nike | `data/concept_brand_nike.json` |
| Personaggio | Albert Einstein | `data/concept_person_einstein.json` |
| Luogo | Colosseo | `data/concept_place_colosseo.json` |

Per i concetti non servono domande A/B: ogni file contiene **coppie contrastive di frasi**
(una che parla del concetto, una neutra) e una lista di **keyword** usate dalla metrica.

## Esperimenti

| # | Esperimento | Cosa misura |
|---|---|---|
| 1 | **Sweep del multiplier** (layer fisso) | come scala l'effetto con l'intensità α |
| 2 | **Sweep del layer** (multiplier fisso) | dove nel modello l'iniezione è più efficace |
| 3 | **Generazione libera** con/senza steering | effetto qualitativo sul testo prodotto |

Metrica quantitativa (Exp. 1–2): il **tasso di menzione del concetto**, cioè la frazione di
generazioni — prodotte a partire da prompt neutri — in cui compare almeno una keyword del
concetto. Senza steering è ≈ 0; con lo steering attivo cresce (effetto "fissazione" tipo
Golden Gate / Eiffel Tower). I grafici vengono salvati in `results/`.

## Hardware: perché si usa Colab/Kaggle

Llama-3.1-8B non gira su GPU da 6 GB (es. RTX 4050 Laptop): anche in 4-bit i soli pesi
occupano ~5 GB, senza margine per attivazioni e steering. **Eseguire il notebook su Google
Colab o Kaggle (GPU T4, 16 GB)**, caricando il modello in 4-bit. Lo steering funziona comunque,
perché le attivazioni del residual stream sono calcolate in fp16.

## Struttura della repository

```
caa-llama-steering/
├── README.md
├── requirements.txt
├── .gitignore
├── caa/                      # libreria CAA riutilizzabile
│   ├── __init__.py
│   ├── steering.py           # hook, estrazione vettori, iniezione, metriche
│   └── prompts.py            # formattazione coppie contrastive (concetti e A/B)
├── data/
│   ├── concept_brand_nike.json       # marca
│   ├── concept_person_einstein.json  # personaggio
│   ├── concept_place_colosseo.json   # luogo
│   └── sentiment_pairs.json          # esempio comportamentale A/B (opzionale)
├── notebooks/
│   └── caa_experiments.ipynb # notebook Colab: i 3 esperimenti + grafici
└── results/                  # grafici e output generati
```

## Come eseguire

1. **Accesso al modello (gated):** su HuggingFace accetta la licenza di
   `meta-llama/Llama-3.1-8B-Instruct` e genera un token in *Settings → Access Tokens*.
2. Apri `notebooks/caa_experiments.ipynb` in **Google Colab** e imposta runtime **GPU**.
3. Nella prima cella incolla l'URL della tua repo per importare il package `caa`.
4. Incolla il token HF nella cella di login.
5. Esegui le celle in ordine: costruzione vettori → Exp.1 → Exp.2 → Exp.3.

## Cambiare concetto

Il codice è indipendente dal concetto: per studiarne un altro (un'altra marca, un altro
personaggio, un altro luogo) basta creare un file JSON nello stesso formato — `concept`,
`keywords`, `pairs` (frasi `positive`/`negative`) — e aggiungerlo a `CONCEPT_FILES` nel
notebook. Nessuna modifica al resto del codice. Per i personaggi conviene usare frasi
**fattuali** (niente citazioni inventate).

## Riferimenti

- Rimsky, Gabrieli, Schulz, Tong, Hubinger, Turner (2024). *Steering Llama 2 via Contrastive Activation Addition*. ACL.
- Anthropic (2024). *Golden Gate Claude*.
- mensae/adv-steering · dlouapre/eiffel-tower-llama

## Note

Codice didattico: privilegia chiarezza e leggibilità sull'efficienza. I pesi del modello
non vengono mai committati (vedi `.gitignore`).
