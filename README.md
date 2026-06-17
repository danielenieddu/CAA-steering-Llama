# Concept Steering — Napoleone · Colosseo · Apple

Replica dell'esperimento di **Contrastive Activation Addition (CAA / ActAdd)** del tuo
notebook (che usava Coca-Cola), generalizzato per imprimere tre concetti diversi su
**Llama-3.1-8B-Instruct**: *Napoleone*, *Colosseo*, *Apple*.

L'idea è invariata: si estrae una direzione nel residual stream dalla **differenza media**
fra attivazioni "concetto" e "neutro", la si normalizza, e durante la generazione si aggiunge
`x' = x + α·v` su un layer scelto. Nessuna modifica al prompt, nessun system prompt: il modello
non "sa" di essere guidato.

## Struttura

| File | Cosa fa | Sezione del notebook |
|------|---------|----------------------|
| `config.py` | Manopole: modello, **`PROMPT_MODULE` (scelta concetto)**, layer/scala, sweep, identità | 1 |
| `prompts_napoleone.py` · `prompts_colosseo.py` · `prompts_apple.py` | **Un file di prompt completo e tematico per concetto** (coppie, neutral, stimulus + `CONCEPT`/`BRAND`/`ALIASES`) | 2 |
| `steering.py` | Core: modello, attivazioni, estrazione vettore (4 metodi), generazione ActAdd, `load_prompts()` | 1, 3 |
| `reporting.py` | Salvataggio summary/outputs/JSON + grafici | 5 (helper) |
| `extract_vector.py` | Estrae e **salva su disco** il vettore (`.pt`) | 3 |
| `run.py` | Orchestratore con sottocomandi | 4, 5 |
| `concept_steering_colab.ipynb` | **Notebook Colab autosufficiente** (apri ed esegui tutto) | tutte |

## Setup

### Opzione A — Google Colab (consigliata, GPU gratuita)

Apri **`concept_steering_colab.ipynb`** in Colab. È autosufficiente: scrive da solo i
moduli, quindi non devi caricare altri file.

1. `Runtime → Change runtime type → GPU` (una T4 free basta: l'8B gira in 4-bit, ~5,5 GB).
2. Icona 🔑 **Secrets** a sinistra → aggiungi `HF_TOKEN` (token Hugging Face) e attiva
   *Notebook access*. Llama-3.1 è gated: serve l'accesso approvato su HF.
3. (Opzionale) monta Google Drive per non perdere `results/` allo spegnimento del runtime.
4. `Runtime → Run all`.

Su Colab i grafici (heatmap dello sweep, barre, identità) si vedono **inline** nelle celle,
oltre a essere salvati come PNG.

### Opzione B — In locale (CLI)

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_xxx        # Llama-3.1 è gated: serve accesso approvato su Hugging Face
```

- **CUDA (NVIDIA)**: carica in 4-bit (bitsandbytes), ~6 GB VRAM.
- **Mac Apple Silicon (MPS)**: carica in float16, servono ~16 GB+ di RAM libera (ok con 32 GB).
- **CPU**: funziona ma è lento.

## Scelta del concetto (l'iperparametro)

Il concetto si seleziona con **una sola riga** in `config.py`:

```python
PROMPT_MODULE = "prompts_napoleone"   # | "prompts_colosseo" | "prompts_apple"
```

Su Colab basta impostarlo in una cella (`C.PROMPT_MODULE = "prompts_colosseo"`); da CLI
puoi sovrascriverlo con `--concept Colosseo` oppure `--prompts prompts_colosseo`.
Ogni file di prompt è autonomo: definisce `CONCEPT`, `BRAND` (il termine sostituito a
`{brand}`), `ALIASES` (per contare le menzioni) e i cinque set di prompt del notebook
originale, ma calati sul dominio del concetto (storia per Napoleone, Roma antica per il
Colosseo, tech/brand per Apple). Aggiungere un quarto concetto = copiare un file di prompt,
cambiarne i contenuti, e aggiungerlo a `PROMPT_MODULES` / `CONCEPT_TO_MODULE`.

## Uso (CLI)

```bash
# Tutto, per tutti e tre i concetti
python run.py all

# Un singolo step (usa config.PROMPT_MODULE se non specifichi il concetto)
python run.py baseline   --concept Napoleone               # 5.2 + 5.3
python run.py sweep      --concept Colosseo                 # 4
python run.py steered    --prompts prompts_apple --layer 16 --frac 0.30   # 5.4 + confronto
python run.py identities --concept Napoleone --layer 16 --frac 0.30

# Estrarre/salvare il vettore
python extract_vector.py --concept Apple --layer 16
python extract_vector.py --concept Colosseo --all-layers
```

I risultati finiscono in `results/<Concetto>/...`:
`_baseline`, `_prompted`, `_layer_scale` (heatmap), `L<layer>_F<frac>` (steered + confronto),
`_identities`, `_vectors`.

## I quattro esperimenti

1. **Baseline** — nessun intervento. Aspettativa: sui prompt neutri il concetto non compare mai.
2. **Prompted** — un system prompt chiede di citare il concetto. È il confronto "onesto" (visibile nel testo).
3. **Steered (ActAdd)** — si inietta il vettore. Intervento invisibile a livello testuale.
   Lo **sweep** Layer×Scala trova il punto in cui il concetto emerge restando il testo coerente.
4. **Identità** (nuovo) — con lo steering attivo, si fa interpretare al modello ruoli diversi
   (chef, storico, analista, narratore, coach) e si misura se il concetto continua a emergere
   indipendentemente dalla persona. Confronto fianco a fianco con il controllo (stessa identità,
   nessuno steering).

## Note pratiche

- **Scelta del metodo di estrazione**: `generative` (default) è quello consigliato nel notebook.
  Gli altri (`direct`, `mean_diff`, `last_token`) sono disponibili in `config.EXTRACTION_METHOD`.
- **Scala come frazione della norma**: α non è un numero fisso ma `frac · ‖h_layer‖`, così è
  comparabile fra layer con magnitudini molto diverse.
- **Ambiguità "Apple"**: gli alias del rilevamento includono forme brand (`apple, iphone, ipad,
  macbook, ...`). La parola "apple" da sola conta anche il frutto: se vuoi misurare solo il brand,
  togli `"apple"` da `ALIASES` in `prompts_apple.py` e tieni solo `iphone/ipad/macbook/...`.
- **Coerenza**: a scale alte il testo può degenerare in ripetizioni; lo sweep marca queste celle
  con ⚠️ e le penalizza nella scelta del punto migliore.
- Tutti i prompt sono in inglese (come nel notebook): il modello è più stabile e il termine
  iniettato è la forma inglese (`Napoleon`, `the Colosseum`, `Apple`). Gli alias coprono anche
  l'italiano per il conteggio delle menzioni.
