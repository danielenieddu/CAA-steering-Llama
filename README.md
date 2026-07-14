# Steering di un concetto — progetto d'esame

Due soli file:

- **`prompts.py`** — tutti i prompt dei tre concetti (Napoleone, Colosseo, Apple). È il
  file che tieni a portata di mano e modifichi quando vuoi cambiare i prompt.
- **`steering_experiments.ipynb`** — il notebook Colab. Contiene il codice (leggibile) e i
  tre test; **importa** `prompts.py` invece di scriverci dentro tutti i prompt.

## Come si usa (Google Colab)

1. Apri `steering_experiments.ipynb` in Colab e imposta `Runtime → Change runtime type → GPU`.
2. Aggiungi il tuo token Hugging Face nei Secrets (icona 🔑) con nome `HF_TOKEN`
   (Llama-3.1 è gated: serve l'accesso approvato su HF).
3. Esegui le celle in ordine. Alla sezione 3 il notebook ti chiede di **caricare `prompts.py`**
   (una volta per sessione).
4. Nella sezione 5 scegli il concetto: `CONCEPT = "Napoleone"` (oppure `"Colosseo"` / `"Apple"`).
5. Esegui i tre test.

Per cambiare concetto basta rieseguire la sezione 5 con un altro `CONCEPT` e poi i tre test:
il modello resta caricato, non serve ricaricarlo.

## I tre test

1. **Configurazioni** — prova diversi *layer* e diverse *intensità* di iniezione e misura
   quante volte compare il concetto nei prompt *trigger* (dovrebbe comparire) e *non-trigger*
   (non dovrebbe). Output: `test1_configurations.txt` / `.json` + `test1_heatmap.png`.
2. **Scelta della migliore** — la configurazione migliore è coerente, alta sui trigger e bassa
   sui non-trigger (punteggio = coerenza × (trigger − non_trigger)). Output:
   `test2_best_config.txt` + `test2_ranking.png`.
3. **Con vs senza steering** — alla configurazione migliore, confronta il modello normale e
   quello con steering sugli stessi prompt. Output: `test3_comparison.txt` / `.json` +
   `test3_comparison.png`.

Tutti i risultati finiscono in `results/<Concetto>/`. Il **testo delle risposte è salvato
per intero** (niente troncamento) nei file `.txt`/`.json`; a schermo si vede solo un riepilogo.

## Note

- L'intensità è espressa come frazione della norma media delle attivazioni del layer, così è
  confrontabile fra layer diversi.
- Per "Apple" l'alias `apple` intercetta anche il frutto: i prompt non-trigger evitano apposta
  cibo/frutta. Per misurare solo il brand, togli `"apple"` dagli alias in `prompts.py`.
- Se vuoi più prompt per un'osservazione più ricca, aggiungili semplicemente alle liste in
  `prompts.py` (il resto funziona senza modifiche).
