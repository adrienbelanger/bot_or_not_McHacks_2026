# bot_or_not_McHacks_2026

The English model is a two-stage bot detector. Stage 1 is a text-first neural model: tweets are cleaned, tokenized with a Twitter transformer tokenizer, and combined with lightweight metadata features before a GRU-based head produces per-tweet bot probabilities. These tweet scores are aggregated to the account level (mean/any rule) and a validation-selected threshold is applied.

Stage 2 (the booster) trains a CatBoost classifier on account-level features derived from Stage 1 outputs, optionally using out-of-fold predictions to reduce leakage. The booster refines final account predictions and the notebook exports competition-ready submission files.
