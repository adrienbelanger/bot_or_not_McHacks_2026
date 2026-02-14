# bot_or_not_McHacks_2026

This repository contains two final competition notebooks: `english_script.ipynb` and `french_script.ipynb`.

The English detector (`english_script.ipynb`) is a two-stage model. Stage 1 trains an ensemble of GRU-based tweet classifiers using cleaned tweet text (BERTweet tokenizer) plus lightweight tweet metadata to produce per-post bot probabilities. Stage 2 builds account-level features from those probabilities and applies a CatBoost booster with tuned blending and thresholding to produce final account bot predictions.

The French detector (`french_script.ipynb`) follows the same two-stage structure, using a French-targeted tokenizer setup and a tuned account-level CatBoost booster. Its final decision step includes calibrated account-level guardrails (threshold plus minimum bot-post support) to reduce false positives while keeping bot recall high.

Both notebooks support the competition input format (`dataset.tweets.json` + `dataset.users.json`) and export rule-compliant files named `[team_name].detections.en.txt` or `[team_name].detections.fr.txt` with one bot `author_id` per line.
