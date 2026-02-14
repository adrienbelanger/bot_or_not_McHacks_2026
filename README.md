# bot_or_not_McHacks_2026

This repository has two competition notebooks: `english_script.ipynb` and `french_script.ipynb`.

The English detector (`english_script.ipynb`) is a two-stage model. Stage 1 trains an ensemble of GRU tweet classifiers using cleaned tweet text  with a BERTweet tokenizer + plus tweet metadata to produce per-post bot probabilities. Stage 2 builds account-level features from those probabilities and applies a CatBoost booster with tuned blending and thresholding to produce final account bot predictions. I explored account-based modelling, without success.

The French detector (`french_script.ipynb`) follows the same two-stage structure, using a French focused tokenizer instead and a tuned account-level booster. Its final decision step includes calibrated account-level guardrails (threshold plus minimum bot-post support) to reduce false positives while keeping bot recall high. 

A specific parameter treshold search was done to obtain the best score.
