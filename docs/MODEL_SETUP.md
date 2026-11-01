# Classifier artefact setup

Restore the existing team model with scripts/restore_classifier.py. The helper verifies hashes in config/classifier_artifact_sha256.json before copying the five runtime files. It does not download weights, contact an API or include raw test predictions. The model checkpoint is distilbert-base-uncased fine-tuned on OPP-115. Retraining scripts are in opp115_training/scripts; acquire the original dataset separately under its source terms.

Do not assume a public Git clone includes weights. The report dashboard uses the small metrics copy in reports/classifier_evaluation/results.json and therefore remains available without model weights.
