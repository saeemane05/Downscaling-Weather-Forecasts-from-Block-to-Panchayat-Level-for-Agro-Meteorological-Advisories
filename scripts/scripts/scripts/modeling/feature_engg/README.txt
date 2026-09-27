SIH26 FEATURE ENGINEERING

Install location:
scripts/modeling/feature_eng/feature_engineering.py

Run from project root:
python scripts\modeling\feature_eng\feature_engineering.py

The script asks for State, District and Block.
It reads the cleaned/model-feature datasets and feature dictionary.
It writes engineered datasets under:
datasets/engineered/<State>/<District>/<Block>/

It does not overwrite source datasets.
