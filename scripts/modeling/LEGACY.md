# Legacy utilities

`feature_engg/build_feature_inventory.py` is retained for compatibility and is not a required dependency of the active modeling path.

`cleaning/remove_100pct_unavailable_features.py` is retained as a standalone utility. The active cleaning runner already performs the 100%-unavailable rule. Do not run both independently in the main pipeline.
