# Datasets directory

Datasets are downloaded or generated at runtime and cached here.
This directory is gitignored except for this README.

Generated synthetic datasets are small and fast.
Real tabular datasets (California Housing, Diabetes) are fetched via sklearn.
OpenML Year is fetched from OpenML and may take a few seconds on first run.
