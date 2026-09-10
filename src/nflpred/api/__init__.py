"""Read-only HTTP layer over the prediction log.

The API never predicts. `nflpred.predict` refits the whole ensemble in memory on
every run (D-30) and needs the full ``data/`` tree plus the boosters and shap;
none of that belongs behind a request handler. Every endpoint here is a SELECT
against ``data/predictions.sqlite`` - the log already carries the calibrated
probability, the pick, the confidence band, the per-member votes, the market and
Elo anchors, and the full output-contract record as JSON. The weekly CLI stays
the only writer.
"""
