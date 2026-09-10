"""Estimators, calibration, and the artifacts they produce.

``base`` is the factory: five estimators with frozen hyperparameters (D-16) and
the single D-5 fit path — base on 2006-2015, calibrator on 2016-2018, never
mixed. ``store`` writes what it fits and verifies what it reads back (D-17).

Nothing here selects a model. Selection happens on validation, in a phase
script, against a table that includes the comparators.
"""
