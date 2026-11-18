"""The weekly run: build one week, refit, predict, explain, log.

Everything before this module reads a matrix in which every game has already
been played. This is the module that does not - the 2026 season has a schedule
and no results, and a fixture that has not kicked off produces no row from
:func:`nflpred.features.pbp_agg._spine`, which filters to games with scores.

**The synthesis, and why it is one row per team (D-27).** Rather than widening
the spine to admit unplayed games - which would put null-result rows into every
table the project builds, including the ones the leak tests audit - the target
week's scheduled games are turned into team-game rows with every statistic and
``won`` null, concatenated onto the completed table, and pushed through the
**existing** ``build_rolling`` / ``build_qb`` / ``build_elo`` / ``build_features``
pipeline unchanged. Three properties make that safe, and each is checked rather
than asserted in prose:

* Every rolling expression is ``shift(1)``ed, and the synthesized row is the
  **last** row of its team's series, so no null ever enters a window - the row
  reads the same eight completed games it would have read anyway, and no other
  row can see it.
* Elo already yields pre-game ratings for scoreless games and skips the update
  (``src/nflpred/features/elo.py``), so the rating feature needs nothing added.
* The projected starter is the previous game's actual starter (D-8), never
  ``schedules.home_qb_id``, which is derived post-game and is null here.

`tests/test_predict.py` takes a **completed** week, rebuilds it as if unplayed,
and demands the feature row come back identical. That is train/serve skew
measured rather than argued - and it is how D-31's quarterback-form skew was
found, since the composite failed exactly that test before it was fixed.

**The team-game table is built here, not read from disk (D-33).** The parquet
under ``data/processed/`` is a snapshot of the raw cache at the time
``nflpred.features.pbp_agg`` last ran, and the weekly run only refreshes the raw
cache. A snapshot from before last week's games made 2026 week 2 predict as if
week 1 had never happened - every rolling window, every Elo rating and every
projected starter one week stale - and nothing noticed, because the scores the
out-of-order guard reads come from the (fresh) schedule. Building the table from
the raw cache on every run takes well under a second and removes the snapshot
from the live path. :func:`_missing_history` stays as the net under that: it
refuses a week whose history lacks a completed game, whether the table was
passed in stale or the play-by-play for a finished game is not published yet.

**Documented limit.** Only a team's *next* unplayed week is buildable. Asking
for week 12 in August would need weeks 8-11 simulated to know what form each
team carries into it, and inventing that is a different project. It is refused
with a message naming the weeks in the way.

Run::

    python -m nflpred.predict --season 2026 --week 1 --explain
    python -m nflpred.predict --season 2025 --week 8 --json week8.json
    python -m nflpred.predict --settle
    python -m nflpred.predict --report
"""
