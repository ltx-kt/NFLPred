"""Feature construction. Nothing in here is leak-safe on its own.

``pbp_agg`` produces same-game team statistics — the raw substrate. Making those
safe to use as features is ``rolling``'s job: aggregate over prior games only,
then ``shift(1)`` within each team's time-ordered series before joining back to
a game row.
"""
