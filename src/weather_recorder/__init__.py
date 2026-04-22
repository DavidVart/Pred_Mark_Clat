"""Weather edge recorder — empirical thesis validation.

Before committing to build a Kalshi weather trading bot, we measure whether
our NOAA-derived fair-value predictions actually beat Kalshi market pricing.

Every poll interval, we:
  1. Pull all active Kalshi weather markets + their current mid prices
  2. Pull NOAA forecasts for each covered city
  3. Compute our model's predicted probability for each market
  4. Log the snapshot (market + our_pred + noaa_high + timestamp)

When markets resolve, we add outcome (YES/NO) to the log.

Analysis script computes:
  - Brier score: Σ(predicted - actual)² / N  (lower = better)
  - Comparison: our_Brier vs market_Brier on same set of resolved markets

If our_Brier < market_Brier → we have edge, build the bot
Else → we don't, pivot
"""
