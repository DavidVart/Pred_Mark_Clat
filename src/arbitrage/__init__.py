"""Cross-platform arbitrage between Polymarket and Kalshi.

Core idea: the same real-world event (e.g. "BTC above $100k on Apr 30") often
trades at slightly different prices on Polymarket and Kalshi because the two
user bases don't overlap and capital doesn't flow freely between them
(Polymarket = crypto USDC, Kalshi = US-regulated CeFi USD).

When the synthetic basket (YES on one + NO on the other) costs less than $1,
we capture a deterministic profit at resolution.

Pipeline:
    MarketPairRegistry → Scanner → Opportunity → TwoLegExecutor → Reconciler
"""
