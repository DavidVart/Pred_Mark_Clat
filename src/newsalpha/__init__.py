"""NewsAlpha — high-velocity news-reactive trading.

Completely separate from the slow LLM bot. Different DB, different schema,
different strategy, different CLI command.

Pipeline:
    CoinbaseWS (BTC spot) ─┐
                           ├─→ Divergence Detector ─→ FastPredictor ─→ Executor
    PolymarketQuote (BTC) ─┘

Design goals:
    - Sub-3s end-to-end decision latency
    - Position hold times: seconds to minutes
    - Tight exits (time-based + rolling profit lock)
    - Mandatory flatten before market resolution
    - 10-30 trades/day target
"""
