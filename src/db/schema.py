"""SQLite table definitions."""

TABLES = {
    "markets": """
        CREATE TABLE IF NOT EXISTS markets (
            market_id TEXT PRIMARY KEY,
            platform TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            category TEXT DEFAULT '',
            yes_price REAL,
            no_price REAL,
            volume INTEGER DEFAULT 0,
            liquidity REAL DEFAULT 0,
            expiration TEXT,
            status TEXT DEFAULT 'active',
            outcomes TEXT DEFAULT '["Yes","No"]',
            url TEXT DEFAULT '',
            ticker TEXT,
            clob_token_ids TEXT,
            condition_id TEXT,
            last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "market_snapshots": """
        CREATE TABLE IF NOT EXISTS market_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT NOT NULL,
            platform TEXT NOT NULL,
            yes_price REAL,
            no_price REAL,
            volume INTEGER,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (market_id) REFERENCES markets(market_id)
        )
    """,
    "positions": """
        CREATE TABLE IF NOT EXISTS positions (
            position_id TEXT PRIMARY KEY,
            market_id TEXT NOT NULL,
            platform TEXT NOT NULL,
            title TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            quantity REAL NOT NULL,
            cost_basis REAL NOT NULL,
            current_price REAL DEFAULT 0,
            stop_loss REAL DEFAULT 0,
            take_profit REAL DEFAULT 0,
            is_paper INTEGER DEFAULT 1,
            category TEXT DEFAULT '',
            opened_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (market_id) REFERENCES markets(market_id)
        )
    """,
    "trade_log": """
        CREATE TABLE IF NOT EXISTS trade_log (
            trade_id TEXT PRIMARY KEY,
            market_id TEXT NOT NULL,
            platform TEXT NOT NULL,
            title TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL NOT NULL,
            quantity REAL NOT NULL,
            pnl REAL NOT NULL,
            pnl_pct REAL NOT NULL,
            is_paper INTEGER DEFAULT 1,
            category TEXT DEFAULT '',
            outcome TEXT DEFAULT '',
            failure_class TEXT DEFAULT '',
            opened_at TEXT,
            closed_at TEXT DEFAULT CURRENT_TIMESTAMP,
            hold_duration_hours REAL DEFAULT 0
        )
    """,
    "predictions": """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT NOT NULL,
            weighted_probability REAL,
            final_confidence REAL,
            disagreement_score REAL,
            edge REAL,
            models_succeeded INTEGER,
            models_failed INTEGER,
            total_cost_usd REAL,
            individual_json TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (market_id) REFERENCES markets(market_id)
        )
    """,
    "llm_queries": """
        CREATE TABLE IF NOT EXISTS llm_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            role TEXT NOT NULL,
            market_id TEXT,
            prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            cost_usd REAL DEFAULT 0,
            duration_ms INTEGER DEFAULT 0,
            success INTEGER DEFAULT 1,
            error TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "daily_stats": """
        CREATE TABLE IF NOT EXISTS daily_stats (
            date TEXT PRIMARY KEY,
            pnl REAL DEFAULT 0,
            trades_opened INTEGER DEFAULT 0,
            trades_closed INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            ai_cost REAL DEFAULT 0,
            max_drawdown REAL DEFAULT 0
        )
    """,
    "kill_switch": """
        CREATE TABLE IF NOT EXISTS kill_switch (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            active INTEGER DEFAULT 0,
            reason TEXT DEFAULT '',
            activated_at TEXT,
            activated_by TEXT DEFAULT 'system'
        )
    """,
    "analysis_cooldown": """
        CREATE TABLE IF NOT EXISTS analysis_cooldown (
            market_id TEXT NOT NULL,
            analyzed_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (market_id, analyzed_at)
        )
    """,
}

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_snapshots_market ON market_snapshots(market_id, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_positions_platform ON positions(platform)",
    "CREATE INDEX IF NOT EXISTS idx_trade_log_closed ON trade_log(closed_at)",
    "CREATE INDEX IF NOT EXISTS idx_predictions_market ON predictions(market_id, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_llm_queries_timestamp ON llm_queries(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_cooldown_market ON analysis_cooldown(market_id, analyzed_at)",
]
