"""Stage 3: Market Predictor — run ensemble and compute edge."""

from __future__ import annotations

import json

from src.agents.ensemble import EnsembleRunner
from src.db.manager import DatabaseManager
from src.models.market import UnifiedMarket
from src.models.prediction import EnsembleResult
from src.pipeline.calibration import PlattScaler
from src.pipeline.fees import net_edge
from src.utils.cost_tracker import CostTracker
from src.utils.logging import get_logger

logger = get_logger("predictor")


class MarketPredictor:
    """Runs the AI ensemble on candidate markets and computes trading edge."""

    def __init__(
        self,
        ensemble: EnsembleRunner,
        db: DatabaseManager,
        cost_tracker: CostTracker,
        min_edge: float = 0.05,
        min_net_edge: float = 0.03,
        min_confidence: float = 0.45,
        max_analyses_per_day: int = 4,
        prefer_maker_orders: bool = True,
        calibrator: PlattScaler | None = None,
    ):
        self.ensemble = ensemble
        self.db = db
        self.cost_tracker = cost_tracker
        self.min_edge = min_edge
        self.min_net_edge = min_net_edge
        self.min_confidence = min_confidence
        self.max_analyses_per_day = max_analyses_per_day
        self.prefer_maker_orders = prefer_maker_orders
        # Platt scaler fit on past resolved trades. Identity scaler = no calibration.
        self.calibrator = calibrator or PlattScaler.identity()

    async def predict(self, market: UnifiedMarket, context: dict) -> EnsembleResult | None:
        """
        Run ensemble prediction on a market.
        Returns None if budget exhausted, on cooldown, or below thresholds.
        """
        # Check AI budget
        if not await self.cost_tracker.can_spend(0.50):
            logger.warning("ai_budget_exhausted")
            return None

        # Check daily analysis limit for this market
        daily_count = await self.db.get_daily_analysis_count(market.market_id)
        if daily_count >= self.max_analyses_per_day:
            logger.debug("max_analyses_reached", market=market.title[:50], count=daily_count)
            return None

        # Run ensemble
        result = await self.ensemble.predict(market, context)

        # Apply Platt calibration if we have a fitted scaler.
        # This replaces the raw ensemble probability with a calibrated one
        # and recomputes edge accordingly. If the scaler is identity (no history
        # yet), this is a no-op.
        if not self.calibrator.is_identity:
            raw_prob = result.weighted_probability
            calibrated = self.calibrator.apply(raw_prob)
            new_edge = calibrated - market.yes_price
            logger.debug(
                "calibration_applied",
                market=market.title[:50],
                raw=round(raw_prob, 3),
                calibrated=round(calibrated, 3),
                edge_before=round(result.edge, 3),
                edge_after=round(new_edge, 3),
            )
            result = result.model_copy(update={
                "weighted_probability": calibrated,
                "edge": new_edge,
            })

        # Record costs
        for pred in result.individual_predictions:
            await self.cost_tracker.record_spend(
                model_name=pred.model_name,
                role=pred.role,
                market_id=market.market_id,
                cost_usd=pred.cost_usd,
                tokens=pred.tokens_used,
            )

        # Record cooldown
        await self.db.record_analysis(market.market_id)

        # Log prediction to DB
        await self.db.log_prediction({
            "market_id": market.market_id,
            "weighted_probability": result.weighted_probability,
            "final_confidence": result.final_confidence,
            "disagreement_score": result.disagreement_score,
            "edge": result.edge,
            "models_succeeded": result.models_succeeded,
            "models_failed": result.models_failed,
            "total_cost_usd": result.total_cost_usd,
            "individual_json": json.dumps([
                {
                    "model": p.model_name,
                    "role": p.role,
                    "probability": p.probability,
                    "confidence": p.confidence,
                    "reasoning": p.reasoning[:200],
                }
                for p in result.individual_predictions
            ]),
        })

        # Check thresholds
        if not result.has_consensus:
            logger.info("no_consensus", market=market.title[:50], models=result.models_succeeded)
            return None

        if result.abs_edge < self.min_edge:
            logger.info(
                "edge_too_small_gross",
                market=market.title[:50],
                edge=f"{result.edge:+.3f}",
                threshold=self.min_edge,
            )
            return None

        # NET edge check: subtract estimated round-trip fees. Assumes we hold
        # to resolution (winning contracts settle at $1 with no exit fee) and
        # that we submit a maker limit if configured.
        entry_price = market.yes_price if result.trade_side == "yes" else market.no_price
        net, fees = net_edge(
            gross_edge=result.abs_edge,
            platform=market.platform,
            category=market.category,
            price=entry_price,
            is_maker_entry=self.prefer_maker_orders,
            settles_at_resolution=True,
        )
        if net < self.min_net_edge:
            logger.info(
                "edge_too_small_net",
                market=market.title[:50],
                gross=f"{result.abs_edge:.3f}",
                fees=f"{fees.round_trip_pct:.3f}",
                net=f"{net:.3f}",
                threshold=self.min_net_edge,
            )
            return None

        if result.final_confidence < self.min_confidence:
            logger.info(
                "confidence_too_low",
                market=market.title[:50],
                confidence=f"{result.final_confidence:.3f}",
                threshold=self.min_confidence,
            )
            return None

        logger.info(
            "prediction_signal",
            market=market.title[:50],
            side=result.trade_side,
            edge_gross=f"{result.edge:+.3f}",
            edge_net=f"{net:+.3f}",
            fees=f"{fees.round_trip_pct:.3f}",
            confidence=f"{result.final_confidence:.3f}",
            probability=f"{result.weighted_probability:.3f}",
            market_price=f"{market.yes_price:.3f}",
        )

        return result
