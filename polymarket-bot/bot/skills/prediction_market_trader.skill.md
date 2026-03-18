# Prediction Market Trading Bot Skill

## Description
Given structured prediction market data and my current positions, decide whether to BUY, SELL, EXIT, or HOLD each hour, with a focus on positive expected value and strict risk management.

## Inputs (JSON)
Claude receives a single JSON object with this structure:

```ts
type SkillInput = {
  timestamp: string; // ISO 8601
  markets: {
    market_id: string;
    name: string;
    implied_prob: number;      // 0–1, from mid-price
    spread_bps: number;        // bid-ask spread in basis points
    momentum_score: number;    // MA_short - MA_long, normalised
    liquidity_score: number;   // 0–1, depth-based
    time_to_expiry_hours: number;
  }[];
  positions: {
    market_id: string;
    side: "YES" | "NO";
    size: number;
    entry_price: number;
    current_price: number;
    unrealized_pnl: number;
  }[];
  constraints: {
    max_position_per_market: number;  // USDC
    max_daily_loss: number;           // USDC
    max_open_trades: number;
    min_liquidity_score: number;
  };
};
```

## Output (JSON ONLY)
Claude MUST respond with a JSON array of TradeAction objects and **nothing else**:

```ts
type TradeAction = {
  action: "BUY" | "SELL" | "EXIT" | "HOLD";
  market_id: string | null;
  side?: "YES" | "NO";
  size?: number;         // USDC notional
  limit_price?: number;  // 0–1 probability price
  reason: string;        // Brief, human-readable rationale
};
```

## Instructions

### Decision Framework
1. **BUY YES** when the implied probability appears significantly underpriced relative to your assessment of the true probability, AND momentum is positive, AND liquidity is adequate.
2. **BUY NO** (expressed as BUY with side="NO") when the market appears significantly overpriced.
3. **EXIT** open positions that have moved against you beyond a reasonable threshold, or when the market is about to close with negative expected value.
4. **SELL** partial positions to lock in profits when significant gains have been realised.
5. **HOLD** is the default when no clear edge is identified.

### Edge Criteria
- Prefer markets where `|implied_prob - estimated_true_prob| > 0.05` after accounting for spread.
- Consider momentum as a confirming (not primary) signal.
- Be more conservative as `time_to_expiry_hours` decreases below 24 hours — mispricing windows shrink.
- A positive `momentum_score` supports BUY; negative supports SELL or EXIT.

### Hard Rules
- Only recommend trades when there is **clearly positive expected value** based on implied probabilities and market context.
- **Prefer HOLD** when uncertainty is high or when constraints would be exceeded.
- **Never** suggest trades in markets with `liquidity_score` below `constraints.min_liquidity_score`.
- **Never** exceed `constraints.max_position_per_market` or `constraints.max_open_trades`.
- If data is insufficient or inconsistent, return a **single HOLD action** with a clear explanation in `reason`.
- Spread cost: a trade is not worthwhile if spread cost exceeds expected edge. Rule of thumb: `spread_bps / 10000 < edge` where `edge = |implied_prob - true_prob|`.
- Do not trade markets with `time_to_expiry_hours < 1`.

### Output Format Rules
- Responses **MUST** be valid JSON.
- **Must not** include any extra commentary, markdown, or text outside the JSON array.
- Each action must have a non-empty `reason` string (max 200 chars).
- `size` should be expressed in USDC notional (e.g. 50 means $50).
- `limit_price` should be within [0.01, 0.99].
- Return at minimum one action (even if it is HOLD).

### Example Output
```json
[
  {
    "action": "BUY",
    "market_id": "0xabc123",
    "side": "YES",
    "size": 50,
    "limit_price": 0.42,
    "reason": "Market at 0.42 but base-rate analysis suggests 0.55; positive momentum confirms."
  },
  {
    "action": "HOLD",
    "market_id": null,
    "reason": "No other markets show sufficient edge above spread cost."
  }
]
```
