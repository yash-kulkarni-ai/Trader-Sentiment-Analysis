# Trader Behavior & Market Sentiment Analysis
### Bitcoin Fear & Greed Index × Hyperliquid Historical Trader Data

---

## Dataset Overview

| Dataset | Records | Period |
|---|---|---|
| Hyperliquid Trader Data | 211,224 trades | May 2023 – May 2025 |
| Bitcoin Fear & Greed Index | 2,644 days | Feb 2018 – May 2025 |
| Unique Traders | 32 accounts | — |
| Unique Coins | 246 | — |

---

## Analyses Performed

1. **EDA** — Trade volume by sentiment regime + F&G index time series (2018–2025)
2. **PnL Distribution** — Histogram & boxplot by sentiment category
3. **Core Performance Metrics** — Mean PnL, Win Rate, Trade Volume by sentiment
4. **BUY vs SELL Dynamics** — Directional performance split by sentiment
5. **Top Trader Profiling** — Top 10 traders ranked by total PnL, compared vs others across sentiment regimes
6. **Coin × Sentiment Heatmap** — Top 15 coins × 5 sentiment categories (mean PnL)
7. **Temporal Patterns** — Mean PnL by day of week and hour of day (IST)
8. **Sentiment Momentum** — Day-over-day F&G change vs trader performance
9. **Monthly Trend** — Monthly total PnL coloured by dominant sentiment + win rate overlay
10. **Statistical Testing** — Kruskal-Wallis + Spearman correlation

---

## Key Findings

| # | Finding |
|---|---|
| 1 | **Extreme Greed delivers highest mean PnL ($130)** — 83% above Neutral ($71) |
| 2 | **Win rate is highest in Extreme Greed (89.2%) and Fear (87.3%)** — but Fear traders exit too early |
| 3 | **BUY dominates in Greed; SELL gaps close in Fear** — sentiment alignment is critical |
| 4 | **Top trader earned $2.14M (79.1% win rate)** — elite traders profit across ALL regimes |
| 5 | **Sharp F&G drops signal the best short opportunities** — panic creates dislocations |
| 6 | **Statistically confirmed**: Kruskal-Wallis H=730.33, p<0.0001 |

---

## Strategy Recommendations

- ✅ Scale up BUY exposure during Greed / Extreme Greed regimes
- ✅ Introduce SELL bias when F&G drops sharply day-over-day
- ✅ Don't cut winners early in Greed — let momentum run
- ✅ Reduce position size in Neutral — statistical edge is lowest
- ✅ HYPE & BTC show the most consistent cross-sentiment returns
- ✅ Monitor daily F&G delta as a leading signal, not just the absolute level

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scipy

# Place both CSVs in the same folder:
#   historical_data.csv
#   fear_greed_index.csv

# Open and run the notebook
jupyter notebook trader_sentiment_analysis.ipynb
# Kernel → Restart & Run All
```

All 9 figures are saved as PNG files automatically.

---

## Project Structure

```
├── trader_sentiment_analysis.ipynb   ← Main notebook
├── historical_data.csv               ← Hyperliquid trader data
├── fear_greed_index.csv              ← Bitcoin F&G Index
├── fig1_eda_overview.png
├── fig2_pnl_distribution.png
├── fig3_pnl_by_sentiment.png
├── fig4_buy_sell_sentiment.png
├── fig5_top_traders.png
├── fig6_coin_sentiment_heatmap.png
├── fig7_temporal_patterns.png
├── fig8_sentiment_momentum.png
├── fig9_monthly_pnl_trend.png
└── README.md
```

---

*Submitted for: Python Development Intern*  
*Analysis: EDA · Sentiment Analysis · Statistical Testing · Trading Strategy Insights*
