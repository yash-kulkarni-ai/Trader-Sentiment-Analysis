# =============================================================================
# TRADER BEHAVIOR & MARKET SENTIMENT ANALYSIS
# Bitcoin Fear & Greed Index x Hyperliquid Historical Trader Data
# =============================================================================

# =============================================================================
# SECTION 1 — IMPORTS
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# Plot styling
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.family'] = 'DejaVu Sans'

# Sentiment order and colors used throughout
SENTIMENT_ORDER = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
SENTIMENT_COLORS = {
    'Extreme Fear': '#d32f2f',
    'Fear':         '#f57c00',
    'Neutral':      '#fbc02d',
    'Greed':        '#388e3c',
    'Extreme Greed':'#1b5e20'
}

print('Libraries loaded.')


# =============================================================================
# SECTION 2 — LOAD DATA
# =============================================================================

TRADER_PATH    = 'historical_data.csv'    # <-- change path if needed
SENTIMENT_PATH = 'fear_greed_index.csv'   # <-- change path if needed

trades = pd.read_csv(TRADER_PATH)
fg     = pd.read_csv(SENTIMENT_PATH)

print(f'Trader data  : {trades.shape[0]:,} rows x {trades.shape[1]} columns')
print(f'Fear & Greed : {fg.shape[0]:,} rows x {fg.shape[1]} columns')
print('Trader columns:', trades.columns.tolist())


# =============================================================================
# SECTION 3 — DATA CLEANING & PREPROCESSING
# =============================================================================

# ── Fear & Greed ─────────────────────────────────────────────────────────────
fg['date'] = pd.to_datetime(fg['date'])
fg = fg.rename(columns={'classification': 'sentiment', 'value': 'fg_value'})

# ── Trader data ───────────────────────────────────────────────────────────────
# Timestamp IST has mixed formats like '2/12/2024 22:50' and '18-03-2025 12:50'
trades['date'] = pd.to_datetime(
    trades['Timestamp IST'], format='mixed', dayfirst=True
).dt.normalize()   # .normalize() strips the time part, keeps only the date

trades['Side'] = trades['Side'].str.upper().str.strip()

# ── Merge trader data with sentiment on date ──────────────────────────────────
df = trades.merge(fg[['date', 'fg_value', 'sentiment']], on='date', how='left')

# Make sentiment an ordered category so groupby always shows in correct order
df['sentiment'] = pd.Categorical(
    df['sentiment'], categories=SENTIMENT_ORDER, ordered=True
)

# Work only with closed trades (non-zero PnL) for performance analysis
closed = df[df['Closed PnL'] != 0].copy()

print(f'Date range        : {trades["date"].min().date()} to {trades["date"].max().date()}')
print(f'Sentiment match   : {df["sentiment"].notna().mean()*100:.1f}%')
print(f'Total trades      : {len(trades):,}')
print(f'Closed PnL trades : {len(closed):,}')
print(f'Unique accounts   : {trades["Account"].nunique()}')
print(f'Unique coins      : {trades["Coin"].nunique()}')
print('Sentiment distribution:')
print(df['sentiment'].value_counts().reindex(SENTIMENT_ORDER))


# =============================================================================
# SECTION 4 — EDA: FIGURE 1 — Trade Volume + F&G Time Series
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Left chart: bar chart of trade volume per sentiment
counts = df['sentiment'].value_counts().reindex(SENTIMENT_ORDER)
bars = axes[0].bar(
    counts.index, counts.values,
    color=[SENTIMENT_COLORS[s] for s in counts.index],
    edgecolor='white'
)
axes[0].set_title('Trade Volume by Sentiment Regime', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Sentiment')
axes[0].set_ylabel('Number of Trades')
for bar, val in zip(bars, counts.values):
    axes[0].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 200,
        f'{val:,}', ha='center', fontsize=9
    )

# Right chart: F&G index over time with fear/greed shading
daily_fg = fg.set_index('date')['fg_value'].sort_index()
daily_fg.plot(ax=axes[1], color='#90caf9', linewidth=0.6, alpha=0.5, label='Daily')
daily_fg.rolling(30).mean().plot(
    ax=axes[1], color='#1565c0', linewidth=2, label='30-day MA'
)
axes[1].axhline(50, color='gray', linestyle='--', linewidth=0.8, label='Neutral (50)')
axes[1].fill_between(
    daily_fg.index, daily_fg, 50,
    where=(daily_fg < 50), alpha=0.08, color='red'
)
axes[1].fill_between(
    daily_fg.index, daily_fg, 50,
    where=(daily_fg > 50), alpha=0.08, color='green'
)
axes[1].set_title('Bitcoin Fear & Greed Index (2018-2025)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('F&G Value (0-100)')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig('fig1_eda_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print('Fig 1 saved.')


# =============================================================================
# SECTION 5 — FIGURE 2 — PnL Distribution
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Left: histogram of all closed PnL values (clipped so extreme outliers don't squash the chart)
clipped = closed['Closed PnL'].clip(-5000, 5000)
axes[0].hist(clipped, bins=120, color='#1565c0', edgecolor='none', alpha=0.85)
axes[0].axvline(0, color='red', linestyle='--', linewidth=1.2, label='Zero')
axes[0].axvline(
    closed['Closed PnL'].mean(), color='orange', linestyle='--', linewidth=1.2,
    label=f'Mean: ${closed["Closed PnL"].mean():.2f}'
)
axes[0].set_title('Closed PnL Distribution (clipped +/-$5K)', fontweight='bold')
axes[0].set_xlabel('Closed PnL (USD)')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# Right: boxplot per sentiment
bp_data = [
    closed[closed['sentiment'] == s]['Closed PnL'].clip(-3000, 3000).dropna()
    for s in SENTIMENT_ORDER
]
bp = axes[1].boxplot(
    bp_data, labels=SENTIMENT_ORDER, patch_artist=True,
    medianprops=dict(color='white', linewidth=2),
    flierprops=dict(marker='.', markersize=2, alpha=0.3)
)
for patch, s in zip(bp['boxes'], SENTIMENT_ORDER):
    patch.set_facecolor(SENTIMENT_COLORS[s])
    patch.set_alpha(0.85)
axes[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
axes[1].set_title('PnL Distribution by Sentiment (clipped +/-$3K)', fontweight='bold')
axes[1].set_ylabel('Closed PnL (USD)')
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('fig2_pnl_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print('Fig 2 saved.')


# =============================================================================
# SECTION 6 — CORE ANALYSIS: PnL Stats by Sentiment
# =============================================================================

pnl_stats = closed.groupby('sentiment', observed=True)['Closed PnL'].agg(
    mean_pnl   = 'mean',
    median_pnl = 'median',
    total_pnl  = 'sum',
    trade_count= 'count',
    std_pnl    = 'std',
    win_rate   = lambda x: (x > 0).mean() * 100
).reindex(SENTIMENT_ORDER)

# Sharpe proxy = mean / std (higher = better risk-adjusted return)
pnl_stats['sharpe_proxy'] = pnl_stats['mean_pnl'] / pnl_stats['std_pnl']

print('=== PnL Summary by Sentiment ===')
print(pnl_stats.round(3).to_string())


# =============================================================================
# SECTION 7 — FIGURE 3 — Mean PnL, Win Rate, Trade Volume
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colors = [SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER]

# Mean PnL
b0 = axes[0].bar(pnl_stats.index, pnl_stats['mean_pnl'], color=colors, edgecolor='white')
axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
axes[0].set_title('Mean Closed PnL by Sentiment', fontweight='bold')
axes[0].set_ylabel('Mean PnL (USD)')
axes[0].tick_params(axis='x', rotation=20)
for bar, val in zip(b0, pnl_stats['mean_pnl']):
    axes[0].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + (2 if val >= 0 else -8),
        f'${val:.1f}', ha='center', fontsize=8
    )

# Win Rate
b1 = axes[1].bar(pnl_stats.index, pnl_stats['win_rate'], color=colors, edgecolor='white')
axes[1].axhline(50, color='black', linewidth=0.8, linestyle='--', label='50% baseline')
axes[1].set_title('Win Rate by Sentiment', fontweight='bold')
axes[1].set_ylabel('Win Rate (%)')
axes[1].set_ylim(0, 100)
axes[1].tick_params(axis='x', rotation=20)
axes[1].legend(fontsize=8)
for bar, val in zip(b1, pnl_stats['win_rate']):
    axes[1].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 1,
        f'{val:.1f}%', ha='center', fontsize=8
    )

# Trade Volume
b2 = axes[2].bar(pnl_stats.index, pnl_stats['trade_count'], color=colors, edgecolor='white')
axes[2].set_title('Closed Trade Volume by Sentiment', fontweight='bold')
axes[2].set_ylabel('Number of Closed Trades')
axes[2].tick_params(axis='x', rotation=20)
for bar, val in zip(b2, pnl_stats['trade_count']):
    axes[2].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 50,
        f'{val:,}', ha='center', fontsize=8
    )

plt.suptitle(
    'Trader Performance Across Market Sentiment Regimes',
    fontsize=14, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig('fig3_pnl_by_sentiment.png', dpi=150, bbox_inches='tight')
plt.show()
print('Fig 3 saved.')


# =============================================================================
# SECTION 8 — FIGURE 4 — BUY vs SELL by Sentiment
# =============================================================================

side_stats = closed.groupby(['sentiment', 'Side'], observed=True)['Closed PnL'].agg(
    mean_pnl = 'mean',
    win_rate = lambda x: (x > 0).mean() * 100,
    count    = 'count'
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, metric, title, ylabel, ref in zip(
    axes,
    ['mean_pnl', 'win_rate'],
    ['Mean PnL: BUY vs SELL by Sentiment', 'Win Rate: BUY vs SELL by Sentiment'],
    ['Mean PnL (USD)', 'Win Rate (%)'],
    [0, 50]
):
    pivot = side_stats.pivot(
        index='sentiment', columns='Side', values=metric
    ).reindex(SENTIMENT_ORDER)
    pivot.plot(kind='bar', ax=ax, edgecolor='white', width=0.7,
               color=['#1565c0', '#c62828'])
    ax.axhline(ref, color='black', linestyle='--', linewidth=0.8)
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=15)
    ax.legend(title='Side', fontsize=9)

plt.tight_layout()
plt.savefig('fig4_buy_sell_sentiment.png', dpi=150, bbox_inches='tight')
plt.show()
print('Fig 4 saved.')


# =============================================================================
# SECTION 9 — TOP TRADER PROFILING
# =============================================================================

trader_summary = closed.groupby('Account')['Closed PnL'].agg(
    total_pnl  = 'sum',
    mean_pnl   = 'mean',
    trade_count= 'count',
    win_rate   = lambda x: (x > 0).mean() * 100
).reset_index()

# Only include traders with at least 10 trades (removes noise)
trader_summary = trader_summary[
    trader_summary['trade_count'] >= 10
].sort_values('total_pnl', ascending=False)

top10 = trader_summary.head(10)

print(f'Qualified traders (>=10 trades): {len(trader_summary)}')
print('\n-- Top 10 Traders by Total PnL --')
print(top10[['Account', 'total_pnl', 'win_rate', 'trade_count']].round(2).to_string(index=False))


# =============================================================================
# SECTION 10 — FIGURE 5 — Top Traders Chart
# =============================================================================

top_accts = set(top10['Account'])
top_tr    = closed[closed['Account'].isin(top_accts)]
other_tr  = closed[~closed['Account'].isin(top_accts)]

# Compare mean PnL of top 10 vs everyone else, per sentiment
compare = pd.DataFrame({
    'Top 10 Traders': top_tr.groupby('sentiment', observed=True)['Closed PnL'].mean().reindex(SENTIMENT_ORDER),
    'All Others':     other_tr.groupby('sentiment', observed=True)['Closed PnL'].mean().reindex(SENTIMENT_ORDER)
})

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Top 10 horizontal bar chart
top10_plot = top10.copy()
top10_plot['label'] = top10_plot['Account'].str[:10] + '...'
axes[0].barh(top10_plot['label'], top10_plot['total_pnl'],
             color='#1b5e20', edgecolor='white')
axes[0].axvline(0, color='black', linewidth=0.8, linestyle='--')
axes[0].set_title('Top 10 Traders — Total Closed PnL', fontweight='bold')
axes[0].set_xlabel('Total PnL (USD)')
axes[0].invert_yaxis()

# Grouped bar: top 10 vs others per sentiment
compare.plot(kind='bar', ax=axes[1], edgecolor='white', width=0.7,
             color=['#1b5e20', '#b71c1c'])
axes[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
axes[1].set_title('Mean PnL: Top 10 vs All Others by Sentiment', fontweight='bold')
axes[1].set_ylabel('Mean PnL (USD)')
axes[1].tick_params(axis='x', rotation=15)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig('fig5_top_traders.png', dpi=150, bbox_inches='tight')
plt.show()
print('Fig 5 saved.')


# =============================================================================
# SECTION 11 — FIGURE 6 — Coin x Sentiment Heatmap
# =============================================================================

# Top 15 most traded coins
top_coins = closed['Coin'].value_counts().head(15).index.tolist()

# Mean PnL for each coin x sentiment combination
coin_pnl = (
    closed[closed['Coin'].isin(top_coins)]
    .groupby(['Coin', 'sentiment'], observed=True)['Closed PnL'].mean()
    .unstack('sentiment')
    .reindex(columns=SENTIMENT_ORDER)
)

plt.figure(figsize=(14, 7))
sns.heatmap(
    coin_pnl, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
    linewidths=0.5, cbar_kws={'label': 'Mean PnL (USD)'}, annot_kws={'size': 9}
)
plt.title('Mean PnL Heatmap: Top 15 Coins x Market Sentiment',
          fontsize=13, fontweight='bold')
plt.xlabel('Market Sentiment')
plt.ylabel('Coin')
plt.tight_layout()
plt.savefig('fig6_coin_sentiment_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print('Fig 6 saved.')


# =============================================================================
# SECTION 12 — FIGURE 7 — Temporal Patterns (Day & Hour)
# =============================================================================

closed = closed.copy()
closed['hour'] = pd.to_datetime(
    closed['Timestamp IST'], format='mixed', dayfirst=True
).dt.hour
closed['day_of_week'] = closed['date'].dt.day_name()

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Mean PnL by day of week
dow_pnl = closed.groupby('day_of_week')['Closed PnL'].mean().reindex(day_order)
axes[0].bar(
    dow_pnl.index, dow_pnl.values,
    color=['#c62828' if v < 0 else '#2e7d32' for v in dow_pnl],
    edgecolor='white'
)
axes[0].axhline(0, color='black', linestyle='--', linewidth=0.8)
axes[0].set_title('Mean PnL by Day of Week', fontweight='bold')
axes[0].set_ylabel('Mean PnL (USD)')
axes[0].tick_params(axis='x', rotation=30)

# Mean PnL by hour of day
hour_pnl = closed.groupby('hour')['Closed PnL'].mean()
axes[1].bar(
    hour_pnl.index, hour_pnl.values,
    color=['#c62828' if v < 0 else '#2e7d32' for v in hour_pnl],
    edgecolor='white', width=0.8
)
axes[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
axes[1].set_title('Mean PnL by Hour of Day (IST)', fontweight='bold')
axes[1].set_ylabel('Mean PnL (USD)')
axes[1].set_xlabel('Hour (IST)')
axes[1].set_xticks(range(0, 24, 2))

plt.tight_layout()
plt.savefig('fig7_temporal_patterns.png', dpi=150, bbox_inches='tight')
plt.show()
print('Fig 7 saved.')


# =============================================================================
# SECTION 13 — FIGURE 8 — Sentiment Momentum Analysis
# =============================================================================

# Calculate day-over-day change in F&G value
fg_sorted = fg.sort_values('date').copy()
fg_sorted['fg_change'] = fg_sorted['fg_value'].diff()

# Merge that change into closed trades
df2 = closed.merge(fg_sorted[['date', 'fg_change']], on='date', how='left')

# Bin the change into 5 categories
df2['sentiment_shift'] = pd.cut(
    df2['fg_change'],
    bins=[-100, -15, -5, 5, 15, 100],
    labels=['Sharp Drop', 'Slight Drop', 'Stable', 'Slight Rise', 'Sharp Rise']
)

shift_pnl = df2.groupby('sentiment_shift', observed=True)['Closed PnL'].agg(
    mean_pnl = 'mean',
    win_rate = lambda x: (x > 0).mean() * 100,
    count    = 'count'
).dropna()

shift_colors = ['#b71c1c', '#e57373', '#ffd54f', '#81c784', '#1b5e20']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].bar(shift_pnl.index, shift_pnl['mean_pnl'],
            color=shift_colors[:len(shift_pnl)], edgecolor='white')
axes[0].axhline(0, color='black', linestyle='--', linewidth=0.8)
axes[0].set_title('Mean PnL by F&G Day-over-Day Change', fontweight='bold')
axes[0].set_ylabel('Mean PnL (USD)')
axes[0].tick_params(axis='x', rotation=15)

axes[1].bar(shift_pnl.index, shift_pnl['win_rate'],
            color=shift_colors[:len(shift_pnl)], edgecolor='white')
axes[1].axhline(50, color='black', linestyle='--', linewidth=0.8)
axes[1].set_title('Win Rate by F&G Day-over-Day Change', fontweight='bold')
axes[1].set_ylabel('Win Rate (%)')
axes[1].set_ylim(0, 100)
axes[1].tick_params(axis='x', rotation=15)

plt.suptitle('Impact of Sentiment Momentum on Trader Performance',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig8_sentiment_momentum.png', dpi=150, bbox_inches='tight')
plt.show()
print('Fig 8 saved.')


# =============================================================================
# SECTION 14 — FIGURE 9 — Monthly PnL Trend
# =============================================================================

closed['month'] = closed['date'].dt.to_period('M')

monthly = closed.groupby('month').agg(
    total_pnl = ('Closed PnL', 'sum'),
    win_rate  = ('Closed PnL', lambda x: (x > 0).mean() * 100)
).reset_index()
monthly['month_dt'] = monthly['month'].dt.to_timestamp()

# Find dominant sentiment for each month
dom_sent = (
    df.groupby(df['date'].dt.to_period('M'))['sentiment']
    .agg(lambda x: x.value_counts().idxmax())
    .reset_index()
    .rename(columns={'date': 'month'})
)
monthly = monthly.merge(dom_sent, on='month', how='left')

fig, ax1 = plt.subplots(figsize=(16, 6))
bar_colors = [SENTIMENT_COLORS.get(str(s), '#999') for s in monthly['sentiment']]

ax1.bar(monthly['month_dt'], monthly['total_pnl'],
        color=bar_colors, edgecolor='white', width=20, alpha=0.85)
ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax1.set_title('Monthly Total Closed PnL (coloured by Dominant Sentiment)',
              fontsize=13, fontweight='bold')
ax1.set_ylabel('Total PnL (USD)')
ax1.set_xlabel('Month')

# Overlay win rate on secondary axis
ax2 = ax1.twinx()
ax2.plot(monthly['month_dt'], monthly['win_rate'],
         color='#1565c0', linewidth=2, marker='o', markersize=5, label='Win Rate')
ax2.set_ylabel('Win Rate (%)', color='#1565c0')
ax2.tick_params(axis='y', labelcolor='#1565c0')
ax2.set_ylim(0, 100)

legend_elements = [Patch(facecolor=SENTIMENT_COLORS[s], label=s) for s in SENTIMENT_ORDER]
ax1.legend(handles=legend_elements, fontsize=8, loc='upper left')
ax2.legend(fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig('fig9_monthly_pnl_trend.png', dpi=150, bbox_inches='tight')
plt.show()
print('Fig 9 saved.')


# =============================================================================
# SECTION 15 — STATISTICAL TESTS
# =============================================================================

# --- Kruskal-Wallis Test ---
# Tests whether PnL distributions are significantly different across sentiment groups
# We use Kruskal-Wallis (not ANOVA) because PnL data is skewed, not normally distributed
groups = [
    closed[closed['sentiment'] == s]['Closed PnL'].dropna().values
    for s in SENTIMENT_ORDER
]
stat, p = kruskal(*groups)

print('=== Kruskal-Wallis Test ===')
print(f'H-statistic : {stat:.2f}')
print(f'p-value     : {p:.2e}')
print(f'Result      : {"SIGNIFICANT — sentiment affects PnL (p < 0.05)" if p < 0.05 else "Not significant"}')

# --- Spearman Correlation ---
# Tests monotonic relationship between F&G value and PnL
# We use Spearman (not Pearson) because PnL has outliers
tmp = closed[closed['fg_value'].notna()]
rho, p_rho = stats.spearmanr(tmp['fg_value'], tmp['Closed PnL'])

print('\n=== Spearman Correlation (F&G value vs Closed PnL) ===')
print(f'rho   : {rho:.4f}')
print(f'p     : {p_rho:.2e}')
print(f'Meaning: {"Positive" if rho > 0 else "Negative"} correlation — '
      f'traders earn {"more" if rho > 0 else "less"} as market greed increases')


# =============================================================================
# SECTION 16 — FULL SUMMARY STATISTICS
# =============================================================================

print('=' * 65)
print('  FINAL DATASET SUMMARY')
print('=' * 65)
print(f'  Total trade records     : {len(trades):,}')
print(f'  Unique traders          : {trades["Account"].nunique()}')
print(f'  Unique coins traded     : {trades["Coin"].nunique()}')
print(f'  Date range              : {trades["date"].min().date()} to {trades["date"].max().date()}')
print(f'  Closed PnL trades       : {len(closed):,}')
print(f'  Total closed PnL        : ${closed["Closed PnL"].sum():,.2f}')
print(f'  Overall win rate        : {(closed["Closed PnL"] > 0).mean()*100:.1f}%')
print(f'  Mean PnL per trade      : ${closed["Closed PnL"].mean():.4f}')
print(f'  Median PnL per trade    : ${closed["Closed PnL"].median():.4f}')
print(f'  PnL std deviation       : ${closed["Closed PnL"].std():,.2f}')
print()
print('  SENTIMENT BREAKDOWN IN TRADE RECORDS')
print('  ' + '-' * 48)
for s in SENTIMENT_ORDER:
    cnt = (df['sentiment'] == s).sum()
    pct = cnt / len(df) * 100
    print(f'  {s:<15}: {cnt:>7,} trades  ({pct:.1f}%)')
print('=' * 65)

print('\nAll 9 figures saved successfully!')
