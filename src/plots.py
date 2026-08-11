'''
Visualization functions for DRL trading strategy experiments.

All figures are saved to the results/ directory using the non-interactive
Agg backend so plots work in headless environments.
'''
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

# Consistent colour cycle for all multi-strategy plots
_STRATEGY_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
]
_MODEL_COLORS = {'PPO': '#1f77b4', 'A2C': '#ff7f0e', 'DDPG': '#2ca02c'}


def _format_date_axis(ax: plt.Axes, freq: str = 'year') -> None:
    """Apply consistent date formatting to an axis."""
    if freq == 'year':
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')


def plot_portfolio_comparison(
    portfolio_dict: Dict[str, pd.Series],
    title: str = 'Portfolio Value Comparison',
    filename: str = 'portfolio_comparison.png',
) -> None:
    """
    Plot multiple portfolio value series on the same axis.

    Args:
        portfolio_dict: Mapping of strategy name → daily portfolio value Series.
            All Series should share a compatible DatetimeIndex.
        title: Figure title.
        filename: Output filename saved under results/.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, (name, series) in enumerate(portfolio_dict.items()):
        color = _STRATEGY_COLORS[idx % len(_STRATEGY_COLORS)]
        ax.plot(series.index, series.values, label=name, color=color, linewidth=1.5)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    _format_date_axis(ax, freq='year')

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_turbulence_and_portfolios(
    portfolio_with: pd.Series,
    portfolio_without: pd.Series,
    turbulence_series: pd.Series,
    threshold: float,
    filename: str = 'turbulence_comparison.png',
) -> None:
    """
    Dual-axis plot: portfolio values on left axis, turbulence index on right.

    Marks the turbulence threshold as a horizontal dashed line and highlights
    dates where turbulence exceeds the threshold (liquidation events).

    Args:
        portfolio_with: Portfolio value Series with turbulence control active.
        portfolio_without: Portfolio value Series without turbulence control.
        turbulence_series: Daily turbulence index Series.
        threshold: Turbulence threshold above which the agent de-risks.
        filename: Output filename saved under results/.
    """
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    # Portfolio values on left axis
    ax1.plot(portfolio_with.index, portfolio_with.values,
             label='With Turbulence Control', color='#1f77b4', linewidth=1.5, zorder=3)
    ax1.plot(portfolio_without.index, portfolio_without.values,
             label='Without Turbulence Control', color='#ff7f0e',
             linewidth=1.5, linestyle='--', zorder=3)
    ax1.set_ylabel('Portfolio Value ($)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Turbulence on right axis
    common_idx = turbulence_series.index
    ax2.fill_between(common_idx, turbulence_series.values, alpha=0.15,
                     color='#d62728', label='_nolegend_')
    ax2.plot(common_idx, turbulence_series.values,
             label='Turbulence Index', color='#d62728', linewidth=0.8, alpha=0.7)
    ax2.axhline(threshold, color='#d62728', linestyle=':', linewidth=1.5,
                label=f'Threshold ({threshold:.1f})')
    ax2.set_ylabel('Turbulence Index', color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')

    # Mark liquidation events
    liquidation_dates = turbulence_series.index[turbulence_series > threshold]
    if len(liquidation_dates) > 0:
        # Find value series to mark on (use portfolio_with)
        common = portfolio_with.index.intersection(liquidation_dates)
        if len(common) > 0:
            ax1.scatter(common, portfolio_with.loc[common].values,
                        color='red', s=8, zorder=4, label='Liquidation Event',
                        alpha=0.5)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    ax1.set_title('Portfolio Value with Turbulence Control', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.grid(True, alpha=0.3)
    _format_date_axis(ax1, freq='year')

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_ensemble_selection(
    selection_log: pd.DataFrame,
    filename: str = 'ensemble_selection.png',
) -> None:
    """
    Bar chart showing which model was selected at each quarterly rebalance.

    Args:
        selection_log: DataFrame with at least columns 'quarter_start' (or index)
            and 'selected_model' (one of 'PPO', 'A2C', 'DDPG').
        filename: Output filename saved under results/.
    """
    df = selection_log.copy()
    if 'selected_model' not in df.columns:
        raise ValueError("selection_log must contain a 'selected_model' column")

    # Use index as x-axis labels if no explicit date column
    if 'quarter_start' in df.columns:
        x_labels = df['quarter_start'].astype(str)
    else:
        x_labels = df.index.astype(str)

    models = df['selected_model'].values
    model_names = ['PPO', 'A2C', 'DDPG']

    fig, (ax_bar, ax_counts) = plt.subplots(
        1, 2, figsize=(14, 5),
        gridspec_kw={'width_ratios': [3, 1]}
    )

    # Bar chart: one bar per quarter coloured by selected model
    bar_colors = [_MODEL_COLORS.get(m, '#7f7f7f') for m in models]
    bars = ax_bar.bar(range(len(models)), [1] * len(models),
                      color=bar_colors, edgecolor='white', linewidth=0.5)

    ax_bar.set_xticks(range(len(models)))
    ax_bar.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax_bar.set_yticks([])
    ax_bar.set_title('Ensemble Model Selection by Quarter', fontsize=13, fontweight='bold')
    ax_bar.set_xlabel('Quarter Start')

    # Add text labels inside bars
    for i, (bar, model) in enumerate(zip(bars, models)):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, 0.5, model,
                    ha='center', va='center', fontsize=7, color='white', fontweight='bold')

    # Count summary pie/bar on right panel
    counts = {m: int(np.sum(np.array(models) == m)) for m in model_names}
    count_colors = [_MODEL_COLORS.get(m, '#7f7f7f') for m in model_names]
    ax_counts.bar(model_names, [counts[m] for m in model_names],
                  color=count_colors, edgecolor='white')
    ax_counts.set_title('Selection Count', fontsize=11)
    ax_counts.set_ylabel('Quarters Selected')
    for m, c in counts.items():
        ax_counts.text(model_names.index(m), c + 0.05, str(c),
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=_MODEL_COLORS[m], label=m) for m in model_names]
    ax_bar.legend(handles=legend_elements, loc='upper right', fontsize=9)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_metrics_table(
    metrics_df: pd.DataFrame,
    filename: str = 'metrics_table.png',
) -> None:
    """
    Render performance metrics as a matplotlib table figure.

    Args:
        metrics_df: DataFrame with strategies as rows and metric names as
            columns. Numeric values are formatted to 2–4 decimal places.
        filename: Output filename saved under results/.
    """
    fig, ax = plt.subplots(
        figsize=(max(8, len(metrics_df.columns) * 1.8), max(3, len(metrics_df) * 0.6 + 1.5))
    )
    ax.axis('off')

    # Format cells: percentages for return/drawdown, 4 decimals for Sharpe
    cell_text = []
    for _, row in metrics_df.iterrows():
        formatted = []
        for col, val in zip(metrics_df.columns, row):
            col_lower = col.lower()
            if isinstance(val, float):
                if any(k in col_lower for k in ('return', 'drawdown', 'vol')):
                    formatted.append(f'{val * 100:.2f}%')
                elif 'sharpe' in col_lower:
                    formatted.append(f'{val:.4f}')
                else:
                    formatted.append(f'{val:.4f}')
            else:
                formatted.append(str(val))
        cell_text.append(formatted)

    col_labels = list(metrics_df.columns)
    row_labels = list(metrics_df.index)

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    # Style header row
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#2c3e50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Alternate row shading
    for i in range(1, len(row_labels) + 1):
        color = '#f2f2f2' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            table[(i, j)].set_facecolor(color)

    ax.set_title('Strategy Performance Metrics', fontsize=14, fontweight='bold', pad=20)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_crash_period(
    portfolios: Dict[str, pd.Series],
    crash_start: str = '2020-01-01',
    crash_end: str = '2020-05-08',
    filename: str = 'crash_period.png',
) -> None:
    """
    Zoom-in plot of portfolio values during the 2020 COVID crash period.

    Args:
        portfolios: Mapping of strategy name → daily portfolio value Series.
        crash_start: Start date of the zoom window (inclusive).
        crash_end: End date of the zoom window (inclusive).
        filename: Output filename saved under results/.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, (name, series) in enumerate(portfolios.items()):
        sliced = series.loc[crash_start:crash_end].dropna()
        if sliced.empty:
            continue
        color = _STRATEGY_COLORS[idx % len(_STRATEGY_COLORS)]
        ax.plot(sliced.index, sliced.values, label=name, color=color, linewidth=1.5)

    ax.set_title(f'Portfolio Values: {crash_start} to {crash_end} (COVID-19 Crash)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)
    _format_date_axis(ax, freq='month')

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_drawdowns(
    portfolio_dict: Dict[str, pd.Series],
    filename: str = 'drawdowns.png',
) -> None:
    """
    Plot drawdown series for each strategy.

    Drawdown at time t is defined as:
        DD_t = (V_t - max(V_0, ..., V_t)) / max(V_0, ..., V_t)

    Args:
        portfolio_dict: Mapping of strategy name → daily portfolio value Series.
        filename: Output filename saved under results/.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, (name, series) in enumerate(portfolio_dict.items()):
        s = series.dropna()
        if s.empty:
            continue
        running_max = s.cummax()
        drawdown = (s - running_max) / running_max
        color = _STRATEGY_COLORS[idx % len(_STRATEGY_COLORS)]
        ax.plot(drawdown.index, drawdown.values, label=name, color=color, linewidth=1.2)
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.08, color=color)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
    ax.set_title('Portfolio Drawdowns', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)
    _format_date_axis(ax, freq='year')

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
