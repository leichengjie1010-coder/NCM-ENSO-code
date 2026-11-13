"""Draw NCM main Figure 6 from its saved final numerical tables."""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import numpy as np
from NCM_plot_utils import (
    ACC_ORDER,
    COLORS,
    DISPLAY,
    MONTHS,
    WARM_MODELS,
    plt,
    set_style,
    label_panel,
    save_figure,
    values,
    by_model,
    read_table,
    figure_arguments,
)


def forecast(rows):
    set_style(10.5, True)
    canvas_width, canvas_height = (11.2, 5.62)
    plot_width, plot_height = (7.444479166666667, 4.34625)
    fig = plt.figure(figsize=(canvas_width, canvas_height))
    ax = fig.add_axes(
        [
            0.8 / canvas_width,
            0.8 / canvas_height,
            plot_width / canvas_width,
            plot_height / canvas_height,
        ],
    )
    order = ['NCM', 'MMarkov-4', 'MMarkov-3']
    rows = [row for row in rows if row['model'] in order]
    months = sorted({row['target_month'] for row in rows})
    if len(months) != 6:
        raise ValueError('The 2026 figure needs six target months')
    handles = {}
    for name in order[3:] + order[1:3] + ['NCM']:
        group = by_model(rows, name, 'target_month')
        y = values(group, 'prediction_C')
        if y.shape != (6,) or not np.isfinite(y).all():
            raise ValueError(f'Incomplete 2026 trajectory for {name}')
        own = name == 'NCM'
        color = '#000000' if name == 'MMarkov-3' else COLORS[name]
        (line,) = ax.plot(
            range(6),
            y,
            color=color,
            lw=2.15 if own else 1.65 if name in order[1:3] else 1.35,
            ls='--' if name in order[1:3] else '-',
            marker='o',
            ms=5 if own else 3.6,
            mec='white',
            mew=0.35,
            zorder=5 if own else 4,
        )
        handles[name] = line
        if own:
            for i, value in enumerate(y):
                right, below = (i == 5, i in (0, 2, 3, 4))
                offset = (12, 0) if right else (0, -8 if below else 6)
                ax.annotate(
                    f'{value:.2f}',
                    (i, value),
                    xytext=offset,
                    textcoords='offset points',
                    ha='left' if right else 'center',
                    va='center' if right else 'top' if below else 'bottom',
                    fontsize=10,
                    color=COLORS['NCM'],
                    fontweight='bold',
                    zorder=6,
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.3},
                )
    ax.set(
        xlim=(-0.35, 5.35),
        ylim=(1, 5.55),
        xlabel='Target month',
        ylabel='Nino3.4 (°C)',
    )
    ax.set_xticks(range(6), months)
    ax.set_yticks(range(1, 6))
    ax.set_xlabel('Target month', fontsize=11.5, fontweight='bold', labelpad=7)
    ax.set_ylabel('Nino3.4 (°C)', fontsize=11.5, fontweight='bold', labelpad=8)
    ax.tick_params(axis='both', labelsize=10.5, pad=5)
    fig.legend(
        [handles[name] for name in order],
        order,
        ncol=1,
        loc='center left',
        bbox_to_anchor=(8.55 / canvas_width, (0.8 + plot_height / 2) / canvas_height),
        borderaxespad=0,
        fontsize=10.5,
        frameon=False,
        handlelength=2,
        handletextpad=0.65,
        borderpad=0.6,
        labelspacing=1.6,
    )
    return fig


def main():
    args = figure_arguments(__doc__)
    rows = [
        read_table(
            args.data / 'Fig6-Predictions.csv',
            ['model', 'prediction_C', 'target_month'],
        ),
    ]
    fig = forecast(rows[0])
    save_figure(fig, args.output, 6, args.dpi)
    print(f'Figure 6 written to {args.output}')
if __name__ == '__main__':
    main()
