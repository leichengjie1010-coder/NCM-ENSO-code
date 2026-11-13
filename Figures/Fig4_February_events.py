"""Draw NCM main Figure 4 from its saved final numerical tables."""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import numpy as np
import math
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


def warm_events(event_rows, mae_rows, period='winter'):
    if period not in ('winter', 'february'):
        raise ValueError('Expected winter or february event period')
    set_style(18, bold=True)
    fig = plt.figure(figsize=(19.8, 23.2), facecolor='white')
    canvas_width, canvas_height = fig.get_size_inches()

    def axes_inches(left, bottom, width, height):
        return fig.add_axes(
            [
                left / canvas_width,
                bottom / canvas_height,
                width / canvas_width,
                height / canvas_height,
            ],
        )
    legends = []
    for year, bottom, bar_left, letters in (
        (2015, 16.35, 1.25, 'ac'),
        (2023, 9.05, 11.2, 'bd'),
    ):
        ax = axes_inches(1.25, bottom, 11.55, 5.8)
        bx = axes_inches(bar_left, 2.05, 7.6, 5.05)
        rows = [row for row in event_rows if int(row['event']) == year]
        available = (
            {row['model'] for row in rows}
            - {'NCM', 'ORAS5', 'MMarkov-4', 'MMarkov-3', 'NMME', 'ENSMEAN'}
        )
        official = [name for name in WARM_MODELS[year] if name in available]
        official += sorted(available - set(official))
        models = [*official, 'MMarkov-4', 'MMarkov-3', 'ORAS5', 'NCM']
        months = sorted({row['target_month'] for row in rows})
        if len(months) != 6:
            raise ValueError(f'Event {year} needs six target months')
        first = np.datetime64(f'{year}-09' if period == 'winter' else f'{year}-02', 'M')
        expected_months = [str(first + np.timedelta64(i, 'M')) for i in range(6)]
        if months != expected_months:
            raise ValueError(f'Unexpected {period} target months for {year}')
        handles, arrays = ({}, {})
        for name in models:
            group = by_model(rows, name, 'target_month')
            lookup = {
                row['target_month']: float(row['prediction_C'])
                for row in group
                if row['prediction_C'] != ''
            }
            arrays[name] = np.array([lookup.get(month, np.nan) for month in months])
            style = (
                '--'
                if name == 'MMarkov-4'
                else '-.' if name == 'MMarkov-3' else '-'
            )
            width = 9.3 if name in ('NCM', 'ORAS5') else 2.0
            (line,) = ax.plot(
                range(6),
                arrays[name],
                color='#000000' if name == 'ORAS5' else COLORS[name],
                ls=style,
                lw=width,
                marker='o',
                ms=7 if name in ('NCM', 'ORAS5') else 5.2,
                mec='white',
                mew=0.5,
                zorder=6 if name == 'NCM' else 5 if name == 'ORAS5' else 3,
            )
            handles[name] = line
        maximum = max((np.nanmax(a) for a in arrays.values()))
        if period == 'winter':
            ymax = max(2.42, math.ceil((maximum + 0.12) * 5) / 5)
            ymin = 0.7 if year == 2023 else 1.0
        else:
            minimum = min((np.nanmin(a) for a in arrays.values()))
            ymax = math.ceil((maximum + 0.18) * 5) / 5
            ymin = math.floor((minimum - 0.18) * 5) / 5
        ax.set(
            xlim=(-0.35, 5.35),
            ylim=(ymin, ymax),
            ylabel='Nino3.4 (°C)',
            xlabel='Verification month',
        )
        ax.set_xticks(range(6), months, fontsize=18, fontweight='bold')
        ax.set_yticks(
            np.arange(np.ceil(ymin * 2) / 2, np.floor(ymax * 2) / 2 + 0.01, 0.5),
        )
        ax.set_xlabel('Verification month', fontsize=22, labelpad=13)
        ax.set_ylabel('Nino3.4 (°C)', fontsize=22, labelpad=5)
        ax.tick_params(axis='both', labelsize=18, pad=8, width=1.3, length=6)
        offsets = (
            [(0, 12), (-25, 12), (0, -16), (0, -20), (0, 12), (0, 12)]
            if year == 2015
            else [(0, -16), (0, -24), (0, 11), (0, 19), (0, -16), (17, 0)]
        )
        if period == 'february':
            offsets = [(0, -18), (0, -18), (0, 12), (0, -18), (0, -18), (15, 0)]
        for i, (value, (dx, dy)) in enumerate(zip(arrays['NCM'], offsets)):
            ax.annotate(
                f'{value:.2f}',
                (i, value),
                xytext=(dx, dy),
                textcoords='offset points',
                ha='center',
                va='bottom' if dy > 0 else 'top' if dy < 0 else 'center',
                fontsize=19,
                color=COLORS['NCM'],
                fontweight='bold',
                zorder=7,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.4},
            )
        names = ['NCM', 'MMarkov-4', 'MMarkov-3', *official]
        score_rows = {
            row['model']: row
            for row in mae_rows
            if int(row['event']) == year
        }
        ordered = [score_rows[name] for name in names]
        mae = values(ordered, 'MAE_C')
        bar_colors = []
        for rank_row, name in zip(ordered, names):
            paired = np.isfinite(arrays[name]) & np.isfinite(arrays['ORAS5'])
            if period == 'winter' and year == 2015:
                paired[0] = False
            expected_n = (
                5
                if year == 2015 and (period == 'winter' or name in official)
                else 6
            )
            expected_mask = np.ones(6, dtype=bool)
            if expected_n == 5:
                expected_mask[0] = False
            errors = arrays[name][paired] - arrays['ORAS5'][paired]
            if (
                not np.array_equal(paired, expected_mask)
                or int(rank_row['n']) != expected_n
            ):
                raise ValueError(
                    f'Incorrect {period} scoring coverage for {year} {name}',
                )
            if abs(float(np.abs(errors).mean()) - float(rank_row['MAE_C'])) > 1e-10:
                raise ValueError(
                    f'MAE table differs from {period} trajectory for {year} {name}',
                )
            bar_colors.append('#F27C2C' if name == 'NCM' else '#C8C8C8')
        bx.bar(range(len(names)), mae, color=bar_colors, width=0.6, edgecolor='none')
        bx.set_xticks(
            range(len(names)),
            [DISPLAY.get(name, name) for name in names],
            rotation=42,
            ha='right',
            fontsize=16,
            fontweight='bold',
        )
        bx.set_ylim(0, 1.5)
        bx.set_yticks(np.arange(0, 1.51, 0.3))
        bx.set_ylabel('mean MAE (°C)', fontsize=22, labelpad=14)
        bx.set_title(
            f'{year}/{year + 1}' if period == 'winter' else str(year),
            fontsize=22,
            fontweight='bold',
            pad=12,
        )
        bx.tick_params(axis='both', labelsize=17, pad=7, width=1.3, length=6)
        order = ['ORAS5', 'NCM', 'MMarkov-4', 'MMarkov-3', *official]
        legend_options = dict(
            handles=[handles[name] for name in order],
            labels=[
                'observed' if name == 'ORAS5' else DISPLAY.get(name, name)
                for name in order
            ],
            ncol=1,
            loc='upper left',
            alignment='left',
            bbox_to_anchor=(13.45 / canvas_width, (bottom + 5.8) / canvas_height),
            borderaxespad=0,
            prop={'size': 18, 'weight': 'bold'},
            frameon=False,
            handlelength=2.3,
            handletextpad=0.65,
            borderpad=0.7,
            labelspacing=0.8,
        )
        legend = fig.legend(**legend_options)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        height_delta = (
            ax.get_window_extent(renderer).height
            - legend.get_window_extent(renderer).height
        )
        spacing = 0.8 + height_delta * 72 / (fig.dpi * 18 * (len(order) - 1))
        legend.remove()
        legend_options['labelspacing'] = spacing
        legend = fig.legend(**legend_options)
        legends.append(legend)
        for name, text in zip(order, legend.get_texts()):
            if name in ('ORAS5', 'NCM'):
                text.set_color('#000000' if name == 'ORAS5' else COLORS['NCM'])
                text.set_fontweight('bold')
        for axis in (ax, bx):
            for spine in axis.spines.values():
                spine.set_linewidth(1.3)
            for tick in (*axis.get_xticklabels(), *axis.get_yticklabels()):
                tick.set_fontweight('bold')
        label_panel(fig, ax, letters[0], size=25)
        label_panel(fig, bx, letters[1], size=25)
    fig._event_legends = legends
    return fig


def main():
    args = figure_arguments(__doc__)
    rows = [
        read_table(
            args.data / 'Fig4-Predictions.csv',
            ['event', 'model', 'prediction_C', 'target_month'],
        ),
        read_table(args.data / 'Fig4-MAE.csv', ['MAE_C', 'event', 'model', 'n', 'rank']),
    ]
    fig = warm_events(rows[0], rows[1], period='february')
    save_figure(fig, args.output, 4, args.dpi)
    print(f'Figure 4 written to {args.output}')
if __name__ == '__main__':
    main()
