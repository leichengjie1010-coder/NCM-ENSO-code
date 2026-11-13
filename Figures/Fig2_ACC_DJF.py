"""Draw NCM main Figure 2 from its saved final numerical tables."""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import numpy as np
from scipy.stats import linregress, t as student_t
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


def skill_comparison(rows, djf_rows):
    """ACC curves and 34 unsmoothed DJF monthly pairs at fixed lead six."""
    rows = [row for row in rows if 1 <= int(row['lead']) <= 8]
    set_style(11, True)
    width, height = (1430.0, 672.0)
    fig = plt.figure(figsize=(width / 72, height / 72), facecolor='white')

    def axes_points(left, top, panel_width, panel_height):
        return fig.add_axes(
            [
                left / width,
                1 - (top + panel_height) / height,
                panel_width / width,
                panel_height / height,
            ],
        )
    ax = axes_points(102.24, 77.472, 753.48, 406.368)
    bx = axes_points(954.0, 77.472, 406.368, 406.368)
    handles = {}
    for name in ACC_ORDER:
        group = by_model(rows, name)
        if not group:
            continue
        own = name == 'NCM'
        color = (
            '#e31a1c'
            if name in ('NCM', 'MMarkov-4')
            else '#000000' if name == 'MMarkov-3' else COLORS[name]
        )
        (line,) = ax.plot(
            values(group, 'lead'),
            values(group, 'ACC'),
            color=color,
            linewidth=5.4 if own else 2.7 if name.startswith('MMarkov') else 1.65,
            linestyle=(0, (2.261111, 0.977778)) if name == 'MMarkov-4' else '-',
            alpha=0.85 if '(IRI)' in name or '(NMME)' in name else 1,
            label=name,
            zorder=6 if own else 3,
        )
        handles[name] = line
    ax.set(
        xlim=(0.5, 8.5),
        ylim=(0.5, 1),
        xlabel='Prediction lead (months)',
        ylabel='ACC',
    )
    ax.set_xticks(range(1, 9))
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)
    ax.tick_params(labelsize=11, width=1.1, length=3.5, pad=3.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
    short = values(by_model(rows, 'NCM')[:6], 'ACC')
    annotation_top = [
        226.866668,
        263.439788,
        300.012907,
        336.586028,
        373.159148,
        409.732268,
        446.3082,
    ]
    labels = [f'L{lead} = {score:.2f}' for lead, score in enumerate(short, 1)]
    labels.append(f'Mean (L1–L6) = {short.mean():.2f}')
    for text, top in zip(labels, annotation_top):
        fig.text(
            125.78625 / width,
            1 - top / height,
            text,
            fontsize=16,
            ha='left',
            va='baseline',
        )
    legend_columns = [ACC_ORDER[:4], ACC_ORDER[4:8], ACC_ORDER[8:11], ACC_ORDER[11:]]
    column_x = [103.225891, 280.323231, 478.609331, 676.470861]
    text_x = [131.275891, 308.373221, 506.659321, 704.520861]
    for column, line_x, label_x in zip(legend_columns, column_x, text_x):
        for row_index, name in enumerate(column):
            if name not in handles:
                continue
            top = 558.911655 + row_index * 15.272812
            original = handles[name]
            line = plt.Line2D(
                [line_x / width, (line_x + 24.2) / width],
                [1 - (top - 3.85) / height] * 2,
                transform=fig.transFigure,
                color=original.get_color(),
                linestyle=original.get_linestyle(),
                alpha=original.get_alpha(),
                linewidth=original.get_linewidth() * 1.15,
            )
            if name == 'MMarkov-4':
                line.set_dashes((2.261111, 0.977778))
            fig.add_artist(line)
            fig.text(
                label_x / width,
                1 - top / height,
                name,
                fontsize=12,
                ha='left',
                va='baseline',
            )
    x, y = (values(djf_rows, 'prediction_C'), values(djf_rows, 'observation_C'))
    if x.shape != (34,) or not np.isfinite(np.r_[x, y]).all():
        raise ValueError('The published DJF panel needs 34 finite monthly pairs')
    fit = linregress(x, y)
    grid = np.linspace(x.min() - 0.08, x.max() + 0.08, 201)
    fitted = fit.intercept + fit.slope * grid
    variance = np.square(y - fit.intercept - fit.slope * x).sum() / (len(x) - 2)
    half_width = student_t.ppf(0.975, len(x) - 2) * np.sqrt(
        variance * (
            1 / len(x)
            + np.square(grid - x.mean()) / np.square(x - x.mean()).sum()
        )
    )
    orange = '#dc652f'
    bx.axhline(0, color='#d0d0d0', lw=1.1, zorder=0)
    bx.axvline(0, color='#d0d0d0', lw=1.1, zorder=0)
    bx.fill_between(
        grid,
        fitted - half_width,
        fitted + half_width,
        color=orange,
        alpha=0.16,
        linewidth=0,
        zorder=1,
    )
    bx.plot(grid, fitted, color=orange, lw=2.3, ls=(0, (6, 3)), zorder=2)
    bx.scatter(x, y, s=88, c=orange, alpha=0.95, edgecolors='none', zorder=3)
    bx.set(xlim=(-1.5, 3), ylim=(-1.5, 3))
    bx.set_aspect('equal', adjustable='box')
    bx.set_xticks(np.arange(-1.5, 3.01, 0.5))
    bx.set_yticks(np.arange(-1.5, 3.01, 0.5))
    bx.tick_params(labelsize=12, length=4.5, width=1.1, pad=6)
    bx.set_xlabel('Predicted Niño3.4 (°C)', fontsize=18, labelpad=12)
    bx.set_ylabel('Observed Niño3.4 (°C)', fontsize=18, labelpad=12)
    for spine in bx.spines.values():
        spine.set_visible(spine.spine_type in ('left', 'bottom'))
        spine.set_color('#666666')
        spine.set_linewidth(1.3)
    bx.text(
        0.020112,
        0.947747,
        'DJF, lead-6',
        transform=bx.transAxes,
        fontsize=19,
        color=orange,
        va='baseline',
    )
    mantissa, exponent = f'{fit.pvalue:.2e}'.split('e')
    superscript = str(int(exponent)).translate(
        str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'),
    )
    for text, baseline in (
        (f'ACC = {fit.rvalue:.2f}', 411.2998),
        (f'p = {mantissa} × 10{superscript}', 433.02505),
    ):
        fig.text(
            (864 + 324.47827 + 1.2101886) / width,
            1 - (36 + baseline - 9.8055216) / height,
            text,
            fontsize=17,
            color=orange,
            va='baseline',
        )
    for letter, left in (('a', 102.24), ('b', 954.0)):
        fig.text(
            left / width,
            1 - 63.472 / height,
            f'({letter})',
            fontsize=18,
            ha='left',
            va='baseline',
        )
    fig._djf_pairs = np.column_stack([x, y])
    fig._djf_statistics = {
        'n': len(x),
        'ACC': fit.rvalue,
        'p': fit.pvalue,
        'slope': fit.slope,
        'intercept': fit.intercept,
    }
    return fig


def main():
    args = figure_arguments(__doc__)
    rows = [
        read_table(args.data / 'Fig2-ACC.csv', ['ACC', 'lead', 'model']),
        read_table(
            args.data / 'Fig2-DJF-scatter.csv',
            [
                'last_complete_input_month',
                'named_initialization_month',
                'observation_C',
                'prediction_C',
                'target_month',
            ],
        ),
    ]
    fig = skill_comparison(rows[0], rows[1])
    save_figure(fig, args.output, 2, args.dpi)
    print(f'Figure 2 written to {args.output}')
if __name__ == '__main__':
    main()
