"""Draw NCM main Figure 3 from its saved final numerical tables."""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import numpy as np
import calendar
from scipy.stats import t as student_t
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
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


def _label_isolines(
    axis, x, y, field, levels, fontsize=13,
    line_color='#493b32', linewidth=0.85, label_anchors=None,
):
    from matplotlib.font_manager import FontProperties
    from matplotlib.transforms import Bbox
    available = [
        float(level)
        for level in levels
        if np.nanmin(field) < level < np.nanmax(field)
    ]
    contour = axis.contour(
        x,
        y,
        field,
        levels=available,
        colors=line_color,
        linewidths=linewidth,
        alpha=0.95,
        zorder=3,
    )
    figure = axis.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    font = FontProperties(size=fontsize, weight='normal')
    interior = axis.get_window_extent().padded(-9)
    accepted, used, details = ([], set(), [])
    segments = {
        float(level): [segment.copy() for segment in paths]
        for level, paths in zip(contour.levels, contour.allsegs)
    }
    if label_anchors is not None:
        for level, anchor in label_anchors.items():
            segments_at_level = next(
                (
                    paths
                    for value, paths in segments.items()
                    if abs(value - level) < 1e-08
                ),
                [],
            )
            candidates = []
            for segment in segments_at_level:
                for left, right in zip(segment[:-1], segment[1:]):
                    candidates.extend(
                        (
                            left[None, :]
                            + np.linspace(0, 1, 201)[:, None] * (right - left)[None, :]
                        ),
                    )
            if not candidates:
                continue
            candidates = np.asarray(candidates)
            point = candidates[
                np.argmin(
                    np.square(
                        axis.transData.transform(candidates)
                        - axis.transData.transform(anchor)
                    ).sum(axis=1)
                )
            ]
            label = axis.text(
                *point,
                '0' if level == 0 else f'{level:.2f}',
                ha='center',
                va='center',
                fontsize=fontsize,
                color='#202020',
                fontweight='normal',
                clip_on=True,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.1},
                zorder=5,
            )
            details.append({'value': level, 'position': point.tolist()})
        return {
            'levels': available,
            'labels': details,
            'line_color': line_color,
            'linewidth': linewidth,
        }
    for repetition in range(2):
        for level in reversed(available):
            if len(details) >= 12:
                break
            text = '0' if level == 0 else f'{level:.2f}'
            width, height, descent = renderer.get_text_width_height_descent(
                text,
                font,
                ismath=False,
            )
            choices = []
            for index, segment in enumerate(segments[level]):
                if (level, index) in used or len(segment) < 2:
                    continue
                screen = axis.transData.transform(segment)
                distance = np.r_[
                    0.0,
                    np.cumsum(np.linalg.norm(np.diff(screen, axis=0), axis=1)),
                ]
                if distance[-1] < width + 28:
                    continue
                for fraction in (0.5, 0.35, 0.65, 0.2, 0.8):
                    position = distance[-1] * fraction
                    point = np.array(
                        [
                            np.interp(position, distance, screen[:, 0]),
                            np.interp(position, distance, screen[:, 1]),
                        ],
                    )
                    box = Bbox.from_bounds(
                        point[0] - width / 2 - 6,
                        point[1] - (height + descent) / 2 - 6,
                        width + 12,
                        height + descent + 12,
                    )
                    if not (
                        interior.contains(box.x0, box.y0)
                        and interior.contains(box.x1, box.y1)
                    ):
                        continue
                    if any((box.overlaps(previous) for previous in accepted)):
                        continue
                    spacing = min(
                        (
                            np.linalg.norm(
                                point - np.array(previous.get_points()).mean(0),
                            )
                            for previous in accepted
                        ),
                        default=300.0,
                    )
                    choices.append((spacing + 0.03 * distance[-1], index, point, box))
            if not choices:
                continue
            _, index, point, proposed_box = max(choices, key=lambda item: item[0])
            position = axis.transData.inverted().transform(point)
            label = axis.text(
                position[0],
                position[1],
                text,
                ha='center',
                va='center',
                fontsize=fontsize,
                color='#202020',
                fontweight='normal',
                clip_on=True,
            )
            label.set_bbox(
                {'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.0, 'alpha': 0.97},
            )
            label.set_zorder(5)
            figure.canvas.draw()
            box = label.get_window_extent(figure.canvas.get_renderer()).padded(5)
            if (
                any((box.overlaps(previous) for previous in accepted))
                or not interior.contains(box.x0, box.y0)
                or (not interior.contains(box.x1, box.y1))
            ):
                label.set_visible(False)
                continue
            accepted.append(box)
            used.add((level, index))
            details.append({'value': level, 'position': position.tolist()})
    return {
        'levels': available,
        'labels': details,
        'line_color': line_color,
        'linewidth': linewidth,
    }


def seasonal_skill(rows):
    set_style(16, True)
    leads, months = (np.arange(1, 9), np.arange(1, 13))
    fields, counts = ({}, {})
    for name in ('MMarkov-4', 'NCM'):
        fields[name] = np.full((12, 8), np.nan)
        counts[name] = np.zeros((12, 8), dtype=int)
        for row in rows:
            if row['model'] == name:
                month, lead = (int(row['initial_month']) - 1, int(row['lead']) - 1)
                fields[name][month, lead] = float(row['ACC'])
                counts[name][month, lead] = int(row['n'])
        if not np.isfinite(fields[name]).all() or (counts[name] < 3).any():
            raise ValueError(f'Incomplete month–lead table for {name}')
    if not np.array_equal(counts['NCM'], counts['MMarkov-4']):
        raise ValueError('The two heatmaps need matching sample counts')
    width, height = (17.6, 25.7)
    figure = plt.figure(figsize=(width, height), facecolor='white')
    columns = (1.5, 9.7 + 63.562689 / 72)
    panel_width, panel_height = (6.4, 6.0)
    positions = {
        'a': (columns[0], 18.8),
        'd': (columns[1], 18.8),
        'b': (columns[0], 10.4),
        'e': (columns[1], 10.4),
        'c': (columns[0], 2.0),
        'f': (columns[1], 2.0),
    }

    def make_axis(letter):
        x, y = positions[letter]
        axis = figure.add_axes(
            [x / width, y / height, panel_width / width, panel_height / height],
            label=f'panel_{letter}',
        )
        axis.tick_params(labelsize=18, width=1.8, length=5.5, pad=5)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.0)
        label_panel(figure, axis, letter, 24)
        return axis
    colors = [
        '#fff4a3',
        '#ffe27a',
        '#ffc247',
        '#ff9a33',
        '#ff7130',
        '#f44329',
        '#e52b26',
    ]
    levels = np.arange(60, 96, 5) / 100
    cmap = ListedColormap(colors)
    cmap.set_under('white')
    cmap.set_over('#d71920')
    norm = BoundaryNorm(levels, len(colors))
    delta_colors = ['#fff4a3', '#ffdb65', '#ffb13c', '#ff8530', '#f74b27', '#e52b26']
    delta_levels = np.linspace(0, 30, len(delta_colors) + 1)
    delta_map = ListedColormap(delta_colors)
    delta_map.set_under('white')
    delta_map.set_over('#d71920')
    delta_norm = BoundaryNorm(delta_levels, len(delta_colors))
    arrays = [
        fields['MMarkov-4'],
        fields['NCM'],
        100 * (fields['NCM'] ** 2 - fields['MMarkov-4'] ** 2),
    ]
    titles = ['MMarkov-4', 'NCM', 'ΔACC² (NCM − MMarkov-4)']
    label_anchors = {
        'a': {
            0.95: [4.931034479167, 8.486976597222],
            0.9: [2.643057560764, 2.034482752315],
            0.85: [6.482758613281, 5.197491604167],
            0.8: [7.603542163628, 9.125928530093],
            0.75: [5.762637395833, 11.674285900463],
            0.7: [3.99999999566, 1.731884592593],
            0.65: [6.781685212674, 11.031897104167],
            0.6: [5.517241382378, 3.382666229167],
        },
        'b': {
            0.95: [4.724137934028, 7.305824606481],
            0.9: [3.479087233073, 2.655172418981],
            0.85: [7.00000000651, 3.642891523148],
            0.8: [7.612093875868, 9.482758625],
            0.75: [5.902752786458, 11.55495537963],
            0.7: [5.931034487847, 1.626064134259],
            0.65: [7.083958838976, 10.866487710648],
            0.6: [6.678495429688, 11.690638164352],
        },
        'c': {
            30.0: [3.706801471354, 2.247567648148],
            25.0: [7.206896551649, 1.308480333333],
            20.0: [2.676668559028, 2.76975794213],
            15.0: [7.00000000651, 4.87361262037],
            10.0: [6.522855017361, 9.999999988426],
            5.0: [5.482758619792, 6.775410946759],
            0.0: [1.262322994792, 7.950606233796],
        },
    }
    contour_checks = {}
    display_leads = leads
    display_fields = []
    for index, (array, title) in enumerate(zip(arrays, titles)):
        letter = 'abc'[index]
        axis = make_axis(letter)
        difference = index == 2
        boundaries = delta_levels if difference else levels
        display = array.copy()
        display_fields.append(display)
        axis.contourf(
            display_leads,
            months,
            display,
            levels=boundaries,
            cmap=delta_map if difference else cmap,
            norm=delta_norm if difference else norm,
            extend='both',
            antialiased=False,
            corner_mask=False,
        )
        axis.set(xlim=(1, 8), ylim=(1, 12))
        axis.set_xticks(leads)
        axis.set_yticks(months, MONTHS)
        axis.set_title(title, fontsize=22 if difference else 24, pad=18)
        axis.set_ylabel('Initial month', fontsize=22, labelpad=10)
        axis.set_xlabel('lead (months)', fontsize=22, labelpad=10)
        contour_checks[letter] = _label_isolines(
            axis,
            display_leads,
            months,
            display,
            boundaries,
            fontsize=14,
            line_color='white',
            linewidth=1.4,
            label_anchors=label_anchors[letter],
        )
    for letter, bottom in (('a', 17.75), ('b', 9.35)):
        cax = figure.add_axes(
            [1.85 / width, bottom / height, 5.7 / width, 0.16 / height],
            label=f'colorbar_ACC_{letter}',
        )
        colorbar = figure.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            cax=cax,
            orientation='horizontal',
            boundaries=levels,
            ticks=levels,
            spacing='proportional',
            extend='both',
            drawedges=True,
            extendfrac=0.08,
        )
        colorbar.ax.set_xticklabels(
            [f'{value:.2f}' for value in levels],
            fontweight='normal',
        )
        colorbar.ax.tick_params(length=0, pad=7, labelsize=16)
        colorbar.set_label('ACC', fontsize=17, labelpad=5)
    dbar = figure.add_axes(
        [1.85 / width, 0.95 / height, 5.7 / width, 0.16 / height],
        label='colorbar_difference',
    )
    delta_bar = figure.colorbar(
        ScalarMappable(norm=delta_norm, cmap=delta_map),
        cax=dbar,
        orientation='horizontal',
        boundaries=delta_levels,
        ticks=delta_levels,
        spacing='proportional',
        extend='both',
        drawedges=True,
    )
    delta_bar.ax.set_xticklabels(
        [f'{value:.2f}'.rstrip('0').rstrip('.') for value in delta_levels],
        fontweight='normal',
    )
    delta_bar.ax.tick_params(length=0, pad=7, labelsize=15)
    delta_bar.set_label('ΔACC² (%)', fontsize=17, labelpad=5)
    for letter, month in zip('def', (3, 4, 6)):
        axis = make_axis(letter)
        for name in ('NCM', 'MMarkov-4'):
            axis.plot(
                leads,
                fields[name][month - 1],
                color='#ff1a1a' if name == 'NCM' else '#335ca5',
                linestyle='-',
                linewidth=5.4,
                marker='o',
                markersize=10.4,
                markerfacecolor='#ff1a1a' if name == 'NCM' else 'white',
                markeredgewidth=2.2,
                label=name,
            )
        n = counts['NCM'][month - 1]
        critical_t = student_t.ppf(0.975, n - 2)
        critical_r = critical_t / np.sqrt(critical_t ** 2 + n - 2)
        if critical_r.min() >= 0.4:
            axis.plot(
                leads,
                critical_r,
                color='#808080',
                ls=(0, (5, 3)),
                lw=1.5,
                label='95% significance',
                zorder=1,
            )
        target = (month + leads - 2) % 12
        ticklabels = [
            MONTHS[value] + ('(+1)' if month + lead - 1 > 12 else '')
            for value, lead in zip(target, leads)
        ]
        axis.set(xlim=(1, 8), ylim=(0, 1))
        axis.margins(x=0)
        axis.set_xticks(leads, ticklabels)
        axis.set_yticks(np.arange(0, 1.01, 0.2))
        axis.set_title(
            f'{calendar.month_name[month]} initialization',
            fontsize=24,
            pad=18,
        )
        axis.set_ylabel('ACC', fontsize=22, labelpad=10)
        axis.set_xlabel('Forecast target month', fontsize=22, labelpad=10)
        axis.tick_params(axis='x', labelsize=16.5, pad=9)
        if month == 6:
            axis.get_xticklabels()[-1].set_fontsize(15)
        axis.legend(
            loc='lower left',
            bbox_to_anchor=(0.01, 0.015),
            frameon=False,
            fontsize=16,
            handlelength=2.5,
            borderaxespad=0.45,
        )
    figure._contour_checks = contour_checks
    figure._heatmap_values = arrays
    figure._heatmap_display_fields = display_fields
    figure._heatmap_display_leads = display_leads
    figure._sample_counts = counts
    figure._panel_dimensions_inches = [panel_width, panel_height]
    figure._layout = [['a', 'd'], ['b', 'e'], ['c', 'f']]
    figure._palette = {
        'ACC': colors,
        'difference': delta_colors,
        'over': '#d71920',
        'white_mix_fraction': 0.0,
    }
    figure._difference_definition = '100 * (ACC_NCM**2 - ACC_MM4**2)'
    figure._difference_levels = delta_levels
    padding = 0.35
    old_width, old_height = figure.get_size_inches()
    new_width, new_height = (old_width + 2 * padding, old_height + 2 * padding)
    axes_positions = [axis.get_position(original=True).bounds for axis in figure.axes]
    text_positions = [text.get_position() for text in figure.texts]
    figure.set_size_inches(new_width, new_height, forward=True)
    for axis, (x, y, w, h) in zip(figure.axes, axes_positions):
        axis.set_position(
            [
                (x * old_width + padding) / new_width,
                (y * old_height + padding) / new_height,
                w * old_width / new_width,
                h * old_height / new_height,
            ],
        )
    for text, (x, y) in zip(figure.texts, text_positions):
        text.set_position(
            (
                (x * old_width + padding) / new_width,
                (y * old_height + padding) / new_height,
            ),
        )
    figure._outer_padding_inches = padding
    return figure


def main():
    args = figure_arguments(__doc__)
    rows = [
        read_table(
            args.data / 'Fig3-ACC.csv',
            ['ACC', 'initial_month', 'lead', 'model', 'n'],
        ),
    ]
    fig = seasonal_skill(rows[0])
    save_figure(fig, args.output, 3, args.dpi)
    print(f'Figure 3 written to {args.output}')
if __name__ == '__main__':
    main()
