import os
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO
from PIL import ImageEnhance
from functools import lru_cache

import umap
import panel as pn
import pandas as pd

import holoviews as hv
from holoviews.streams import RangeXY, Tap
from bokeh.models import HoverTool

def get_borders(df):
    xmin, xmax = df[f"UMAP1"].min(), df[f"UMAP1"].max()
    ymin, ymax = df[f"UMAP2"].min(), df[f"UMAP2"].max()
    # add a margin so that the points are not at the edge
    margin = 0.03
    xmin -= (xmax - xmin) * margin
    xmax += (xmax - xmin) * margin * 12  # to make space for the legend
    ymin -= (ymax - ymin) * margin
    ymax += (ymax - ymin) * margin
    borders_x = (xmin, xmax)
    borders_y = (ymin, ymax)
    return {"x_range": borders_x, "y_range": borders_y}


def build_hover_tooltip(cfg):
    lines = []
    if cfg.tooltip.get("show_thumbnail", False):
        lines.append("<div>@thumbnail{safe}</div>")

    for item in cfg.tooltip["fields"]:
        col = item["column"]
        label = item.get("label", col)
        lines.append(f"<div><strong>{label}:</strong> @{col}</div>")

    html = "<div>" + "\n".join(lines) + "</div>"

    return html



def create_filter_controls(df, filter_cols, cfg, max_visible=15, per_item_px=17):

    filters = {}
    filter_controls = {}
    max_height = per_item_px * max_visible

    for col in filter_cols:
        options = sorted(df[col].unique())
        n = len(options)
        total_height = per_item_px * n
        computed_height = min(max_height, total_height)

        checkbox = pn.widgets.CheckBoxGroup(options=options)
        filters[col] = checkbox

        if n > max_visible:
            # add a scrollbar
            checkbox_container = pn.Column(
                checkbox,
                height=computed_height,
                scroll=True,
            )
        else:
            checkbox_container = checkbox

        # select-all button
        select_all_button = pn.widgets.Button(name="Select All", button_type="success")
        select_all_button.on_click(lambda event, cb=checkbox, opts=options: setattr(cb, 'value', list(opts)))

        filter_controls[col] = pn.Column(checkbox_container, pn.Row(select_all_button))

    filter_widgets = [
        pn.Card(
            filter_controls[col],
            title=col,
            collapsed=True,
            min_width=cfg.min_width
        )
        for col in filter_cols
    ]
    return filters, filter_widgets


def ensure_df_columns(df, df_path):
    # check if there are nans
    if df.isnull().values.any():
        nan_rows = df[df.isnull().any(axis=1)]
        print(nan_rows)
        raise ValueError(f"DataFrame {df_path} contains NaN values. Please check the input data.")

    if f'UMAP1' not in df.columns:
        if f'UMAP1' in df.columns:
            df = df.drop(columns=[f'UMAP1', f'UMAP2'])
        
        print(f"Calculating UMAP")
        embeddings = df[[col for col in df.columns if col.startswith("feat")]].to_numpy()
        umap_model = umap.UMAP(n_neighbors=20, metric="cosine", min_dist=0.5, random_state=42)
        umap_embeddings = umap_model.fit_transform(embeddings)
        df_umap = pd.DataFrame(umap_embeddings, columns=[f"UMAP1", f"UMAP2"])
        df = pd.concat([df_umap, df], axis=1)

        # remove features columns
        df = df.drop(columns=[col for col in df.columns if col.startswith("feat")])
        df['cell_id'] = df['cell_id'].astype(int)
        df['top_class'] = df['top_class'].astype(int)

        # remove extension from file name
        exp_name = Path(df_path).name
        save_path = Path('umaps') / exp_name
        # check if the directory exists, if not create it
        save_path.parent.mkdir(exist_ok=True)
        print(f"Saving UMAP results to {save_path}")
        df.to_csv(save_path, index=False, sep='\t')

    return df

from collections import OrderedDict

def derive_shape_labels(marker_types_for_shape):
    """
    marker_types_for_shape: dict[value -> marker]
    returns: OrderedDict[marker -> label_string]
    """
    marker_to_values = OrderedDict()

    for value, marker in marker_types_for_shape.items():
        marker_to_values.setdefault(marker, []).append(value)

    return {
        marker: ", ".join(values)
        for marker, values in marker_to_values.items()
    }


def build_shape_legend(cfg, x0, y_start, dx, dy, offset):
    shape_entries = []
    shape_texts = []

    # position below color legend if both are enabled
    y_start_shapes = y_start - (offset + 1) * dy

    shape_labels = derive_shape_labels(cfg.marker_types[cfg.shape_by])


    for i, (marker_sym, label_text) in enumerate(shape_labels.items()):
        lx = x0
        ly = y_start_shapes - i * dy
        # include a dummy vdims column so Scatter has at least one vdims
        leg_df = pd.DataFrame({"x":[lx], "y":[ly], "marker_type":[marker_sym]})
        shape_entries.append(
        hv.Scatter(leg_df, kdims=['x','y'], vdims=['marker_type'])
            .opts(
                marker=marker_sym,
                size=12,
                alpha=1.0,
                color='white',
                line_color='black',
                line_width=0.5,
                show_legend=False
            )
        )
        label_df = pd.DataFrame({"x":[lx + dx], "y":[ly], "text":[label_text]})
        shape_texts.append(
            hv.Labels(label_df, kdims=['x','y'], vdims=['text']).opts(
                text_font_size='15pt',
                text_color='black',
                text_align='left',
                text_baseline='middle',
                show_legend=False
            )
        )

    overlay = hv.Overlay(shape_entries + shape_texts)
    return overlay

def build_color_legend(cfg, x0, y_start, dx, dy, unique_colors, color_by):
    color_entries = []
    color_labels = []

    for i, val in enumerate(unique_colors):
        lx = x0
        ly = y_start - i * dy
        leg_df = pd.DataFrame({
            "UMAP1": [lx],
            "UMAP2": [ly],
            color_by: [val]
        })
        color_entries.append(
            hv.Scatter(leg_df, kdims=["UMAP1", "UMAP2"], vdims=[color_by])
            .opts(
                marker='circle',
                size=12,
                alpha=1.0,
                color=color_by,
                cmap=cfg.colormap_dict.get(color_by, "Category10"),
                line_color='black',
                line_width=0.3,
                show_legend=False
            )
        )
        label_df = pd.DataFrame({"x":[lx + dx], "y":[ly], "text":[str(val)]})
        color_labels.append(
            hv.Labels(label_df, kdims=['x','y'], vdims=['text']).opts(
                text_font_size='15pt',
                text_color='black',
                text_align='left',
                text_baseline='middle',
                show_legend=False
            )
        )
    overlay = hv.Overlay(color_entries + color_labels)
    return overlay

def build_too_many_entries_label(x0, y_start, text):
    df = pd.DataFrame({
        "x": [x0],
        "y": [y_start],
        "text": [text]
    })
    return hv.Labels(df, kdims=["x", "y"], vdims=["text"]).opts(
        text_font_size="14pt",
        text_color="gray",
        text_align="left",
        text_baseline="top",
        show_legend=False
    )

def create_tab(file_name, cfg):

    df = pd.read_csv(file_name, low_memory=False, sep='\t')

    df = ensure_df_columns(df, file_name)
    
    print(f"Loaded {file_name} with {len(df)} rows")
    filter_cols = df.columns
    filter_cols = [col for col in filter_cols if 'sig_prob' not in col]
    filter_cols = [col for col in filter_cols if 'soft_prob' not in col]
    filter_cols = [col for col in filter_cols if 'UMAP' not in col]
    filter_cols = [col for col in filter_cols if col not in ['cell_id', 'top_class', 'top_3_classes', 'top_3_classes_names', 'id']]


    borders = get_borders(df)
    initial_xlim = borders["x_range"]
    initial_ylim = borders["y_range"]
    
    # in the condition column replace ActinomycinD with ActD, SodiumArsenite with SA
    if 'condition' in df.columns:
        df['condition'] = df['condition'].replace({
            'ActinomycinD': 'ActD',
            'SodiumArsenite': 'SA'
        })

    # legend computations
    x0 = initial_xlim[0] + 0.75 * (initial_xlim[1] - initial_xlim[0])
    y_start = initial_ylim[1] - 0.05 * (initial_ylim[1] - initial_ylim[0])
    dy = 0.05 * (initial_ylim[1] - initial_ylim[0])
    dx = 0.02 * (initial_xlim[1] - initial_xlim[0])

    # streams for interaction
    range_stream = RangeXY(x_range=initial_xlim, y_range=initial_ylim)
    tap_stream = Tap(source=None)
    # trigger for updating the plot after a tap/zoom
    trigger = pn.widgets.Toggle(visible=False)

    # ---------------------------------------------- WIDGET DEFINITIONS -----------------------------------------------------
    # Color by widget
    color_by = pn.widgets.Select(name="Color by", options=filter_cols, value="protein", sizing_mode="stretch_width", min_width=cfg.min_width)

    # Filters based on columns elements
    filters, filter_widgets = create_filter_controls(df, filter_cols, cfg)

    # Settings widgets
    alpha_slider = pn.widgets.FloatSlider(name="Alpha", start=0.1, end=1.0, step=0.1, value=cfg.init_alpha)

    clear_filters_button = pn.widgets.Button(name="Clear Filters", button_type="primary")
    def clear_filters(event):
        for w in filters.values():
            w.value = []
    clear_filters_button.on_click(clear_filters)

    reset_zoom_button = pn.widgets.Button(name="Reset zoom", button_type="primary")
    def reset_zoom(event):
        range_stream.update(x_range=initial_xlim, y_range=initial_ylim)
        trigger.value = not trigger.value
    reset_zoom_button.on_click(reset_zoom)

    # ---------------------------------------- LAYOUT SETUP -----------------------------------------
    # LEFT sidebar widgets
    left_sidebar = pn.WidgetBox(color_by, *filter_widgets)
    # RIGHT sidebar widgets
    right_sidebar = pn.WidgetBox(alpha_slider, clear_filters_button, reset_zoom_button)

    df["marker_type"] = df[cfg.shape_by].map(cfg.marker_types[cfg.shape_by]).fillna("circle")
    df["size"] = df["marker_type"].apply(lambda x: cfg.marker_sizes.get(x, 6))
    df["is_highlighted"] = df["unique_cell_id"].isin(cfg.highlighted_cells)


    @pn.depends(color_by, alpha_slider.param.value_throttled, trigger, **filters)
    def plot_umap(color_by, alpha, trigger, show_color_legend=True, show_shape_legend=True, **kwargs):
        # Filter data
        filtered = df
        for col, selected in kwargs.items():
            if selected:
                filtered = filtered[filtered[col].isin(selected)]
        filtered = filtered.copy()

        highlighted_df = filtered[filtered["is_highlighted"]]
        filtered = filtered[filtered["is_highlighted"] == False]
        hover_tool = HoverTool(tooltips=build_hover_tooltip(cfg))


        layers = []
        for marker, subdf in filtered.groupby("marker_type"):
            layers.append(
                hv.Scatter(
                    subdf,
                    kdims=["UMAP1", "UMAP2"],
                    vdims=filter_cols + ["size", "is_highlighted"]
                ).opts(
                    marker=marker,
                    size="size",
                    color=color_by,
                    cmap=cfg.colormap_dict.get(color_by, "Category10"),
                    alpha=alpha,
                    line_color="black",
                    line_width=0.2,
                    line_alpha=1.0,
                    tools=[hover_tool, "tap"],
                    show_legend=False,
                )
            )

        points = hv.Overlay(layers)

        highlight_layers = []

        for marker, subdf in highlighted_df.groupby("marker_type"):
            highlight_layers.append(
                hv.Scatter(
                    subdf,
                    kdims=["UMAP1", "UMAP2"],
                    vdims=filter_cols + ["size", "is_highlighted"]
                ).opts(
                    marker=marker,
                    size=hv.dim("size") * 1.5,
                    color=color_by,
                    cmap=cfg.colormap_dict.get(color_by, "Category10"),
                    alpha=1.0,
                    line_color=cfg.highlight_color,
                    line_width=cfg.highlight_width,
                    line_dash="dashed",
                    tools=[hover_tool, "tap"],
                    show_legend=False,
                )
            )

        points_highlight = hv.Overlay(highlight_layers)

        overlays = [points, points_highlight]

        unique_colors = list(pd.unique(filtered[color_by]))
        unique_colors = ['Untreated', 'ActD', 'SA'] if color_by == 'condition' else unique_colors
        n_colors = len(unique_colors)

        # ---------- MANUAL COLOR LEGEND ----------
        if show_color_legend and n_colors <= cfg.max_legend_entries:
            color_ov = build_color_legend(cfg, x0, y_start, dx, dy, unique_colors, color_by)
            overlays.append(color_ov)
        elif show_color_legend and n_colors > cfg.max_legend_entries:
            label = build_too_many_entries_label(x0, y_start, f"Legend hidden\nToo many categories ({n_colors})")
            overlays.append(label)
            n_colors = 2

        # ---------- MANUAL SHAPE LEGEND ----------
        n_colors = n_colors if show_color_legend else 0
        if show_shape_legend:
            shape_ov = build_shape_legend(cfg, x0, y_start, dx, dy, n_colors)         
            overlays.append(shape_ov)

        combined = hv.Overlay(overlays).opts(aspect=None, responsive=True)

        # Attach streams
        combined = combined.apply.opts(
            xlim=range_stream.param.x_range,
            ylim=range_stream.param.y_range
        )
        range_stream.source = combined
        tap_stream.source = combined

        return combined


    central_plot = pn.panel(plot_umap)
    left_sidebar.sizing_mode = 'stretch_both'
    left_sidebar.max_width = int(cfg.min_width*1.1)
    right_sidebar.sizing_mode = 'stretch_both'
    right_sidebar.max_width = cfg.min_width
    central_plot.sizing_mode = 'stretch_both'

    layout = pn.Row(left_sidebar, central_plot, right_sidebar, sizing_mode='stretch_both')


    return layout, plot_umap
