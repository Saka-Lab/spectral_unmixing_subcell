from pathlib import Path
import umap
import panel as pn
import pandas as pd
from collections import OrderedDict
from loguru import logger
import holoviews as hv
from holoviews.streams import RangeXY, Tap
from bokeh.models import HoverTool


def load_data(input_dir, annotations_dir, round_name, model, interphase_only=False, rename_map=None):
    data_file = Path(input_dir) / f"{round_name}_{model}.tsv"
    annotation_file = Path(annotations_dir) / f"{round_name}_{model}_annotations.tsv"
    # Read data
    # check if the file exists
    if data_file.exists():
        df = pd.read_csv(data_file, sep="\t")
    else:
        raise FileNotFoundError(f"Data file not found: {data_file}")
    # check if annotations exist
    if annotation_file.exists():
        annotations = pd.read_csv(annotation_file, sep="\t")
        # check if unique_cell_id column exists in both dataframes
        if 'unique_cell_id' not in annotations.columns:
            raise ValueError(f"'unique_cell_id' column not found in annotation file {annotation_file}. It is required to merge with data.")

        if 'unique_cell_id' not in df.columns:
            raise ValueError(
                f"'unique_cell_id' column not found in data file {data_file}. It is required to merge with annotations.")
        df = pd.merge(df, annotations, on="unique_cell_id", how="left")

        if interphase_only:
            if 'cell_cycle_phase' not in df.columns:
                raise ValueError(f"'cell_cycle_phase' column not found in merged dataframe. It is required to filter for interphase cells.")
            df = df[df["cell_cycle_phase"] == "Interphase"]
    else:
        logger.warning(f"Annotation file not found: {annotation_file}. Proceeding without annotations.")

    # Features columns are not needed
    feat_cols = [col for col in df.columns if 'feat' in col]
    df = df.drop(columns=feat_cols)
    # Sigmoid probabilities not needed
    sigmoid_cols = [col for col in df.columns if 'sig' in col]
    df = df.drop(columns=sigmoid_cols)

    # Classification from Subcell is not needed
    class_cols = ['classification', 'id', 'top_class_name', 'top_class', 'top_3_classes_names', 'top_3_classes']
    df = df.drop(columns=class_cols)

    if rename_map is not None:
        df = df.rename(columns=rename_map)
    return df


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


def ensure_df_columns(df, df_path, annotation_folder=None, tsv_dir=None):

    # check if there are nans
    if df.isnull().values.any():
        nan_rows = df[df.isnull().any(axis=1)]
        logger.debug(nan_rows)
        raise ValueError(f"DataFrame {df_path} contains NaN values. Please check the input data.")

    if f'UMAP1' not in df.columns:
        logger.info(f"Calculating UMAP")
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
        save_path = tsv_dir / exp_name
        # check if the directory exists, if not create it
        save_path.parent.mkdir(exist_ok=True)
        logger.info(f"Saving UMAP results to {save_path}")
        df.to_csv(save_path, index=False, sep='\t')

    if annotation_folder is not None:
        annotation_file = Path(annotation_folder) / (Path(df_path).stem + "_annotations.tsv")
        if annotation_file.exists():
            logger.info(f"Loading annotations from {annotation_file}")
            df_annotations = pd.read_csv(annotation_file, sep="\t")
            print(df_annotations.columns)

            # check whether unique_cell_id exist in both dataframes
            if 'unique_cell_id' not in df_annotations.columns:
                raise ValueError(f"Annotation file {annotation_file} must contain a 'unique_cell_id' column.")
            if 'unique_cell_id' not in df.columns:
                raise ValueError(
                    f"Data file {df_path} must contain a 'unique_cell_id' column to merge with annotations.")
            # check if there are common columns between df and df_annotations (except
            # unique_cell_id), if so raise an error to avoid confusion after merge
            common_cols = set(df.columns).intersection(set(df_annotations.columns)) - {'unique_cell_id'}
            if common_cols:
                raise ValueError(
                    f"Data file {df_path} and annotation file {annotation_file} have common columns: {common_cols}. Please rename these columns to avoid confusion after merge.")
            annotation_columns = [col for col in df_annotations.columns if col != 'unique_cell_id']
            df = df.merge(df_annotations, on='unique_cell_id', how='left', validate='many_to_one')
            for col in annotation_columns:
                if df[col].isna().any():
                    missing_ids = df.loc[df[col].isna(), "unique_cell_id"].unique()
                    raise ValueError(
                        f"Missing annotations for {len(missing_ids)} cells in column '{col}'."
                        f" Example missing unique_cell_id: {missing_ids[:15]}")
        else:
            logger.info(f"No annotation file found for {df_path} in {annotation_folder}. Skipping annotations.")

    return df


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
        leg_df = pd.DataFrame({"x": [lx], "y": [ly], "marker_type": [marker_sym]})
        shape_entries.append(
            hv.Scatter(leg_df, kdims=['x', 'y'], vdims=['marker_type'])
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
        label_df = pd.DataFrame({"x": [lx + dx], "y": [ly], "text": [label_text]})
        shape_texts.append(
            hv.Labels(label_df, kdims=['x', 'y'], vdims=['text']).opts(
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
        label_df = pd.DataFrame({"x": [lx + dx], "y": [ly], "text": [str(val)]})
        color_labels.append(
            hv.Labels(label_df, kdims=['x', 'y'], vdims=['text']).opts(
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


def create_tab(file_name, cfg, annotation_folder=None, tsv_dir=None):

    df = pd.read_csv(file_name, low_memory=False, sep='\t')

    df = ensure_df_columns(df, file_name, annotation_folder, tsv_dir)

    # sort by condition
    if 'condition' in df.columns:
        df['condition'] = pd.Categorical(df['condition'], categories=['Untreated', 'ActD', 'SA'], ordered=True)
        df = df.sort_values('condition')

    logger.info(f"Loaded {file_name} with {len(df)} rows")
    filter_cols = df.columns
    filter_cols = [col for col in filter_cols if 'sig_prob' not in col]
    filter_cols = [col for col in filter_cols if 'soft_prob' not in col]
    filter_cols = [col for col in filter_cols if 'UMAP' not in col]
    filter_cols = [
        col for col in filter_cols if col not in [
            'cell_id',
            'top_class',
            'top_3_classes',
            'top_3_classes_names',
            'id']]

    borders = get_borders(df)
    initial_xlim = borders["x_range"]
    initial_ylim = borders["y_range"]

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

    # ---------------------------------------------- WIDGET DEFINITIONS ------
    # Color by widget
    color_by = pn.widgets.Select(
        name="Color by",
        options=filter_cols,
        value="protein",
        sizing_mode="stretch_width",
        min_width=cfg.min_width)

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
    def plot_umap(color_by, alpha, trigger, show_color_legend=True, show_shape_legend=False, **kwargs):
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
                    line_dash="solid",
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
    left_sidebar.max_width = int(cfg.min_width * 1.1)
    right_sidebar.sizing_mode = 'stretch_both'
    right_sidebar.max_width = cfg.min_width
    central_plot.sizing_mode = 'stretch_both'

    layout = pn.Row(left_sidebar, central_plot, right_sidebar, sizing_mode='stretch_both')

    return layout, plot_umap
