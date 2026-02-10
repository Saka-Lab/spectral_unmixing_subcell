import os
from pathlib import Path

from utils.plot_utils import create_tab
import json
import panel as pn
import holoviews as hv
import yaml
from types import SimpleNamespace
from bokeh.io import export_svgs

hv.extension('bokeh')
pn.extension()

with open('configs/umap_config.yaml') as f:
    config_data = yaml.safe_load(f)

with open('configs/colormaps.json') as f:
    colormaps = json.load(f)
    config_data['colormap_dict'] = colormaps


cfg = SimpleNamespace(**config_data)




# ---- BUILD ALL TABS ----

def build_dashboard():
    umap_folder = 'umaps_fixed'
    # check if folder exists
    if not os.path.exists(umap_folder):
        print(f"Folder {umap_folder} does not exist.")
        return pn.Tabs(), None
    files = os.listdir(umap_folder)
    files = [f for f in files if f.endswith('.csv') or f.endswith('.tsv')]
    csv_files = [str(Path(umap_folder) / f) for f in files]
    
    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found in folder {umap_folder}")

    print(f"\nFound {len(csv_files)} files to plot:")
    tabs = pn.Tabs()

    for i, file in enumerate(csv_files):
        print(f" - {file}")
        layout, plot_umap = create_tab(file, cfg)
        tabs.append((file, layout))

    print(f"Created {len(tabs)} tabs for the dashboard")
    # ---- SERVE ----
    return tabs, plot_umap


dashboard, plot_umap = build_dashboard()
dashboard.servable()



# ---- EXPORT STATIC SNAPSHOT ----
color_by_val = 'protein'
alpha_val = 0.7
filters_val = {
    #'cell_cycle_phase': ['Interphase'],
    "protein": ['G3BP1', 'alphaTUBULIN', 'NPM1'],
}

kwargs = {**filters_val}
plot_snapshot = plot_umap(color_by=color_by_val, alpha=alpha_val, trigger=False, show_shape_legend=False, **kwargs)


# Convert HoloViews object to Bokeh
bokeh_obj = hv.render(plot_snapshot)
bokeh_obj.output_backend = "svg"

# Export (requires selenium + web driver)
export_svgs(bokeh_obj, filename="snapshot.svg")