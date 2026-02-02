## Overview
This repository provides the end-to-end workflow used in the paper [here](to_be_decided).
The workflow starts with raw image data, processes it and runs SubCell classicication. Results
can be interactively visualized afterwards. The code is specific to the data used in the paper,
but can be adjusted to your own data.

For more information about SubCell, see the [SubCell paper](https://www.biorxiv.org/content/10.1101/2024.12.06.627299v2).

## Expected directory structure and naming

### Configs
First there should be a `configs` directory. This directory contains configurations for the interactive
visualization with bokeh:
        
- colormaps.json: explain more here
- constant.yaml:
- plot_config.yaml:
- umap_config.yaml:

### Image Data
The raw image data should be stored in a directory called `data`, which is present in this project directory.
In this directory, for running the workflow with default parameters, it is expected that there is a directory called 
`raw`. This directory contains experiment directories with names of the user's choice, that should contain either
'.lif' or '.tif', '.tiff' images. The dimension names and order of these images is expected to be `czyx`. The images
can contain one or multiple scenes (Fields of View).

## Getting started

### Installing Pixi

This project uses [pixi](https://pixi.sh) for dependency management. To install pixi, follow these steps
(check installation instructions [here](https://pixi.prefix.dev/latest/installation/) in case of potential outdated instructions):

#### Windows
You can download and run the [installer](https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-pc-windows-msvc.msi)
or run the following:
```
# Using PowerShell (recommended)
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```
The terminal needs to be restarted to make the installation effective. See details below for what this does.
<details>
The above invocation will automatically download the latest version of pixi, extract it, and move the pixi binary to 
%UserProfile%\.pixi\bin. The command will also add %UserProfile%\.pixi\bin to your PATH environment variable, allowing 
you to invoke pixi from anywhere.
</details>

If you want to turn on autocompletion check the location of your profile by running in the
PowerShell: `$PROFILE`. At the end of this file add the following line:

```(& pixi completion --shell powershell) | Out-String | Invoke-Expression```
#### macOS/Linux
```
# Using curl
curl -fsSL https://pixi.sh/install.sh | sh

# Or using wget if you don't have curl
wget -qO- https://pixi.sh/install.sh | sh
```

Now restart your terminal or shell to make the installation effective. For what this does, see the windows section.
To enable autocompletion, add to your `.bashrc`:
```eval "$(pixi completion --shell bash)"```

### Installing Dependencies

Once pixi is installed, you can install the project dependencies:

```
# Install all dependencies
pixi install -a
```

This installs the dependencies for all environments related to this project.

## Running the workflow

### Overview
#### Splitting images into 1 scene per image file
The workflow starts from either raw `.lif` or `.tif` / `tiff` images. These are split into individual channel
images by running a CLI. To run with default parameters, in the command line ensure that you are in this repository directory and run: 

`pixi run split_images`

Additional parameters can be provided:
![split_lif](readme_images/split_help.png)

Both `experiments` and `channels` can be a list of strings. The way to provide multiple channels for example is:
`pixi run split_images -c ch1 -c ch2 -c ch3`

#### Preprocessing of images



Raw images (LIF / TIFF)
        ↓
1. split_lif
        ↓
2. preprocessing
        ↓
3. subcell
        ↓
4. hv_plot
        ↓
Interactive dashboard (Panel) and/or SVG figures


- split_lif.py, split raw image files containing multiple scenes.

- preprocessing, prepares to run subcell with preprocessing, segmenting, and mask post-processing.

- subcell, prepares the dataset for subcell, and runs it.

- hv_plot, creates an interactive plot and saves an svg file.