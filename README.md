# MLDE Notebooks

Evalute ML approaches to emulate a CPM and downscale rainfall.

## Setup

This guide assumes you are using conda (or mamba) to manage packages and python environments.

1. Install conda environment:
  * If you wish to re-use the exact environment: `conda env create --file environment.lock.yml` and activate it: `conda activate mlde-notebooks`
  * OR install the needed conda and pip packages to your own environment: `conda install --file environment.txt`
2. Install this package (including a few pip dependencies which may not have been included in the previous step): `pip install -e .`
3. Create .env file: cp .env.example .env and then update to match your needs:
  * `DERIVED_DATA`: path to where derived data such datasets and model artefacts are kept

### Updating conda environment

To add new packages or update their version, it is recommended to use the `environment.txt` file (for conda packages) and `requirements.txt` file (for pip packages) then run:
```sh
conda env install -f environment.txt
pip install -e . # this will implicitly use requirement.txt
conda env export -f environment.lock.yml
```
then commit any changes (though make sure not to include mlde-notebooks package in the lock file since that is not distributed via PyPI).

To sync environment with the lock file use:
```sh
conda env update -f environment.lock.yml --prune
```

## Configuration

### Environment variables

Manage environment variables as you wish though the python-dotenv package is included for those wishing to use `.env` file.

| Name | Description |
|------|-------------|
|`DERIVED_DATA`| The common path to where datasets and emulator artefacts (including samples) are to be found |


## Usage

### Running Notebooks

The main expected way to run notebooks with full datasets is via batch jobs running a helper script that uses papermill but it is possible to run the notebooks interactively (and this is the expected way to develop and test new data analysis).

#### Batch

This can be used with a helper script to run a notebook in batch mode along with an (optional) parameter files: `bin/run-notebook nbs/my/notebook.ipynb /output/base/dir/ nbs/my/parameters/setA.yml`.
See `run-notebook` file itself for further details.

There are further helpers (`queue-run-notebook`) for doing this on Blue Pebble and JASMIN in `bin/bp` and `bin/jasmin` respectively. These include support for options like memory and time requirements or partition.

#### Interactive

Use jupyter: `jupyter lab`

### CLI

There is also commandline interface that uses the same parameter sets to analyze data and/or produce summaries of the climate model data and raw emulator prediction files. e.g.

```sh
mlde-nbs hist2d seasonsal nbs/evaluation/output/mv-test/hist2d-rh-temp.nc --xvar relhum150cm --yvar tmean150cm --params-file nbs/evaluation/parameters/mv-test.yml
```

Again there are helper scripts (`queue-job`) in `bin/bp` and `bin/jasmin` respectively for using commands like this via the batch systems on Blue Pebble and JASMIN. These include support for options like memory and time requirements or partition.

## Development

There are a set of development dependencies if you do more than just tweak a notebook: `pip install -e '.[dev]'`
