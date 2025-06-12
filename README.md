# MLDE Notebooks

Evalute ML approaches to emulate a CPM and downscale rainfall.

## Setup

Assumes you have [pixi](https://pixi.sh) installed for managing dependencies.

1. Clone repo and cd into it
2. Install dependencies: `pixi install`
3. Create .env file: cp .env.example .env and then update to match your needs:
  * `DERIVED_DATA`: path to where derived data such datasets and model artefacts are kept

### Updating conda environment

To add new packages or update their version, can update the dependencies in pixi.toml then run
```sh
pixi install
```
or add them using:
```sh
pixi add NEW_DEP
```
then commit any changes to pixi.toml and pixi.lock

## Configuration

### Environment variables

Manage environment variables as you wish though the python-dotenv package is included for those wishing to use `.env` file.

| Name | Description |
|------|-------------|
|`DERIVED_DATA`| The base path to where datasets are to be found |
|`WORKDIRS_PATH`| The common path to where emulator artefacts (including samples) are to be found |


## Usage

It is expected to prepend each command with `pixi run` to ensure getting the expected environment.

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

There are a set of development dependencies if you do more than just tweak a notebook.
Use the dev pixi environment (e.g. `pixi run -e dev pytest`).
