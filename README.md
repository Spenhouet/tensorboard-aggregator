# tensorboard-aggregator

This project contains an easy to use method to aggregate multiple tensorboard runs. The max, min, mean, median, standard deviation and variance of the scalars from multiple runs is saved either as new tensorboard summary or as `.csv` table.

There is a similar tool which uses pytorch to output the tensorboard summary: [TensorBoard Reducer](https://github.com/janosh/tensorboard-reducer)

## Feature Overview

- Aggregates scalars of multiple tensorboard files
- Saves aggregates as new tensorboard summary or as `.csv`
- Aggregate by any numpy function (default: max, min, mean, median, std, var)
- Allows any number of subpath structures
- Keeps step numbering
- Saves wall time average per step

## Setup and run configuration

1. Download or clone repository files to your computer
1. Go into repository folder
1. Install requirements: `pip3 install -r requirements.txt --upgrade`
1. You can now run the aggregation with: `python aggregator.py`

### Parameters

| Parameter    |          | Default                   | Description |
| ------------ | -------- | ------------------------- | ----------- |
| _--path_     | optional | current working directory | Path to folder containing runs |
| _--subpaths_ | optional | `['.']`       | List of all subpaths |
| _--output_   | optional | `summary`                 | Possible values: `summary`, `csv` |

### Recommendation

- Add the repository folder to the PATH (global environment variables).
- Create an additional script file within the repository folder containing `python static/path/to/aggregator.py` 
    - Script name: `aggregate.sh` / `aggregate.bat` / ... (depending on your OS)
    - Change default behavior via parameters
    - Do not change `path` parameter since this will by default be the path the script is run from
- Workflow from here: Open folder with tensorboard files and call the script: aggregate files will be created for the current directory

## Explanation

Example folder structure:

    .
    ├── ...
    ├── test_param_xy      # Folder containing the runs for aggregation
    │   ├── run_1          # Folder containing tensorboard files of one run
    │   │   ├── test       # Subpath containing one tensorboard file
    │   │   │   └── events.out.tfevents. ...
    │   │   └── train   
    │   │       └── events.out.tfevents. ...
    │   ├── run_2
    │   ├── ...
    │   └── run_X
    └── ...

The folder `test_param_xy` will be the base path (`cd test_param_xy`).
The tensorboard summaries for the aggregation will be created by calling the `aggregate` script (containing: `python static/path/to/aggregator.py --subpaths ['test', 'train'] --output summary`)

The base folder contains multiple subfolders. Each subfolder contains the tensorboard files of different runs for the same model and configuration as all other subfolders.

The resulting folder structure for `summary` looks like this:

    .
    ├── ...
    ├── test_param_xy
    │   ├── ...
    │   └── aggregate
    │       ├── test
    │       │   ├── max
    │       │   │   └── test_param_xy 
    │       │   │       └── events.out.tfevents. ...
    │       │   ├── min
    │       │   ├── mean
    │       │   ├── median
    │       │   └── std    
    │       └── train
    └── ...

Multiple aggregate summaries can be put together in one directory.
Since the original base folder name is kept as subfolder to the aggregate function folder the summaries are distinguishable within tensorboard.

    .
    ├── ...
    ├── max
    │   ├── test_param_x
    │   ├── test_param_y
    │   ├── test_param_z
    │   └── test_param_v 
    ├── min
    ├── mean
    ├── median
    └── std   


The `.csv` table files for the aggregation will be created by calling the `aggregate` script (containing: `python static/path/to/aggregator.py --subpaths ['test', 'train'] --output csv`)

The resulting folder structure for `summary` looks like this:

    .
    ├── ...
    ├── test_param_xy
    │   ├── ...
    │   └── aggregate
    │       ├── test
    │       │   ├── max_test_param_xy.csv
    │       │   ├── min_test_param_xy.csv
    │       │   ├── mean_test_param_xy.csv
    │       │   ├── median_test_param_xy.csv
    │       │   └── std_test_param_xy.csv
    │       └── train
    └── ...

The `.csv` files are primarily for latex plots.

## Limitations

- The aggregation only works for scalars and not for other types like histograms 
- All runs for one aggregation need the exact same tags. Basically the naming and number of scalar metrics needs to be equal for all runs.
- All runs for one aggregation need the same steps. Basically the number of iterations, epochs and the saving frequency needs to be equal for all runs of one scalar.

## Contributions

If there are potential problems (bugs, incompatibilities to newer library versions or to a OS) or feature requests, please create an GitHub issue [here](https://github.com/Spenhouet/tensorboard-aggregator/issues).

Dependencies are managed using [pip-tools](https://github.com/jazzband/pip-tools).
Just add new dependencies to `requirements.in` and generate a new `requirements.txt` using `pip-compile` in the command line.

## License

[MIT License](LICENSE)
