# Overview
This README is intended for setting up tsne analysis following the DANNCE prediction.
Please refer to the [DANNCE prediction protocol](https://github.com/Sooophy/dannce/tree/stroke_analysis/trace_protocol) for precious steps.

## Install your environment
Please install your environment following the README [here](https://github.com/Sooophy/neuroposelib/blob/stroke_analysis_sophie/README.md).

## Folder structure
Now you already have a folder structure like this for the DANNCE prediction.
```
experiment collection folder
    |
    experiment1 (contains all the videos for the series of the experiment)
        |
        20240612_c1_m1 (YYYYMMDD_cX_mX)
        20240612_c1_m1_cylinder (YYYYMMDD_cX_mX_cylinder)
        20240612_c1_m2
        ...
    |
    experiment2
    |
    experiment3
```

Now create a new folder under your netid_folder, you can call it stroke_exp here.
Organize your folder to the following structure:

```
experiment collection
    |
    stroke_exp
        |
        configs
            |
            stroke_exp.yaml
        data
            |
            metadata.csv
        stroke_exp.py
        stroke_exp.sh
        preprocess_for_tsne.py
```

## Prepare your metadata file <a id="metadata"></a>
We have provided an example `metadata.csv` for your reference. You will need your own file that contains your experiment information.

Note that the videoID need to be unique for each row. You can leave the `ClusterDirectory` blank or simply not have this column, since this will be generated automatically after running the follwoing script (if you follow the naming rules). You can add more columns if you have more categories to compare. 

## Data Processing
First, activate your environment, run:

 `conda activate neuroposelib`

You should have your `metadata.csv` ready following the instruction described in the [above section](#metadata).

Then, change path in the script and run:

`python preprocess_for_tsne.py`. 


Prepare your `stroke_exp.yaml`. See comments inside the config file for more information of each parameters. When you have the parameters,run:

`sh stroke_exp.sh`

This will request memory and run `stroke_exp.py`. Remember to change the config path in `stroke_exp.py`. You can check `stroke_exp.out` for the running result.

## Authors
- **Sophie Shi** - sophie.shi@duke.edu
- **Joshua Wu** - joshua.wu@duke.edu