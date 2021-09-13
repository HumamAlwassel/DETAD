# Diagnosing Error in Temporal Action Detectors (DETAD)
This repository is intended to host the diagnosis tool for analyzing temporal action localization algorithms. It includes three analyses: false positive analysis, localization metric sensitivity analysis, and false negative analysis. Please take a look at our [DETAD paper](http://humamalwassel.com/publication/detad/), in which we first proposed this tool and presented an in-depth diagnosis of the state-of-the-art temporal action localization algorithms on both ActivityNet v1.3 and THUMOS14.

<img src="http://humamalwassel.com/publication/detad/featured.jpg">

If you find any piece of code valuable for your research, please cite this work:
```
@inproceedings{alwassel_2018_detad,
  title={Diagnosing Error in Temporal Action Detectors},
  author={Alwassel, Humam and Caba Heilbron, Fabian and Escorcia, Victor and Ghanem, Bernard},
  booktitle={The European Conference on Computer Vision (ECCV)},
  month={September}
  year={2018}
}
```

# How to run it?

> :warning: **DETAD current code is based on Python 2. Switching to Python 3 might result in _logical_ errors in the false postivie analysis.** Please either use the Python 2 environment we provide in environment.yml, or make the suggested fixes from this [pull request](https://github.com/HumamAlwassel/DETAD/pull/8) if you are using your own Pyhton 3 environment. In either case, you should verify that you can reproduce the same example plots in [sample_output](sample_output/).

Install conda, create the environment from the `environment.yml` file, and then activate the environment:
```
conda env create -f environment.yml
source activate detad
```

Run the false positive analysis (the other two analyses are run in a similar way):
```
cd src
python false_positive_analysis.py --ground_truth_filename ../data/activity_net_train_val_extra_characteristics.v1-3.min.json --subset validation --prediction_filename ../data/anet18_winner_validation.json --output_folder ../sample_output/validation
``` 

The code will produce PFD files with the analysis results in the `--output_folder` specified. Take a look at the `sample_output` folder for sample analysis results for a sample method on the validation and testing ActivityNet v1.3 dataset. We thank Tianwei Lin (the 2018 ActivityNet Challenge winner of the Temporal Action Localization Task) for providing us with a sample submission file on the ActivityNet dataset.

# What datasets can you use?

We provide augmented dataset files for the ActivityNet v1.3 (validation) and THUMOS14 dataset in the `data` folder. However, this tool is developed for any general dataset and any augmentation characteristics.  

# Who is behind it?

| <img src="http://activity-net.org/challenges/2018/images/humam.jpg" width="135" height="135"> | <img src="http://activity-net.org/challenges/2018/images/fabian.png" width="135" height="135"> | <img src="http://activity-net.org/challenges/2018/images/victor.png" width="135" height="135"> | <img src="http://activity-net.org/challenges/2018/images/bernard.jpg" width="135" height="135">  |
| :---: | :---: | :---: | :---: |
| Contributor | Contributor | Contributor | Advisor |
| [Humam Alwassel][web-humam] | [Fabian Caba][web-cabaf] | [Victor Escorcia][web-victor] | [Bernard Ghanem][web-bernard] |

# Do you want to contribute?

1. Check the open issues or open a new issue to start a discussion around your new idea or the bug you found
2. Fork the repository and make your changes
3. Send a pull request

[web-humam]: http://www.humamalwassel.com
[web-cabaf]: http://www.fabiancaba.com
[web-victor]: http://escorciav.github.io/
[web-bernard]: http://www.bernardghanem.com/
