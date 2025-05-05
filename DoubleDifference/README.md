# Graph Double Difference (GraphDD)

A graph neural based earthquake relocation framework.

## Applying the model

First run "build_initial_files.py" to convert the catalog earthquaked data to one reference file

Then run "build_subsets_of_paired_sources.py" to build the sets of training input graphs.

Then "train_double_difference_model.py" to train the double difference model.

These scripts assume you have setup a GENIE environment for a study region of interest, and have trained the travel time neural network for a chosen velocity model. The catalog files also have to be formated in the GENIE format in the corresponding "Catalog/%d"%year folders.

Different training loss terms can be selected in "train_double_difference_model.py", and different graph construction parameters in "build_subsets_of_paired_sources.py".

## Extra Information

The method is described in https://arxiv.org/abs/2410.19323v1.
