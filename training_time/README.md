# Training time prediction experiment (RQ2)

To replicate the experiment, follow these steps.

1. First, make sure to follow the instructions in the main [README.md](../README.md) to install the required dependencies;
2. Move to the `training_time` folder:
    - `cd training_time`
3. Get the list of ML model training times by running these commands:
    - `python get_training_times.py -m <ml_model>`, where `<ml_model>` can be `logreg` (for logistic regression) or `rf` (for random forest);
    - The training times are saved in the `training_times_<ml_model>` folder, where `<ml_model>` can be `logreg` (for logistic regression) or `rf` (for random forest);
4. Run the `errors_logreg-ipynb` and `errors_rf.ipynb` notebooks to generate the plots shown in Figure 2 and 3 of the paper.
