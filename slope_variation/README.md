# Slope variation experiment (RQ1)

To replicate the experiment, follow these steps.

1. First, make sure to follow the instructions in the main [README.md](../README.md) to install the required dependencies;
2. Move to the `slope_variation` folder:
    - `cd slope_variation`
3. Create the synthetic dataset by running this command:
    - `python synthetic_dataset.py`
    - The synthetic dataset is created in the `raw_data` folder;
4. Run the experiment by running this command:
    - `python slope_variation.py -m <ml_model>`, where `<ml_model>` can be `logreg` (for logistic regression) or `rf` (for random forest);
    - The results are saved in the `slopes_<ml_model>` folder, where `<ml_model>` can be `logreg` (for logistic regression) or `rf` (for random forest);
5. Run the `slope_variation.ipynb` notebook to generate the plots shown in Figure 1 of the paper.
6. Run the `anova_test.ipynb` notebook to replicate the ANOVA test.
