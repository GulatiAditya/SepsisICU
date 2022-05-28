# SepsisICU
# Early-stage sepsis prediction from Intensive Care Unit (ICU) data (Work In Progress Due June 8, 2022)

CS229 Machine Learning project by Aditya Gulati and Niveditha Lakshmi Narasimhan.

## Description of codefiles

* `get_sepsis_score.py` makes predictions on clinical time-series data.  Add your prediction code to the `get_sepsis_score` function.  To reduce your code's run time, add any code to the `load_sepsis_model` function that you only need to run once, such as loading weights for your model.
* `driver.py` calls `load_sepsis_model` once and `get_sepsis_score` many times. It also performs all file input and output.  **Do not** edit this script -- or we will be unable to evaluate your submission.

All notebooks provide analysis of the dataset


## Use

You can run this prediction code by installing the NumPy package and running

    python driver.py input_directory output_directory

where `input_directory` is a directory for input data files and `output_directory` is a directory for output prediction files. 
