# SUB-PLAY
## Installation

python = 3.8.8

gym = 0.26.1

pettingzoo = 1.17.0

pygame = 2.1.2

## Replacement of Files
After installing the required environment, you need to replace two specific files for proper functionality. Follow these steps:

1. **Replace `clip_out_of_bounds.py`:**
    - Navigate to the following path in your Anaconda environment: 
      ```
      Anaconda\envs\your_conda_environment_name\Lib\site-packages\pettingzoo\utils\wrappers
      ```

2. **Replace `simple_tag.py`:**
    - Go to the following directory in your Anaconda environment: 
      ```
      Anaconda\envs\your_conda_environment_name\Lib\site-packages\pettingzoo\mpe\scenarios
      ```

Make sure to perform these replacements to ensure the correct functioning of the environment.

**Note 1:** Replace `your_conda_environment_name` with the actual name of your Conda environment.

**Note 2:** The `clip_out_of_bounds.py` and `simple_tag.py` used for replacement can be found in "materials".
