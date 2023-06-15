# fire-forecast
Tool for forecast of fire radiative power.

## Contribution
### 1. Clone the repository with:
```
git clone git@github.com:ECMWFCode4Earth/fire-forecast.git
```
### 2. Install the conda environment
To create the environment switch to the `fire-forcast` folder created by the `git clone` above and execute:
```
conda env create -f ci/environment.yaml
```
This will install `poetry` as well as the dependencies that are not installable with `poetry` in the new environment called `fire-forecast` (Further packages will then be managed by poetry in this environment).

### 3. Activate the environment with:
```
conda activate fire-forecast
```
### 4. Install the fire-forecast module with poetry
To install the other dependencies of the project run the following from the `fire-forcast` folder created by the `git clone`:
```
poetry install 
```
This will install all dependencies of `fire-forecast` and the module itself.
### 5. Install the pre-commit hook to automatically format and test your code before a commit:
```
pre-commit install
```

**Done!**
### Remarks
 * To test the installation you can run:
   * `pytest` in the `fire-forcast` folder created by the `git clone`
   * `pre-commit run --all-files` tests the pre-commit hook. `black` `isort` and `flake8` should be installed and the hook should be run and show: "Passed" or "Skipped" for the different modules.
 * To contribute within the project create a new branch with:
  `git checkout -b branch_name`
   and use a pull request on github.com to merge with the main branch, when you are done with the new feature.
 * To add a new dependency with `poetry`:
   * `poetry add package name` (works similar to `conda install` or `pip install`)   
