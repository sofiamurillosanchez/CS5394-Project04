# CS5394-Project04

## Usage Requirements
* Install Tesseract: https://tesseract-ocr.github.io/tessdoc/Installation.html
* Configure environment using `conda env create -f environment.yaml`
  * Optionally, update current environment using `conda env update -f environment.yaml`
* Save `reddit_auth.pickle` file to the project directory

If you make changes to the environment at all (like using other libraries), make sure to update the environment configuration file before you push any changes.
```bash
conda env export > environment.yaml
```

Then, open the `environment.yaml` file and delete the very last line that says `prefix: /some/file/path`