# CS5394-Project04

**Link to Writeup:** https://github.com/sofiamurillosanchez/CS5394-Project04/blob/main/P04%20Writeup.pdf 

## Running the Streamlit App
* The Streamlit App is currently running off an AWS EC2 instance because of memory constrictions of other options, so it doen't have a static location. (It could crash if multiple people try to use it at once.)
* To run it locally, it requires access to the corpus of scraped memes (about 6 GB worth of captions and image features) so we'll have to figure out a way to transfer the file. It's hosted on S3, so we could figure it out. (It takes about an hour and a half to do all of the scraping, parsing, and processing of the subreddits to build the corpus, but that's always an option if ya want.)
* To run, save the corpus file as `meme_repo.pkl` in the project directory, then simply run `streamlit run app.py` to launch the app. 

## Usage Requirements to run the Sandbox Notebook
* Install Tesseract: https://tesseract-ocr.github.io/tessdoc/Installation.html
* Configure environment using `conda env create -f environment.yaml`
  * Optionally, update current environment using `conda env update -f environment.yaml`
* Save `reddit_auth.pickle` file to the project directory

* (If you make changes to the environment at all (like using other libraries), make sure to update the environment configuration file before you push any changes. (`conda env export > environment.yaml`) Then, open the `environment.yaml` file and delete the very last line that says `prefix: /some/file/path`
