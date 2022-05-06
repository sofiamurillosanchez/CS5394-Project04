from PIL import Image
import pytesseract
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from operator import xor
import pickle
import praw
from pathlib import Path

_REDDIT_CLIENT = None


def _open_image(filepath=None, url=None):
    """
    Opens an image from a local filepath or a remote URL
    :param filepath: str, relative or absolute filepath
    :param url: str, URL to image
    :return: Image as cast to a numpy.array
    """
    # Error handling to verify filepath or url is provided
    if not xor(bool(filepath), bool(url)):
        raise AttributeError('Function _open_image must accept exactly one argument, either filepath or url')

    # Open image file
    if filepath:
        image = Image.open(filepath)
    else:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))

    # Cast to numpy array and return
    return np.array(image)


def _preprocess_image(image):
    """
    Performs preprocessing using OpenCV to optimize OCR reading
    :param image: numpy.array representation of an image file
    :return: Processed image as a numpy.array
    """
    try:
        # Perform bilateral filtering and color correction to image
        image = cv2.bilateralFilter(image, 5, 55, 60)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, 240, 255, 1)

        # Return processed image
        return image

    except Exception:
        # Sometimes, OpenCV cannot decipher the image data, in which case return None
        return None


def read_image(filepath=None, url=None, get_image=False):
    """
    Opens an image and reads text using Tesseract
    :param filepath: str, relative or absolute filepath
    :param url: str, URL to image
    :param get_image: optional boolean to return
    :return: str, text as read from image; if get_image is true, tuple containing image text and np.array of image
    """
    # Error handling to verify filepath or url is provided
    if not xor(bool(filepath), bool(url)):
        raise AttributeError('Function read_image must accept exactly one argument, either filepath or url')

    # Open image and preprocess
    image = _open_image(filepath=filepath) if filepath else _open_image(url=url)
    processed = _preprocess_image(image)

    # If unable to process image, return nothing
    if processed is None:
        return None if not get_image else (None, None)

    # Return text read using Tesseract
    return pytesseract.image_to_string(processed) if not get_image else (pytesseract.image_to_string(processed), image)


def show_image(image):
    """
    Displays image
    :param image: numpy.array representation of image
    :return: None
    """
    if image is not None:
        # Configure matplotlib to show image
        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()


def _authorize_reddit_client():
    """
    Authorizes PRAW Reddit client using credentials found in reddit_auth.pickle
    :return: None
    """
    # Verify existance of reddit_auth.pickle
    auth_file = Path('reddit_auth.pickle')
    if not auth_file.is_file():
        raise EnvironmentError('Valid reddit_auth.pickle file required in project directory to authorize '
                               'reddit client')

    # Read API credentials
    pickle_off = auth_file.open('rb')
    auth = pickle.load(pickle_off)
    pickle_off.close()

    # Authorize Reddit client object
    global _REDDIT_CLIENT
    _REDDIT_CLIENT = praw.Reddit(**auth)


def get_top_image_urls_from_subreddit(subreddit_name, get_large_set=True):
    """
    Returns URLs of static images identified in the top posts of a provided subreddit
    :param subreddit_name: str, name of a subreddit
    :param get_large_set: boolean, will use top 1000 posts if True, will use top 100 posts if False
    :return: str array containing all identified image URLs
    """
    # Check that Reddit client is authorized
    if not _REDDIT_CLIENT:
        _authorize_reddit_client()

    # Initialize subreddit object
    subreddit = _REDDIT_CLIENT.subreddit(subreddit_name)

    # Define subreddit function arguments and return all static image URLs
    post_args = {'time_filter': 'all', 'limit': None} if get_large_set else {'time_filter': 'all'}
    return [post.url for post in subreddit.top(**post_args) if post.url[-4:] in ['.jpg', '.png']]


def get_image_features(image, get_feature_image=False):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(grayscale, None)
    if get_feature_image:
        kp_image = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return kp, des, kp_image
    else:
        return kp, des


def get_image_matches(des1, des2, get_match_image=False, img1=None, img2=None, kp1=None, kp2=None):
    if get_match_image and (img1 is None or img2 is None or kp1 is None or kp2 is None):
        raise ValueError('To product a match image, original images img1 and img2, along with key points kp1 and kp2, must be provided')

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if get_match_image:
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return matches, match_img

    return matches
