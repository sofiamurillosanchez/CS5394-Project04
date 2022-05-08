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
import pandas as pd
import praw
from pathlib import Path
from autocorrect import Speller
import re
import markovify
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import matplotlib.font_manager as fm

_REDDIT_CLIENT = None


def _open_image(filepath=None, url=None, raw=None):
    """
    Opens an image from a local filepath or a remote URL
    :param filepath: str, relative or absolute filepath
    :param url: str, URL to image
    :return: Image as cast to a numpy.array
    """
    # Error handling to verify filepath or url is provided
    if filepath is None and url is None and raw is None:
        raise AttributeError('Function _open_image must accept exactly one argument, either filepath or url')

    # Open image file
    if filepath:
        image = Image.open(filepath)
    elif url:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(BytesIO(raw))

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
        kp_image = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0),
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return kp, des, kp_image
    else:
        return kp, des


def get_image_matches(des1, des2, get_match_image=False, img1=None, img2=None, kp1=None, kp2=None):
    if get_match_image and (img1 is None or img2 is None or kp1 is None or kp2 is None):
        raise ValueError('To product a match image, original images img1 and img2, along with key points kp1 and kp2, '
                         'must be provided')

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if get_match_image:
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return matches, match_img

    return matches


def build_meme_repo(subreddits, save_as=None, get_large_set=True):
    image_repo = pd.DataFrame(columns=['subreddit', 'url', 'text', 'features'])
    for subreddit in subreddits:
        urls = get_top_image_urls_from_subreddit(subreddit, get_large_set=get_large_set)

        for url in urls:
            try:
                text, image = read_image(url=url, get_image=True)
                if text:
                    _, des = get_image_features(image)
                    image_repo.loc[len(image_repo.index)] = [subreddit, url, text, des]
            except Exception:
                continue

    spell = Speller()
    image_repo.text = image_repo.text.apply(lambda x: re.sub(r'[^a-zA-Z\s.\'-]', '', x)).apply(
        lambda x: re.sub(r'\s', ' ', x)).apply(spell).apply(str.upper)
    image_repo = image_repo[image_repo.text.apply(lambda x: re.sub('[^A-Z]', '', x)) != '']

    if save_as:
        with open(save_as, 'wb') as f:
            pickle.dump(image_repo, f)

    return image_repo


def open_meme_repo(filepath):
    with open(filepath, 'rb') as f:
        repo = pickle.load(f)
    return repo


def get_caption_model(repo, img_url=None, img_filepath=None, img_raw=None, sample=100, get_top_match_urls=False):
    if img_filepath:
        image = _open_image(filepath=img_filepath)
    elif img_url:
        image = _open_image(url=img_url)
    elif img_raw:
        image = _open_image(raw=img_raw)

    _, des = get_image_features(image)

    df = repo.copy().sample(frac=1)
    df['feature_matches'] = df.features.apply(lambda x: len(get_image_matches(x, des)))

    best_matches = df.sort_values(['feature_matches'], ascending=False).head(sample)
    best_match_captions = best_matches.text

    models = []
    for caption in best_match_captions:
        models.append(markovify.Text(caption, well_formed=False))

    model = markovify.combine(models, list(range(sample, 0, -1)))

    if get_top_match_urls:
        top_match_urls = best_matches.head().url
        return model, top_match_urls

    return model


def _draw_line(text, x, y, draw, font):
    draw.text((x-2, y-2), text, (0, 0, 0), font=font)
    draw.text((x+2, y-2), text, (0, 0, 0), font=font)
    draw.text((x+2, y+2), text, (0, 0, 0), font=font)
    draw.text((x-2, y+2), text, (0, 0, 0), font=font)
    draw.text((x, y), text, (255, 255, 255), font=font)


def _draw_text(img, text, draw, pos, font):
    w, h = draw.textsize(text, font)

    line_count = 1
    if w > img.width:
        line_count = int(round((w / img.width) + 1))

    lines = []
    if line_count > 1:
        last_cut = 0
        is_last = False
        for i in range(0, line_count):
            if last_cut == 0:
                cut = int((len(text) / line_count) * i)
            else:
                cut = int(last_cut)

            if i < line_count-1:
                next_cut = int((len(text) / line_count) * (i+1))
            else:
                next_cut = int(len(text))
                is_last = True

            if not (next_cut == len(text) or text[next_cut] == ' '):
                try:
                    while text[next_cut] != ' ':
                        next_cut += 1
                except Exception as e:
                    print(text)
                    print(len(text))
                    print(next_cut)
                    print(next_cut == len(text))

                    raise e

            line = text[cut:next_cut].strip()

            w, h = draw.textsize(line, font)
            if not is_last and w > img.width:
                next_cut -= 1
                while text[next_cut] != ' ':
                    next_cut -= 1

            last_cut = next_cut
            lines.append(text[cut:next_cut].strip())

    else:
        lines.append(text)

    last_y = -h
    if pos == 'bottom':
        last_y = img.height - h * (line_count + 1) - 10

    for i in range(0, line_count):
        w, h = draw.textsize(lines[i], font)
        x = (img.width / 2) - (w / 2)
        y = last_y + h
        _draw_line(lines[i], x, y, draw, font)
        last_y = y


def make_meme(img_url=None, img_filepath=None, img_raw=None, top_text=None, bottom_text=None, width=1125):
    if img_url:
        response = requests.get(img_url)
        image = Image.open(BytesIO(response.content))
    elif img_filepath:
        image = Image.open(img_filepath)
    elif img_raw:
        image = Image.open(BytesIO(img_raw))
    else:
        raise ValueError('img_url, img_filepath, or img_raw must be provided')

    h_old, w_old = image.size
    scale = width / w_old
    h_new, w_new = int(h_old * scale), int(w_old * scale)
    image = image.resize((h_new, w_new))

    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='Impact')), 100)
    if top_text:
        _draw_text(image, top_text, draw, 'top', font)
    if bottom_text:
        _draw_text(image, bottom_text, draw, 'bottom', font)

    return image


def generate_memes(repo, img_url=None, img_filepath=None, img_raw=None, sample=50, number_of_memes=5, get_top_match_urls=False):
    try:
        if not img_url and not img_filepath and not img_raw:
            raise ValueError('img_url or img_filepath must be provided')

        for i in range(3):
            if get_top_match_urls:
                model, urls = get_caption_model(repo, img_url=img_url, img_filepath=img_filepath, img_raw=img_raw,
                                                sample=sample, get_top_match_urls=True)
            else:
                model = get_caption_model(repo, img_url=img_url, img_filepath=img_filepath, img_raw=img_raw,
                                          sample=sample, get_top_match_urls=False)
            captions = {model.make_short_sentence(300) for _ in range(number_of_memes)}
            try:
                captions.remove(None)
            except Exception:
                pass
            if len(captions) > 0:
                break
        else:
            print('Oops, no captions could be generated.')
            return None

        splits = [
            (
                ' '.join(caption.split(' ')[:int(len(caption.split(' ')) / 2)]),
                ' '.join(caption.split(' ')[int(len(caption.split(' ')) / 2):])
            ) for caption in captions
        ]

        memes = [
            make_meme(img_url=img_url, img_filepath=img_filepath, img_raw=img_raw, top_text=top, bottom_text=bottom)
            for top, bottom in splits
        ]

        if get_top_match_urls:
            return memes, urls
        return memes

    except Exception as e:
        print('Oops, something went wrong:')
        raise e
