import html
import math
import re
import unicodedata

from PIL import Image
import torchvision.transforms as T


CLIP_IMAGE_SIZE = 224
CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_MENTION_RE = re.compile(r"(?<!\w)[@＠][\w.]+", re.UNICODE)
_HASHTAG_RE = re.compile(r"(?<!\w)[#＃]\w+", re.UNICODE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value):
    """Remove social-media markup and normalize a post to plain text."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if value == "<PAD>":
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(" ", text)
    text = "".join(
        " " if unicodedata.category(char).startswith("C") else char
        for char in text
    )
    return _WHITESPACE_RE.sub(" ", text).strip()


def preprocess_image(
    image_path,
    image_size=CLIP_IMAGE_SIZE,
    image_mean=CLIP_IMAGE_MEAN,
    image_std=CLIP_IMAGE_STD,
    training=False,
):
    """Load an image and apply the CLIP train or evaluation transform."""
    original_path = image_path
    if image_path is None:
        image = Image.new("RGB", (image_size, image_size), color=(0, 0, 0))
        original_path = "Blank image"
    else:
        try:
            with Image.open(image_path) as opened_image:
                image = opened_image.convert("RGB")
        except (OSError, ValueError):
            image = Image.new("RGB", (image_size, image_size), color=(0, 0, 0))
            original_path = "Blank image"

    spatial_transforms = (
        [
            T.RandomResizedCrop(
                image_size,
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(),
        ]
        if training
        else [
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
        ]
    )
    transform = T.Compose(
        spatial_transforms
        + [
            T.ToTensor(),
            T.Normalize(mean=image_mean, std=image_std),
        ]
    )
    return transform(image), original_path
