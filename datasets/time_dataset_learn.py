from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from datasets.preprocessing import clean_text, preprocess_image


class TimeDatasetLearn(Dataset):
    """
    A PyTorch Dataset for loading multimodal data including images, texts, and timestamps.

    This dataset handles preprocessing steps such as ordering, windowing, and padding of input data.
    It supports computing time differences based on specified timestamp types and produces a sample
    dictionary containing processed images, texts, timestamps, and associated metadata.
    """

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: The number of users (samples) available in the dataset.
        """
        return len(self.users)

    def _emb_size(self, kind):
        """
        Return the embedding size for the specified modality.

        Args:
            kind (str): The type of embedding. Expected values are "image" or "text".

        Returns:
            int: The embedding size for the given modality.

        Note:
            For both "image" and "text", the embedding size is determined from
            `self.args.IMAGE_EMBEDDING_SIZES` using the key `self.args.image_embeddings_type`.
        """
        if kind == "image":
            return self.args.IMAGE_EMBEDDING_SIZES[self.args.image_embeddings_type]

        if kind == "text":
            return self.args.IMAGE_EMBEDDING_SIZES[self.args.image_embeddings_type]

    def load_multimodal(self, images, texts, dates, label, user_name, images_paths):
        """
        Load and preprocess multimodal data (images, texts, dates) and return a sample dictionary.

        This method orders the input data based on the dates provided (or applies a random permutation
        if position embeddings are set to "zero"), applies windowing and padding to maintain a consistent
        window size, transforms images to tensors, and computes time differences according to the
        specified timestamp kind.

        Args:
            images (list): List of image data.
            texts (list): List of text entries corresponding to each image.
            dates (list): List of timestamps corresponding to each data entry.
            label: The label associated with the data sample.
            user_name (str): The name or identifier of the user (author) of the data.
            images_paths (list): List of file paths for each image.

        Returns:
            dict: A dictionary containing preprocessed data with the following keys:
                - "author": The name of the user.
                - "images": The list of image tensors (including padded placeholder images if needed).
                - "images_paths": The list of image file paths (with placeholders for padded entries).
                - "texts": The list of text entries (padded as necessary).
                - "time": The processed timestamps or computed time differences.
                - "label": The provided label.
                - "padding_amount": The number of padded entries added.
                - "image_mask": A numpy array mask (reshaped) indicating valid (1) or padded (0) images.
                - "text_mask": A numpy array mask (reshaped) indicating valid (1) or padded (0) text entries.

        Notes:
            - If `self.args.position_embeddings` is not "zero", data is sorted by the `dates`; otherwise,
              a random permutation is applied.
            - If the number of available dates is less than the desired window size (`self.window_size`),
              padding is applied to reach the required length.
            - Timestamps are processed based on `self.args.timestamp_kind`:
                * "delta": Computes the difference between consecutive timestamps, prepended by 0,
                  and converts the result to hours.
                * "relative": Computes the difference from the minimum timestamp and converts it to hours.
            - Text entries and image data are padded with default placeholders if necessary:
                * Texts are padded with the string "<PAD>".
                * Images are padded with a placeholder image (a 224x224 black image).
                * Image paths are padded with the string "<PAD_PATH>".
            - Masks for images and texts are generated to indicate valid versus padded entries.
        """
        if self.args.position_embeddings != "zero":
            order_idx = np.argsort(dates).ravel()
        else:
            order_idx = np.random.permutation(len(images))

        high_ = len(dates) - self.window_size

        if high_ <= 0:
        # 貼文數不足 window_size：全部保留，再做 padding
            start_idx = 0
            end_idx = len(dates)
            padding_amount = abs(high_)
        else:
            padding_amount = 0

            # train：隨機取一段連續窗口
            if getattr(self, "kind", "train") == "train":
                start_idx = np.random.randint(low=0, high=high_ + 1)
                end_idx = start_idx + self.window_size

            # valid / test：固定取最後 window_size 篇
            else:
                start_idx = len(dates) - self.window_size
                end_idx = len(dates)

        # Apply ordering to indices
        idxs = order_idx[start_idx:end_idx]

        # Use reordered indices to slice your data
        texts = [clean_text(texts[i]) for i in idxs]
        dates = [dates[i] for i in idxs]
        images = [images[i] for i in idxs]
        images_paths = [images_paths[i] for i in idxs]

        # Create the image_mask array (1 for valid images, 0 for "Blank image")
        image_mask = np.array(
            [0 if path == "Blank image" else 1 for path in images_paths]
        )

        # Empty posts after cleaning must not contribute to text attention.
        text_mask = np.array([1 if text else 0 for text in texts])

        # Process only valid timestamps. Padding before this step would make
        # relative time start at -1 and create a large artificial delta.
        dates = np.asarray(dates, dtype=np.float64)
        if self.args.timestamp_kind == "delta":
            dates = np.hstack(([0], np.diff(dates))) / 60 / 60
        elif self.args.timestamp_kind == "relative":
            dates = (dates - np.min(dates)) / 60 / 60  # Convert to hours difference
        dates = dates.tolist() + [0.0] * padding_amount

        # Apply padding to texts and images at the end if necessary
        if padding_amount > 0:
            # Pad texts with a specific padding token (e.g., "<PAD>")
            texts = texts + ["<PAD>"] * padding_amount
            text_mask = np.pad(
                text_mask, (0, padding_amount), "constant", constant_values=0
            )
            placeholder_image, _ = preprocess_image(
                None,
                training=getattr(self, "kind", "train") == "train",
            )
            images = images + [placeholder_image] * padding_amount
            images_paths = images_paths + ["<PAD_PATH>"] * padding_amount
            image_mask = np.pad(
                image_mask, (0, padding_amount), "constant", constant_values=0
            )

        sample = {
            "author": user_name,
            "images": images,
            "images_paths": images_paths,
            "texts": texts,
            "time": dates,
            "label": label,
            "padding_amount": padding_amount,
            "image_mask": image_mask.astype(np.float32).reshape(
                (-1, 1, self.window_size)
            ),
            "text_mask": text_mask.astype(np.float32).reshape(
                (-1, 1, self.window_size)
            ),
        }

        return sample
