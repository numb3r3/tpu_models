import math

import tensorflow as tf


class Dataset(tf.keras.utils.Sequence):
    def __init__(self, tokenizer, texts, block_size, batch_size):
        ids = []
        for text in texts:
            ids.extend(tokenizer.encode(text, add_special_tokens=False))

        samples = []
        for idx in range(0, len(ids) - block_size + 1, block_size):
            sample = ids[idx : idx + block_size]
            samples.append(sample)

        # Define attributes
        self._batch_size = batch_size
        self._samples = samples

    def __getitem__(self, idx):
        inputs = []
        labels = []

        for i in range(
            idx * self._batch_size,
            min((idx + 1) * self._batch_size, len(self._samples)),
        ):
            sample = self._samples[i]
            inputs.append(sample[:-1])
            labels.append(sample[1:])

        return {"input_ids": np.array(inputs)}, np.array(labels)

    def __len__(self):
        return math.ceil(len(self._samples) / self._batch_size)
