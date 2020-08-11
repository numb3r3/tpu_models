
import tensorflow as tf
import transformers
import numpy as np
import math
from transformers.tokenization_bert import BertTokenizer

from tpu_models.models.gpt2_tf import load_or_init_model, train
from tpu_models.utils import load_yaml, set_seed
from tpu_models.config import Config

def load_dataset(path):
    texts = []
    for line in open(path, "r", encoding="utf-8"):
        texts.append(line.strip("\n"))
    return texts


def build_tokenizer(tokenizer_name_or_path):
    tokenizer = BertTokenizer.from_pretrained(
        tokenizer_name_or_path, do_lower_case=True
    )
    return tokenizer


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


def get_dataset(filenames, max_seq_len, batch_size, is_train: bool = True):
    AUTO = tf.data.experimental.AUTOTUNE
    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([max_seq_len], tf.int64),
    }
    def parse_tfrecord(example):
        
        example = tf.io.parse_single_example(example, name_to_features)
        input_ids= example["input_ids"]
        # return input_ids[:-1], input_ids[1:]
        # TODO: consider `attention_mask`
        return {"input_ids": input_ids[:-1], "label": input_ids[1:]}
        
    
    # Read from TFRecords. For optimal performance, we interleave reads from multiple files.
    records = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = records.map(parse_tfrecord, num_parallel_calls=AUTO)
  
    if is_train:
        dataset = dataset.shuffle(2048)

    # Prefetch the next batch while training (autotune prefetch buffer size).
    return dataset.batch(batch_size, drop_remainder=True).prefetch(AUTO) 


def main(config):
    params = Config(**load_yaml(config))
    print(params)

    set_seed(params.train.seed)

    # Build and save tokenizer
    # Build and save tokenizer
    tokenizer = build_tokenizer(params.input.tokenizer_file)

    # bucket_name="pretrained_dataset"
    # # gcs_pattern = 'gs://flowers-public/tfrecords-jpeg-331x331/*.tfrec'
    # gcs_pattern = f'gs://{bucket_name}/clue_datasets/*.tfrec'
    validation_split = 0.1
    filenames = tf.io.gfile.glob(params.input.train_file)
    split = len(filenames) - int(len(filenames) * validation_split)
    train_fns = filenames[:split]
    validation_fns = filenames[split:]

    # train_texts = load_dataset(params.input.train_file)
    # valid_texts = load_dataset(params.input.valid_file)

    
    
    # Tokenizer.from_file(params.input.tokenizer_file)
    # tokenizer.save(params.output.tokenizer_file)
    # tokenizer = TokenizerWrapper(tokenizer)

    # # Build data
    # train_dataset = Dataset(
    #     tokenizer, train_texts, params.train.block_size, params.train.batch_size
    # )
    # valid_dataset = Dataset(
    #     tokenizer, valid_texts, params.train.block_size, params.train.batch_size
    # )
    train_dataset = get_dataset(train_fns, max_seq_len=128, batch_size=params.train.batch_size, is_train=True)
    valid_dataset = get_dataset(validation_fns, max_seq_len=128, batch_size=params.train.batch_size)

    # Train model
    model = load_or_init_model(
        pretrained_model_dir=params.input.pretrained_model_dir,
        vocab_size=len(tokenizer),
        params=params.model_params,
    )
    val_best_model = train(params, model, tokenizer, train_dataset, valid_dataset)
    val_best_model.summary()

    # Evaluate best model with validation set
    val_best_model.evaluate(valid_dataset)


if __name__ == "__main__":
    import fire

    fire.Fire(main)