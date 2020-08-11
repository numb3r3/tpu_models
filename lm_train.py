
import tensorflow as tf
import transformers
import numpy as np
import math
from transformers.tokenization_bert import BertTokenizer

from tpu_models.models.gpt2_tf import load_or_init_model, train
from tpu_models.utils import load_yaml, set_seed
from tpu_models.config import Config
from tpu_models.tpu_utils import tpu_initialize

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


def get_dataset(input_files, max_seq_len, batch_size, num_cpu_threads: int=4, is_training: bool = True, evaluate_for_fixed_number_of_steps=True):
    AUTO = tf.data.experimental.AUTOTUNE
    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([max_seq_len], tf.int64),
    }
    def parse_tfrecord(example):
        example = tf.io.parse_single_example(example, name_to_features)
                
        input_ids= example["input_ids"]

        # return input_ids[:-1], input_ids[1:]

        # # TODO: consider `attention_mask`
        # return {"input_ids": input_ids[:-1]}, input_ids[1:]
        return {"input_ids": input_ids[:-1], "label": input_ids[1:]}
        # return example
    
    # Read from TFRecords. For optimal performance, we interleave reads from multiple files.
    records = tf.data.TFRecordDataset(input_files, num_parallel_reads=AUTO)
    dataset = records.map(parse_tfrecord, num_parallel_calls=AUTO)
  
    if is_training:
        dataset = dataset.shuffle(2048)

    def parse_mini_batch(batch_data):
        return {"input_ids": batch_data["input_ids"]}, batch_data["label"]

    # Prefetch the next batch while training (autotune prefetch buffer size).
    return dataset.batch(batch_size, drop_remainder=True).map(parse_mini_batch, num_parallel_calls=AUTO).prefetch(AUTO)

    # name_to_features = {
    #     "input_ids": tf.io.FixedLenFeature([max_seq_len], tf.int64),
    # }


    # def _decode_record(record, name_to_features):
    #     """Decodes a record to a TensorFlow example."""
    #     example = tf.io.parse_single_example(record, name_to_features)

    #     # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    #     # So cast all int64 to int32.
    #     for name in list(example.keys()):
    #         t = example[name]
    #         if t.dtype == tf.int64:
    #             t = tf.cast(t, tf.int32)
    #         example[name] = t
    #     return example

    # # For training, we want a lot of parallel reading and shuffling.
    # # For eval, we want no shuffling and parallel reading doesn't matter.
    # if is_training:
    #     d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
    #     d = d.repeat()
    #     d = d.shuffle(buffer_size=len(input_files))

    #     # `cycle_length` is the number of parallel files that get read.
    #     cycle_length = min(num_cpu_threads, len(input_files))

    #     # `sloppy` mode means that the interleaving is not exact. This adds
    #     # even more randomness to the training pipeline.
    #     d = d.apply(
    #         tf.data.experimental.parallel_interleave(
    #             tf.data.TFRecordDataset,
    #             sloppy=is_training,
    #             cycle_length=cycle_length))
    #     d = d.shuffle(buffer_size=100)
    # else:
    #     d = tf.data.TFRecordDataset(input_files)
    #     # If we evaluate for a fixed number of steps we don't want to encounter
    #     # out-of-range exceptions.
    #     if evaluate_for_fixed_number_of_steps:
    #         d = d.repeat()

    # # We must `drop_remainder` on training because the TPU requires fixed
    # # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # # and we *don't* want to drop the remainder, otherwise we wont cover
    # # every sample.
    # d = d.apply(
    #     tf.data.experimental.map_and_batch(
    #         lambda record: _decode_record(record, name_to_features),
    #         batch_size=batch_size,
    #         num_parallel_batches=num_cpu_threads,
    #         drop_remainder=True))
    # return d




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
    train_dataset = get_dataset(train_fns, max_seq_len=128, batch_size=params.train.batch_size, is_training=True)
    valid_dataset = get_dataset(validation_fns, max_seq_len=128, batch_size=params.train.batch_size)

    # When tpu_address is an empty string, we communicate with local TPUs.
    cluster_resolver = tpu_initialize.tpu_initialize("tpu-quickstart")
    print('Running on TPU ', cluster_resolver.as_dict()['worker'])
    tpu_strategy = tf.distribute.TPUStrategy(cluster_resolver)

    # try:
    #     tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    #     print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    # except ValueError:
    #     raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')
    

    # tf.config.experimental_connect_to_cluster(tpu)
    # tf.tpu.experimental.initialize_tpu_system(tpu)
    # tpu_strategy = tf.distribute.TPUStrategy(tpu)

    # Train model
    with tpu_strategy.scope(): # creating the model in the TPUStrategy scope means we will train the model on the TPU
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