import math
import random

import numpy as np
import tensorflow as tf
import transformers
import wandb

from tpu_models import tpu_utils
from tpu_models.config import Config
from tpu_models.models.gpt2_tf import load_or_init_model, train
from tpu_models.utils import load_yaml, set_seed

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


logger = tf.get_logger()
logger.info(tf.__version__)


# Initialize a new W&B run – You can change your project name here.
# For more config options, see https://docs.wandb.com/docs/init.html
wandb.init(project="tpu_gpt2", sync_tensorboard=True)


def create_dataset(
    input_files, n_ctx, max_seq_len, batch_size, is_training: bool = True,
):
    AUTO = tf.data.experimental.AUTOTUNE
    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([n_ctx + 1], tf.int64),
    }

    def parse_tfrecord(example):
        example = tf.io.parse_single_example(example, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example[name] = t

        input_ids = example["input_ids"][: max_seq_len + 1]

        return {"input_ids": input_ids[:-1], "label": input_ids[1:]}

    random.shuffle(input_files)

    # Read from TFRecords. For optimal performance, we interleave reads from multiple files.
    records = tf.data.TFRecordDataset(input_files, num_parallel_reads=AUTO)
    dataset = records.map(parse_tfrecord, num_parallel_calls=AUTO)

    if is_training:
        dataset = dataset.shuffle(2048)

    def parse_mini_batch(batch_data):
        return {"input_ids": batch_data["input_ids"]}, batch_data["label"]

    # Prefetch the next batch while training (autotune prefetch buffer size).
    return (
        dataset.batch(batch_size, drop_remainder=True)
        .map(parse_mini_batch, num_parallel_calls=AUTO)
        .prefetch(AUTO)
    )


def main(config):
    params = Config(**load_yaml(config))
    print(params)

    # set_seed(params.train.seed)

    # # gcs_pattern = 'gs://flowers-public/tfrecords-jpeg-331x331/*.tfrec'
    train_fns = tf.io.gfile.glob(params.input.train_file)[:-10]
    validation_fns = tf.io.gfile.glob(params.input.valid_file)[-10:]

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
            "tpu-quickstart"
        )  # TPU detection
        print("[INFO] Running on TPU ", tpu.cluster_spec().as_dict()["worker"])

        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        tpu_strategy = tf.distribute.TPUStrategy(tpu)
    except ValueError:
        raise BaseException("ERROR: Not connected to a TPU runtime;")

    print(
        "[INFO] Running with TPUStrategy on TPU {} with {} cores ".format(
            tpu.cluster_spec().as_dict()["worker"], tpu_strategy.num_replicas_in_sync
        )
    )

    batch_size = params.train.batch_size * tpu_strategy.num_replicas_in_sync
    vocab_size = params.model_params.vocab_size

    train_dataset = create_dataset(
        train_fns,
        n_ctx=params.model_params.n_ctx,
        max_seq_len=params.train.max_seq_len,
        batch_size=batch_size,
        is_training=True,
    )
    valid_dataset = create_dataset(
        validation_fns,
        n_ctx=params.model_params.n_ctx,
        max_seq_len=params.train.max_seq_len,
        batch_size=batch_size,
    )

    # creating the model in the TPUStrategy scope means we will train the model on the TPU
    with tpu_strategy.scope():

        model, global_step_init = load_or_init_model(
            pretrained_model_dir=params.input.pretrained_model_dir,
            vocab_size=vocab_size,
            params=params.model_params,
        )

        val_best_model = train(
            params,
            model,
            train_dataset,
            valid_dataset,
            vocab_size,
            pad_token_id=0,
            global_step_init=global_step_init,
        )
        val_best_model.summary()

    # # Evaluate best model with validation set
    # val_best_model.evaluate(valid_dataset)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
