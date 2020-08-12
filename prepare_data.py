import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import tensorflow as tf
import transformers
from tqdm import tqdm
from transformers.tokenization_bert import BertTokenizer


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def filter_text(text):
    return text


def build_tfrecord(
    raw_data_path, tokenizer, min_length: int = 5, max_seq_len: int = 256,
):
    def ids_example(ids):
        feature = {
            "input_ids": _int64_feature(ids),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    # tf_writer = tf.io.TFRecordWriter(save_tfrecord_path)

    # max_len = 0

    with open(raw_data_path, "r", encoding="utf8") as f:
        for line in tqdm(f):
            line = line.strip()
            line = filter_text(line)
            if not line or len(line) <= min_length:
                continue

            tokenized_dict = tokenizer.encode_plus(
                line,
                text_pair=None,
                add_special_tokens=False,
                max_length=max_seq_len,
                truncation=True,
                pad_to_max_length=True,
            )
            input_ids, input_masks = (
                tokenized_dict["input_ids"],
                tokenized_dict["attention_mask"],
            )

            # if len(input_ids) > max_len:
            #     max_len = len(input_ids)

            example = ids_example(input_ids)
            # tf_writer.write(example.SerializeToString())
            yield example

    # tf_writer.close()
    # return max_len


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tokenizer_name_or_path", type=str, required=True, help="词典地址"
    )
    parser.add_argument("--data_path", type=str, help="原始语料地址")
    parser.add_argument("--target_path", type=str, help="处理后的语料存放地址")
    parser.add_argument(
        "--min_length", default=10, type=int, required=False, help="最短收录句子长度"
    )
    parser.add_argument(
        "--n_ctx", default=256, type=int, required=False, help="每个训练样本的长度"
    )

    args = parser.parse_args()
    print("args:\n" + args.__repr__())

    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer_name_or_path, do_lower_case=True
    )

    tf_writer = tf.io.TFRecordWriter(args.target_path)

    # target_path = args.target_path

    # max_len = 0
    for root, dirname, files in os.walk(args.data_path):
        for fn in tqdm(files):
            base_name = os.path.splitext(fn)[0]
            raw_data_path = os.path.join(root, fn)
            print(f"processing: {base_name}")

            for example in build_tfrecord(
                raw_data_path,
                tokenizer,
                min_length=args.min_length,
                max_seq_len=args.n_ctx,
            ):

                tf_writer.write(example.SerializeToString())

    #         save_tfrecord_path = f"{target_path}/{base_name}.tfrec"
    #         _max_len = build_tfrecord(
    #             raw_data_path,
    #             save_tfrecord_path,
    #             tokenizer,
    #             min_length=args.min_length,
    #             max_seq_len=args.n_ctx,
    #         )
    #         if _max_len > max_len:
    #             max_len = _max_len
    # print("max length: %d" % max_len)
    tf_writer.close()


if __name__ == "__main__":
    main()
