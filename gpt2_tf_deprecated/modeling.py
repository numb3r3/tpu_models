import copy
import json

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class GroverModel(object):
    def __init__(
        self,
        config: GroverConfig,
        is_training,
        input_ids,
        cache=None,
        do_cache=False,
        pad_token_id=0,
        chop_off_last_token=True,
        scope=None,
        reuse=False,
    ):
        """
        :param config:
        :param is_training:
        :param input_ids: Tensor thats of size [batch_size, seq_length]
        :param cache: Optionally, a tensor to use that will contain cached information of the size
            [batch_size, num_layers, 2, num_heads, cache_length, features]
        :param do_cache: Whether to cache again.
        :param pad_token_id: Which token will be used for padding (probably 0.)
        :param chop_off_last_token: True if we will end up using this for TRAINING only. False if we want to generate.
                                    it means the last token in input_ids will not be processed by the model as input
        :param scope: scope to run this on
        """
        self.config = copy.deepcopy(config)
        self.is_training = is_training
        self.pad_token_id = pad_token_id

        if not is_training:
            self.config.hidden_dropout_prob = 0.0
            self.config.attention_probs_dropout_prob = 0.0

        if chop_off_last_token:
            self.target_ids = input_ids[:, 1:]
            self.input_ids = input_ids[:, :-1]
        else:
            self.input_ids = input_ids
            self.target_ids = tf.concat(
                (
                    input_ids[:, 1:],
                    tf.constant(
                        self.pad_token_id,
                        dtype=self.input_ids.dtype,
                        shape=[get_shape_list(self.input_ids, 2)[0], 1],
                    ),
                ),
                1,
            )

        self.batch_size, self.seq_length = get_shape_list(self.input_ids, 2)

        if cache is None:
            caches = [None] * config.num_hidden_layers
            self.cache_length = 0
        else:
            (
                batch_size_,
                num_layers_,
                two_,
                num_heads_,
                self.cache_length,
                features_,
            ) = get_shape_list(cache, expected_rank=6)
            assert batch_size_ == self.batch_size
            assert num_layers_ == config.num_hidden_layers
            assert two_ == 2
            assert num_heads_ == config.num_attention_heads
            assert features_ == (config.hidden_size // config.num_attention_heads)
            caches = tf.unstack(cache, axis=1)

        with tf.variable_scope(scope, default_name="newslm", reuse=reuse):
            with tf.variable_scope("embeddings"):
                embeddings, self.embedding_table = embed(
                    self.input_ids,
                    config.vocab_size,
                    config.hidden_size,
                    position_offset=self.cache_length,
                    initializer_range=config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,
                    use_one_hot_embeddings=True,
                )

            mask = get_attention_mask(
                self.seq_length,
                self.seq_length + self.cache_length,
                dtype=embeddings.dtype,
            )

            # We keep the representation as a 2D tensor to avoid re-shaping it back and
            # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
            # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
            # help the optimizer.
            hidden_state = tf.reshape(
                embeddings, [self.batch_size * self.seq_length, self.config.hidden_size]
            )
            new_kvs = []
            for layer_idx, layer_cache in enumerate(caches):
                with tf.variable_scope("layer{:02d}".format(layer_idx)):
                    # [batch_size * seq_length, hidden_size]
                    attention_output, new_kv = attention_layer(
                        hidden_state,
                        mask,
                        batch_size=self.batch_size,
                        seq_length=self.seq_length,
                        size_per_head=config.hidden_size // config.num_attention_heads,
                        num_attention_heads=config.num_attention_heads,
                        initializer_range=config.initializer_range,
                        hidden_dropout_prob=self.config.hidden_dropout_prob,
                        attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
                        do_cache=do_cache,
                        cache=layer_cache,
                    )
                    new_kvs.append(new_kv)

                    # [batch_size * seq_length, hidden_size]
                    hidden_state = residual_mlp_layer(
                        hidden_state + attention_output,
                        intermediate_size=config.intermediate_size,
                        hidden_dropout_prob=self.config.hidden_dropout_prob,
                    )
            self.hidden_state = hidden_state

        self.new_kvs = tf.stack(new_kvs, axis=1) if do_cache else None

        # Note that the hidden state is still flat (batch_size*hidden_size)
        self.logits_flat = tf.matmul(
            self.hidden_state, self.embedding_table, transpose_b=True
        )

        # THE OUTPUT BIAS DOES NOT SPARK JOY
        # output_bias = tf.get_variable('output_bias', shape=[config.vocab_size], initializer=tf.zeros_initializer())
        # self.logits_flat = tf.nn.bias_add(self.logits_flat, output_bias)

    @property
    def log_probs(self):
        logprobs_flat = tf.nn.log_softmax(self.logits_flat, axis=-1)
        return tf.reshape(logprobs_flat, [self.batch_size, self.seq_length, -1])

    def lm_loss(self):
        """
        :return: stuff
        """
        target_ids_flat = tf.reshape(self.target_ids, [-1])

        # 1 if it's valid and 0 otherwise.
        label_weights = tf.cast(
            tf.not_equal(target_ids_flat, self.pad_token_id),
            dtype=self.logits_flat.dtype,
        )

        # [batch_size * seq_length, vocab_size]
        one_hot_labels = tf.one_hot(
            target_ids_flat, depth=self.config.vocab_size, dtype=self.logits_flat.dtype
        )

        # [batch_size * seq_length, vocab_size]
        logprobs_flat = tf.nn.log_softmax(self.logits_flat, axis=-1)

        per_example_loss = -tf.reduce_sum(logprobs_flat * one_hot_labels, axis=[-1])

        # per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_flat, labels=target_ids_flat)

        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator
        return loss

    def pooled_output(self, clf_token):
        """Extract pooled output given a token that says where we should look.

        :param clf_token:
        :return:
        """
        pool_idx = tf.cast(
            tf.argmax(tf.cast(tf.equal(self.input_ids, clf_token), tf.float32), 1),
            tf.int32,
        )
        return tf.gather(
            self.hidden_state,
            tf.range(self.batch_size, dtype=tf.int32) * self.seq_length + pool_idx,
        )


def model_fn_builder(
    config: GroverConfig,
    init_checkpoint,
    learning_rate,
    num_train_steps,
    num_warmup_steps,
    use_tpu,
):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        model = GroverModel(
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            pad_token_id=config.pad_token_id,
            chop_off_last_token=True,
        )

        total_loss = model.lm_loss()

        if is_training:
            train_op, train_metrics = optimization_adafactor.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu
            )
            tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        else:
            train_op = None
            train_metrics = {}
            tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (
                assignment_map,
                initialized_variable_names,
            ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info(
                "  name = %s, shape = %s%s", var.name, var.shape, init_string
            )

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            if use_tpu:
                output_spec = tf.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    host_call=construct_scalar_host_call(
                        metric_dict=train_metrics,
                        model_dir=params["model_dir"],
                        prefix="training/",
                    ),
                    scaffold_fn=scaffold_fn,
                )
            else:
                output_spec = tf.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    training_hooks=[
                        tf.train.LoggingTensorHook(
                            {"loss": tf.metrics.mean(total_loss)[1]}, every_n_iter=100
                        )
                    ],
                    scaffold_fn=scaffold_fn,
                )

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(total_loss):
                loss = tf.metrics.mean(values=total_loss)
                return {
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn, [total_loss])
            output_spec = tf.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn,
            )
        else:
            gt_logprobs = tf.squeeze(
                tf.batch_gather(model.log_probs, model.target_ids[:, :, None]), axis=2
            )

            # Need top-p required under topp sampling!
            better_than_gt = model.log_probs > gt_logprobs[:, :, None]
            top_p_required = tf.reduce_sum(
                tf.cast(better_than_gt, tf.float32) * tf.exp(model.log_probs), axis=2
            )

            # No top-p sampling for now, since this seems to be too slow on TPUs
            if use_tpu:
                predictions = tf.reshape(
                    tf.random.categorical(logits=model.logits_flat, num_samples=1),
                    get_shape_list(model.target_ids),
                )
            else:
                # Argmax
                # predictions = tf.math.argmax(model.log_probs, axis=-1, output_type=tf.int32)
                predictions = tf.reshape(
                    _top_p_sample(model.logits_flat, num_samples=1, p=0.99)["sample"],
                    get_shape_list(model.target_ids),
                )
            pred_logprobs = tf.squeeze(
                tf.batch_gather(model.log_probs, predictions[:, :, None]), axis=2
            )

            output_spec = tf.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={
                    "gt_logprobs": gt_logprobs,
                    "top_p_required": top_p_required,
                    "predictions": predictions,
                    "pred_logprobs": pred_logprobs,
                    "labels": input_ids,
                },
                scaffold_fn=scaffold_fn,
            )
        return output_spec

    return model_fn
