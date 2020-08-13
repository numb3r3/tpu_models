import os

import tensorflow as tf
import transformers
from transformers import optimization_tf
from wandb.keras import WandbCallback

from ..callbacks import TransformersCheckpoint, WarmupScheduler
from ..optimizers_tf import AdafactorOptimizer, WarmUpLinearDecayScheduler

logger = tf.get_logger()
logger.info(tf.__version__)


def init_model(vocab_size, params):
    config = transformers.GPT2Config(
        vocab_size=vocab_size,
        n_ctx=params.n_ctx,
        n_positions=params.n_ctx,
        n_embd=params.n_embd,
        n_layer=params.n_layer,
        n_head=params.n_head,
    )
    model = transformers.TFGPT2LMHeadModel(config=config)
    return model


def load_or_init_model(pretrained_model_dir, vocab_size, params):
    global_step = 0
    # Train model
    if pretrained_model_dir:
        print(f"[INFO] load model from {pretrained_model_dir}")
        global_step = int(pretrained_model_dir.split("-")[-1].split("/")[0])
        print(f"[INFO] starting from global step {global_step}")
        model = transformers.TFGPT2LMHeadModel.from_pretrained(pretrained_model_dir)
    else:
        print(f"[INFO] initialize model with parameters: {params}")
        model = init_model(vocab_size, params)

    return model, global_step


def cross_entropy_loss_with_padding(num_labels, pad_token_id):
    loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )

    def loss(y_true, y_pred):
        input_mask = tf.not_equal(y_true, pad_token_id)
        active_loss = tf.reshape(input_mask, (-1,))
        logits = tf.reshape(y_pred, (-1, num_labels))
        active_logits = tf.boolean_mask(logits, active_loss)

        train_labels = tf.reshape(y_true, (-1,))
        active_labels = tf.boolean_mask(train_labels, active_loss)
        cross_entropy = loss_fct(active_labels, active_logits)

        return cross_entropy

    return loss


# To know more about how to train TFGPT2LMHead, read
#   https://github.com/huggingface/transformers/issues/2169
#   https://github.com/tensorflow/tensorflow/issues/41074
def train(
    params,
    model,
    train_dataset,
    valid_dataset,
    vocab_size,
    pad_token_id=0,
    global_step_init=0,
    run_eagerly=False,
):
    # Prepare model directory and path
    os.makedirs(params.output.model_dir, exist_ok=True)

    # Set from_logits=True because TFGPT2LMHeadModel returns the logits (before Softmax)
    # loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = cross_entropy_loss_with_padding(
        num_labels=vocab_size, pad_token_id=pad_token_id,
    )

    learning_rate = params.train.learning_rate
    steps_per_epoch = params.train.steps_per_epoch
    num_train_steps = params.train.num_epochs * steps_per_epoch
    num_warmup_steps = params.train.warmup_rate * steps_per_epoch

    # # Setup the optimizer and the learning rate scheduler.
    # optimizer, lr_scheduler = optimization_tf.create_optimizer(
    #     learning_rate,
    #     num_train_steps,
    #     num_warmup_steps,
    #     # adam_beta1=0.9,
    #     # adam_beta2=0.999,
    #     # adam_epsilon=1e-8,
    #     weight_decay_rate=params.train.weight_decay_rate,
    # )

    optimizer = AdafactorOptimizer(learning_rate=learning_rate)

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=[loss, *[None] * model.config.n_layer],
        # metrics=[
        #     keras.metrics.SparseCategoricalCrossentropy(from_logits=True),
        #     keras.metrics.SparseCategoricalAccuracy(),
        # ],
        run_eagerly=run_eagerly,
    )

    callbacks_list = [
        WarmUpLinearDecayScheduler(
            learning_rate_base=learning_rate,
            total_steps=num_train_steps,
            warmup_steps=num_warmup_steps,
            global_step_init=global_step_init,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=params.train.patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=params.output.checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        ),
        TransformersCheckpoint(
            model=model,
            save_dir=params.output.model_dir,
            global_step_init=global_step_init,
            intervals=params.train.checkpoint_intervals,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=params.output.tensorboard_dir,
            update_freq="batch",
            # To automatically refresh Tensorboard , set profile_batch=0
            # See more details here https://github.com/tensorflow/tensorboard/issues/2412
            profile_batch=0,
        ),
        # WarmupScheduler(
        #     total_steps * params.train.warmup_rate, params.train.learning_rate
        # ),
        # WandbCallback(),
    ]

    # Train model
    model.fit(
        train_dataset,
        epochs=params.train.num_epochs,
        callbacks=callbacks_list,
        validation_data=valid_dataset,
    )

    # Restore the best model and save it as pretrained model format
    # If restore_best_weights=False, this process is required
    model.load_weights(params.output.checkpoint_path)
    model.save_pretrained(params.output.model_dir)

    # Save model with best performance
    return model
