import os

import tensorflow.keras as keras


class WarmupScheduler(keras.callbacks.Callback):
    def __init__(self, warmup_steps, learning_rate):
        super().__init__()

        self._warmup_steps = warmup_steps
        self._learning_rate = learning_rate

        # The argument passed to on_train_batch_begin
        # is resetted every epoch.
        # self._total_steps is used to keep total step
        self._total_steps = 0

    def on_train_batch_begin(self, step, logs=None):
        self._total_steps += 1
        step = self._total_steps

        if step > self._warmup_steps:
            return

        # Get the current learning rate from model's optimizer.
        lr = float(keras.backend.get_value(self.model.optimizer.lr))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self._learning_rate * (step / self._warmup_steps)
        # Set the value back to the optimizer before this epoch starts
        keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        # print('\nStep {}: lr is schedulerd {:.4e} -> {:.4e}]'.format(step, lr, float(tf.keras.backend.get_value(self.model.optimizer.lr))))


class TransformersCheckpoint(keras.callbacks.Callback):
    def __init__(self, model, save_dir, global_step_init=0, intervals=100):
        super().__init__()

        self._model = model
        self._save_dir = save_dir
        self.global_steps = global_step_init
        self.intervals = intervals

    def on_batch_end(self, batch, logs=None):
        if (self.global_steps + 1) % self.intervals == 0:
            checkpoint_dir = os.path.join(
                self._save_dir, f"checkpoint-step-{self.global_steps}"
            )

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            self._model.save_pretrained(checkpoint_dir)
            print(f"saving model at iteration {self.n_batch}")

        self.global_steps += 1

    def on_epoch_end(self, epoch, logs=None):
        save_dir = os.path.join(self.save_dir, f"checkpoint-epoch-{epoch}")
        print(f"Save transformers model in {save_dir}")
        self._model.save_pretrained(save_dir)
