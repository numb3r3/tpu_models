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
    def __init__(self, model, save_dir, global_step_init=0, 
            model_save_intervals=1000,
            checkpoint_manager=None,
            checkpoint_intervals=1000):
        super().__init__()

        self._model = model
        self._save_dir = save_dir
        self.global_steps = global_step_init
        self.model_save_intervals = intervals
        self.checkpoint_manager = checkpoint_manager
        self.checkpoint_intervals=checkpoint_intervals

    def on_batch_end(self, batch, logs=None):
        if (self.global_steps + 1) % self.model_save_intervals == 0:
            checkpoint_dir = os.path.join(
                self._save_dir, f"checkpoint-step-{self.global_steps}"
            )

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            self._model.save_pretrained(checkpoint_dir)
            print(f"[DEBUG] saving model at iteration {self.global_steps}")
            
        if (self.global_steps + 1) % self.checkpoint_intervals == 0:
            ckpt_save_path = self.checkpoint_manager.save()
            print ('Saving checkpoint for steps {} at {}'.format(self.global_steps, ckpt_save_path))

        self.global_steps += 1

    def on_epoch_end(self, epoch, logs=None):
        save_dir = os.path.join(self.save_dir, f"checkpoint-epoch-{epoch}")
        print(f"[DEBUG] save transformers model in {save_dir} at epoch {epoch}")
        self._model.save_pretrained(save_dir)
