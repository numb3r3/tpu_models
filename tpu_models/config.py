from typing import List, Union

from pydantic import BaseModel


class ConfigInput(BaseModel):
    train_file: str
    valid_file: str
    tokenizer_file: str
    pretrained_model_dir: Union[None, str]


class ConfigOutput(BaseModel):
    model_dir: str
    tokenizer_file: str
    tensorboard_dir: str
    checkpoint_dir: str

    model_save_intervals: int
    checkpoint_intervals: int


class ConfigTrain(BaseModel):
    block_size: int
    seed: int
    num_epochs: int
    batch_size: int
    max_seq_len: int
    steps_per_epoch: int
    learning_rate: float
    max_grad_norm: float
    weight_decay_rate: float
    warmup_rate: float
    patience: float

    optimizer: str


class ConfigPred(BaseModel):
    do_sample: bool
    seed: int
    max_length: int
    top_k: int
    top_p: float
    bad_words: List[str]


class ConfigModelParams(BaseModel):
    n_embd: int
    n_layer: int
    n_head: int
    n_ctx: int
    vocab_size: int


class Config(BaseModel):
    input: ConfigInput
    output: ConfigOutput
    model_params: ConfigModelParams
    train: ConfigTrain
    pred: ConfigPred
