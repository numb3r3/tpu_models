import math
import torch
from transformers import GPT2LMHeadModel, GPT2Config, BertTokenizer
from tqdm import tqdm

def load_model_tokenizer(model_name_or_path, tokenizer_name_or_path, device: str="cuda"):
    config = GPT2Config.from_json_file(f'{model_name_or_path}/config.json')
    model = GPT2LMHeadModel.from_pretrained(
        f'{model_name_or_path}/tf_model.h5', config=config, from_tf=True)
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained(
        tokenizer_name_or_path, do_lower_case=True
    )

    return model, tokenizer

def evaluate(model, tokenizer, data_path, batch_size: int = 128, max_seq_len: int = 64):

    sum_ppl = 0.0
    count = 0
    with open(data_path, "r", encoding="utf-8") as fin:
        batched_input_ids, batched_attention_masks, batched_labels = [], [], []
        with torch.no_grad():
            for line in tqdm(fin):
                tokenized_dict = tokenizer.encode_plus(
                    line,
                    text_pair=None,
                    add_special_tokens=False,
                    max_length=max_seq_len,
                    truncation=True,
                    pad_to_max_length=False,
                )
                input_ids, attention_mask = (
                    tokenized_dict["input_ids"],
                    tokenized_dict["attention_mask"],
                )
                lm_labels = input_ids.copy()

                # padding sequence
                input_ids += [tokenizer.pad_token_id] * (
                    max_seq_len - len(input_ids)
                )

                attention_mask += [0] * (max_seq_len - len(attention_mask))
                lm_labels += [-100] * (max_seq_len - len(lm_labels))

                batched_input_ids.append(input_ids)
                batched_attention_masks.append(attention_mask)
                batched_labels.append(lm_labels)

                if len(batched_input_ids) >= batch_size:
                    input_ids = torch.tensor(
                        batched_input_ids, dtype=torch.long, device=model.device)
                    attention_mask = torch.tensor(
                        batched_attention_masks, dtype=torch.long, device=model.device)
                    labels = torch.tensor(
                        batched_labels, dtype=torch.long, device=model.device)

                    loss, logits, *_ = model(
                        input_ids,
                        labels=input_ids,
                        attention_mask=attention_mask,
                    )

                    ppl = math.exp(loss)
                    sum_ppl += ppl
                    count += 1

                    batched_input_ids.clear()
                    batched_attention_masks.clear()
                    batched_labels.clear()

            if len(batched_input_ids) > 0:
                input_ids = torch.tensor(
                    batched_input_ids, dtype=torch.long, device=model.device)
                attention_mask = torch.tensor(
                    batched_attention_masks, dtype=torch.long, device=model.device)
                labels = torch.tensor(
                    batched_labels, dtype=torch.long, device=model.device)

                loss, logits, *_ = model(
                    input_ids,
                    labels=input_ids,
                    attention_mask=attention_mask,
                )

                ppl = math.exp(loss)
                sum_ppl += ppl
                count += 1

    print("PPL: %.3f" % (sum_ppl / count))

def main(model_name_or_path, tokenizer_name_or_path, data_path, batch_size: int = 128, max_seq_len: int = 64, device: str = "cuda"):
    model, tokenizer = load_model_tokenizer(model_name_or_path, tokenizer_name_or_path, device=device)
    evaluate(model, tokenizer, data_path, batch_size=batch_size, max_seq_len=max_seq_len, device=device)

if __name__ == "__main__":
    import fire

    fire.Fire(main)
