from __future__ import absolute_import, division, print_function, unicode_literals
from tqdm import trange
import torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in
                  [GPT2Config]), ())

MODEL_CLASSES = {'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
                 'gpt2-medium': (GPT2LMHeadModel, GPT2Tokenizer),
                 'gpt2-large': (GPT2LMHeadModel, GPT2Tokenizer),
                 'gpt2-xl': (GPT2LMHeadModel, GPT2Tokenizer),
                 'distilgpt2': (GPT2LMHeadModel, GPT2Tokenizer)}


def set_seed(device, seed):
    """
    :param device: Sets the seed for cuda if the GPU is used.
    :param seed: Random seed for initialization.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = device.lower()
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317.
    :param logits: logits distribution shape (batch size x vocabulary size).
    :param top_k: keep only top k tokens with highest probability (top-k filtering).
    :param top_p: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    :param filter_value:
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.9, repetition_penalty=1.0,
                    device='cpu'):
    """
    From: https://github.com/huggingface/transformers/blob/master/examples/run_generation.py.
    :param model: Which gpt2 model to use.
    :param length: Number of words to generate.
    :param context: Encoded text.
    :param num_samples: Number of complete generations that should be performed.
    :param temperature: Creativity of the generated text. 0 implies greedy sampling
    :param top_k: Keep only top k tokens with highest probability (top-k filtering).
    :param top_p: Keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    :param repetition_penalty:
    :param device: Chose if the generation should be performed by the CPU or CUDA.
    :return: Returns the generated text.
    """
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in trange(length):
            inputs = {'input_ids': generated}
            outputs = model(**inputs)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)
            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0:  # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
    return generated


def length_failsafe(length, model):
    """
    :param length: Number of words to generate.
    :param model: Which gpt2 model to use.
    :return: Returns the inputted length if there was no problem with the length,
    otherwise the length is set to the model maximum or 10000 to avoid an infinite loop.
    """
    if length < 0 and model.config.max_position_embeddings > 0:
        length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < length:
        length = model.config.max_position_embeddings  # No generation bigger than model size
    elif length < 0:
        length = int(10000)  # avoid infinite loop
    return length


def model_tokenizer_initializer(model_type, seed=42, device='cpu'):
    """
    :param model_type: Which gpt2 model to use.
    :param device: Specify if the CPU or CUDA should be used.
    :return: Returns the model and tokenizer in a tuple: (model, tokenizer).
    """
    set_seed(device, seed)
    model_class, tokenizer_class = MODEL_CLASSES[model_type]
    tokenizer = tokenizer_class.from_pretrained(model_type)
    model = model_class.from_pretrained(model_type)
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_text(model, tokenizer, length, prompt, num_samples=1, temperature=1, top_k=0, top_p=0.9,
                  repetition_penalty=1.0, device='cpu', stop_token=None):
    context_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    out = sample_sequence(model=model,
                          length=length,
                          context=context_tokens,
                          num_samples=num_samples,
                          temperature=temperature,
                          top_k=top_k,
                          top_p=top_p,
                          repetition_penalty=repetition_penalty,
                          device=device)
    out = out[:, len(context_tokens):].tolist()
    for o in out:
        text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
        text = text[: text.find(stop_token) if stop_token else None]
    return text


def main():
    print(generate_text('distilgpt2', 200, 'Hello world, how are you doing today?'))


if __name__ == '__main__':
    main()
