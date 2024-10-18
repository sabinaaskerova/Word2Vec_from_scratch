import torch
from torch.utils.data import Dataset

class Word2VecDataset(Dataset):
    def __init__(self, words, contexts):
        self.words = words
        self.contexts = contexts

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        return self.words[idx], self.contexts[idx]
    

def preprocessing_fn(x, tokenizer):
    x["review_ids"] = tokenizer(
        x["review"],
        add_special_tokens=False,
        truncation=True,
        max_length=256,
        padding=False,
        return_attention_mask=False,
    )["input_ids"]
    x["label"] = 0 if x["sentiment"] == "negative" else 1
    return x



def extract_words_contexts(token_ids, radius):
    words = []
    contexts = []
    for i in range(len(token_ids)):
        target = token_ids[i]
        left_context = token_ids[max(0, i-radius):i]
        right_context = token_ids[i+1:min(len(token_ids), i+radius+1)]

        # Calculate padding
        total_padding = 2 * radius - len(left_context) - len(right_context)
        left_pad = total_padding // 2
        right_pad = total_padding - left_pad

        # Create symmetrically padded context
        context = ([0] * left_pad) + left_context + right_context + ([0] * right_pad)

        words.append(target)
        contexts.append(context)
    return words, contexts



def flatten_dataset_to_list(dataset, radius):
    all_words = []
    all_contexts = []
    for row in dataset:
        words, contexts = extract_words_contexts(row['review_ids'], radius)
        all_words.extend(words)
        all_contexts.extend(contexts)
    return all_words, all_contexts


def collate_fn(batch, vocab_size, K=5):
    words, contexts = zip(*batch)

    # Convert to tensors
    words = torch.tensor(words, dtype=torch.long)
    contexts = torch.tensor(contexts, dtype=torch.long)

    # Generate negative samples
    batch_size, context_size = contexts.shape
    negative_contexts = torch.randint(0, vocab_size, (batch_size, K * context_size), dtype=torch.long)

    return {
        'word_id': words,
        'positive_context_ids': contexts,
        'negative_context_ids': negative_contexts
    }