from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch
from itertools import product

from word2vec import Word2Vec, train_word2vec, validate_word2vec
from data_preprocessing import preprocessing_fn, flatten_dataset_to_list, Word2VecDataset, collate_fn

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    dataset = load_dataset("scikit-learn/imdb", split="train")
    n_samples = 5000  # the number of training example

    dataset = dataset.shuffle(seed=35)  # We first shuffle the data !
    dataset = dataset.select(range(n_samples))

    # Tokenize the dataset
    dataset_tokenized = dataset.map(lambda row: preprocessing_fn(row, tokenizer), batched=False)

    dataset_new = dataset_tokenized.remove_columns(["review", "sentiment"]) # Remove useless columns
    dataset_new = dataset_new.train_test_split(test_size=0.2, seed=35) # Split the train and validation

    document_train_set = dataset_new["train"]
    document_valid_set = dataset_new["test"]

    # Set up DataLoaders
    vocab_size = tokenizer.vocab_size
    # Set up hyperparameters
    batch_size = 32
    epochs = 10

    # Hyperparameter grids
    embedding_dims = [100, 200]
    learning_rates = [0.001, 0.01]
    Ks = [5, 10]
    Rs = [3, 5]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = vocab_size = tokenizer.vocab_size

    best_valid_acc = float('inf')
    best_hyperparams = None

    # Grid search over hyperparameter combinations
    for R, K, embedding_dim, learning_rate in product(Rs, Ks, embedding_dims, learning_rates):

        print(f"Training with R={R}, K={K}, embedding_dim={embedding_dim}, learning_rate={learning_rate}, batch_size={batch_size}")


        train_words, train_contexts = flatten_dataset_to_list(document_train_set, R)
        valid_words, valid_contexts = flatten_dataset_to_list(document_valid_set, R)
        # Create Word2Vec datasets
        train_dataset = Word2VecDataset(train_words, train_contexts)
        valid_dataset = Word2VecDataset(valid_words, valid_contexts)

        # Set up DataLoaders
        vocab_size = tokenizer.vocab_size
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                    collate_fn=lambda b: collate_fn(b, vocab_size, K))
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                    collate_fn=lambda b: collate_fn(b, vocab_size, K))

        model = Word2Vec(vocab_size, embedding_dim)

        # Train the model
        train_word2vec(model, train_dataloader, valid_dataloader, epochs, learning_rate, K, device)
        valid_acc = validate_word2vec(model, valid_dataloader, device, K)
        # Update best hyperparameters if current model is better
        if valid_acc > best_valid_acc:
            best_valid_loss = valid_acc
            best_hyperparams = {
                'embedding_dim': embedding_dim,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'K': K,
                'R': R,
                'valid_loss': valid_acc
            }


        # After grid search, print the best hyperparameters
        print("\nBest hyperparameters found:")
        print(best_hyperparams)