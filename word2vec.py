import torch
import torch.nn as nn
import torch.optim as optim


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, word_ids, context_ids, negative_context_ids):
        word_embeds = self.word_embeddings(word_ids)  # Shape: (batch_size, embedding_dim)
        positive_context_embeds = self.context_embeddings(context_ids)  # Shape: (batch_size, context_size, embedding_dim)
        negative_context_embeds = self.context_embeddings(negative_context_ids)  # Shape: (batch_size, K*context_size, embedding_dim)

        # Positive scores: (batch_size, context_size)
        positive_scores = torch.bmm(positive_context_embeds, word_embeds.unsqueeze(-1)).squeeze(-1)

        # Negative scores: (batch_size, K*context_size)
        negative_scores = torch.bmm(negative_context_embeds, word_embeds.unsqueeze(-1)).squeeze(-1)

        return positive_scores, negative_scores

def save_model(model, embedding_dim, radius, ratio, batch, epoch):
    filename = f"model_dim-{embedding_dim}_radius-{radius}_ratio-{ratio}_batch-{batch}_epoch-{epoch}.ckpt"
    torch.save(model.state_dict(), filename)  # Save the model's state_dict
    print(f"Model saved to: {filename}")

def validate_word2vec(model, valid_dataloader, device, K):
    model.eval()
    total_positive_correct = 0
    total_negative_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in valid_dataloader:
            word_ids = batch['word_id'].to(device)
            positive_context_ids = batch['positive_context_ids'].to(device)
            negative_context_ids = batch['negative_context_ids'].to(device)

            positive_scores, negative_scores = model(word_ids, positive_context_ids, negative_context_ids)

            positive_pred = (positive_scores > 0).float()
            negative_pred = (negative_scores < 0).float()

            total_positive_correct += positive_pred.sum().item()
            total_negative_correct += negative_pred.sum().item()
            total_samples += word_ids.size(0) * positive_context_ids.size(1)

    accuracy = (total_positive_correct + total_negative_correct) / (2 * total_samples)
    print(f"Validation Accuracy: {accuracy:.4f}")
    return accuracy


def train_word2vec(model, train_dataloader, valid_dataloader, epochs, lr, K, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            word_ids = batch['word_id'].to(device)
            positive_context_ids = batch['positive_context_ids'].to(device)
            negative_context_ids = batch['negative_context_ids'].to(device)

            # Forward pass
            positive_scores, negative_scores = model(word_ids, positive_context_ids, negative_context_ids)

            # Create labels: 1 for positive samples, 0 for negative samples
            positive_labels = torch.ones_like(positive_scores)
            negative_labels = torch.zeros_like(negative_scores)

            # Calculate loss
            loss_positive = loss_fn(positive_scores, positive_labels)
            loss_negative = loss_fn(negative_scores, negative_labels)
            loss = loss_positive + loss_negative

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_dataloader):.4f}")

        # Validate after each epoch
        validate_word2vec(model, valid_dataloader, device, K)
