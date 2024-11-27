import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import BertTokenizer
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import BertModel
import torch.optim as optim
import torch.nn.functional as F
from transformers import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os


df = pd.read_csv('data_task_a.csv')

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_tweets(texts, max_length=128):
    return tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

# Tokenize the clean_tweets column
tokens = tokenize_tweets(df['clean_tweets'].fillna(''))

# Prepare the labels
labels_hate = torch.tensor(df['Hate'].values)
labels_fake = torch.tensor(df['Fake'].values)

# Split into train and validation sets
train_tokens, val_tokens, train_labels_hate, val_labels_hate, train_labels_fake, val_labels_fake = train_test_split(
    tokens['input_ids'], labels_hate, labels_fake, test_size=0.2, random_state=42
)

train_attention_masks, val_attention_masks = train_test_split(tokens['attention_mask'], test_size=0.2, random_state=42)



class MultiTaskDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels_hate, labels_fake):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels_hate = labels_hate
        self.labels_fake = labels_fake

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'label_hate': self.labels_hate[idx],
            'label_fake': self.labels_fake[idx]
        }

# Initialize Datasets
train_dataset = MultiTaskDataset(train_tokens, train_attention_masks, train_labels_hate, train_labels_fake)
val_dataset = MultiTaskDataset(val_tokens, val_attention_masks, val_labels_hate, val_labels_fake)

# Initialize DataLoaders
batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


class MultiTaskBERT(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(MultiTaskBERT, self).__init__()
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(model_name)
        
        # Classification heads for each task
        self.hate_classifier = nn.Linear(self.bert.config.hidden_size, 1)
        self.fake_classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # Pass inputs through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Get [CLS] token output

        # Apply classifiers for each task
        hate_logits = self.hate_classifier(pooled_output)
        fake_logits = self.fake_classifier(pooled_output)

        return hate_logits, fake_logits

# Initialize the model
model = MultiTaskBERT()


# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
model = model.to(device)

# Optimizer and other configurations
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 10

# Directory to save the model
save_directory = "./model_checkpoints"
os.makedirs(save_directory, exist_ok=True)

# Training and evaluation loop
best_val_loss = float("inf")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    train_hate_preds, train_fake_preds, train_hate_labels, train_fake_labels = [], [], [], []

    for batch in tqdm(train_loader):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_hate = batch['label_hate'].unsqueeze(1).float().to(device)
        labels_fake = batch['label_fake'].unsqueeze(1).float().to(device)

        # Zero out previous gradients
        optimizer.zero_grad()

        # Forward pass
        hate_logits, fake_logits = model(input_ids, attention_mask)

        # Loss calculation
        loss_hate = F.binary_cross_entropy_with_logits(hate_logits, labels_hate)
        loss_fake = F.binary_cross_entropy_with_logits(fake_logits, labels_fake)
        loss = loss_hate + loss_fake
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Record predictions and labels for metrics
        train_hate_preds.extend(torch.round(torch.sigmoid(hate_logits)).detach().cpu().numpy())
        train_fake_preds.extend(torch.round(torch.sigmoid(fake_logits)).detach().cpu().numpy())
        train_hate_labels.extend(labels_hate.detach().cpu().numpy())
        train_fake_labels.extend(labels_fake.detach().cpu().numpy())

    # Calculate training metrics
    train_hate_accuracy = accuracy_score(train_hate_labels, train_hate_preds)
    train_fake_accuracy = accuracy_score(train_fake_labels, train_fake_preds)
    train_hate_precision, train_hate_recall, train_hate_f1, _ = precision_recall_fscore_support(
        train_hate_labels, train_hate_preds, average='binary')
    train_fake_precision, train_fake_recall, train_fake_f1, _ = precision_recall_fscore_support(
        train_fake_labels, train_fake_preds, average='binary')

    # Display training metrics
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {total_loss / len(train_loader):.4f}")
    print(f"Hate - Acc: {train_hate_accuracy:.4f}, Precision: {train_hate_precision:.4f}, Recall: {train_hate_recall:.4f}, F1: {train_hate_f1:.4f}")
    print(f"Fake - Acc: {train_fake_accuracy:.4f}, Precision: {train_fake_precision:.4f}, Recall: {train_fake_recall:.4f}, F1: {train_fake_f1:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    val_hate_preds, val_fake_preds, val_hate_labels, val_fake_labels = [], [], [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_hate = batch['label_hate'].unsqueeze(1).float().to(device)
            labels_fake = batch['label_fake'].unsqueeze(1).float().to(device)

            # Forward pass
            hate_logits, fake_logits = model(input_ids, attention_mask)

            # Calculate validation loss
            loss_hate = F.binary_cross_entropy_with_logits(hate_logits, labels_hate)
            loss_fake = F.binary_cross_entropy_with_logits(fake_logits, labels_fake)
            loss = loss_hate + loss_fake
            val_loss += loss.item()

            # Record predictions and labels for metrics
            val_hate_preds.extend(torch.round(torch.sigmoid(hate_logits)).detach().cpu().numpy())
            val_fake_preds.extend(torch.round(torch.sigmoid(fake_logits)).detach().cpu().numpy())
            val_hate_labels.extend(labels_hate.detach().cpu().numpy())
            val_fake_labels.extend(labels_fake.detach().cpu().numpy())

    # Calculate validation metrics
    val_hate_accuracy = accuracy_score(val_hate_labels, val_hate_preds)
    val_fake_accuracy = accuracy_score(val_fake_labels, val_fake_preds)
    val_hate_precision, val_hate_recall, val_hate_f1, _ = precision_recall_fscore_support(
        val_hate_labels, val_hate_preds, average='binary')
    val_fake_precision, val_fake_recall, val_fake_f1, _ = precision_recall_fscore_support(
        val_fake_labels, val_fake_preds, average='binary')

    # Display validation metrics
    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Hate - Acc: {val_hate_accuracy:.4f}, Precision: {val_hate_precision:.4f}, Recall: {val_hate_recall:.4f}, F1: {val_hate_f1:.4f}")
    print(f"Fake - Acc: {val_fake_accuracy:.4f}, Precision: {val_fake_precision:.4f}, Recall: {val_fake_recall:.4f}, F1: {val_fake_f1:.4f}")

    # Save model if validation loss improves
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model_save_path = os.path.join(save_directory, f"bert_multitask_epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
