import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertModel, AdamW
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess the dataset
df = pd.read_csv('data_task_b_cleaned.csv')



# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenization function
def tokenize_tweets(texts, max_length=128):
    return tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

# Tokenize tweets
tokens = tokenize_tweets(df['clean_tweets'].fillna(''))

# Prepare labels
labels_target = torch.tensor(df['Target'].values)
labels_severity = torch.tensor(df['Severity'].values)

# Train-test split
train_tokens, val_tokens, train_target, val_target, train_severity, val_severity = train_test_split(
    tokens['input_ids'], labels_target, labels_severity, test_size=0.2, random_state=42
)
train_attention_masks, val_attention_masks = train_test_split(tokens['attention_mask'], test_size=0.2, random_state=42)

# Define dataset class
class MultiTaskDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels_target, labels_severity):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels_target = labels_target
        self.labels_severity = labels_severity

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'label_target': self.labels_target[idx],
            'label_severity': self.labels_severity[idx]
        }

# Create datasets and loaders
train_dataset = MultiTaskDataset(train_tokens, train_attention_masks, train_target, train_severity)
val_dataset = MultiTaskDataset(val_tokens, val_attention_masks, val_target, val_severity)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define the Multi-task BERT model
class MultiTaskBERT(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(MultiTaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.target_classifier = nn.Linear(self.bert.config.hidden_size, 4)  # 4 classes: 0, 1, 2, 3
        self.severity_classifier = nn.Linear(self.bert.config.hidden_size, 4)  # 4 classes: 0, 1, 2, 3

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        target_logits = self.target_classifier(pooled_output)
        severity_logits = self.severity_classifier(pooled_output)
        return target_logits, severity_logits

# Initialize model, optimizer, and device
model = MultiTaskBERT()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Training and evaluation loop
num_epochs = 20
best_val_f1 = 0

for epoch in range(num_epochs):
    # Training
    model.train()
    total_loss = 0
    train_target_preds, train_severity_preds, train_target_labels, train_severity_labels = [], [], [], []
    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_target = batch['label_target'].to(device)
        labels_severity = batch['label_severity'].to(device)

        optimizer.zero_grad()
        target_logits, severity_logits = model(input_ids, attention_mask)

        loss_target = criterion(target_logits.float(), labels_target.long())
        loss_severity = criterion(severity_logits.float(), labels_severity.long())

        loss = loss_target + loss_severity
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        train_target_preds.extend(torch.argmax(target_logits, dim=1).cpu().numpy())
        train_severity_preds.extend(torch.argmax(severity_logits, dim=1).cpu().numpy())
        train_target_labels.extend(labels_target.cpu().numpy())
        train_severity_labels.extend(labels_severity.cpu().numpy())

    # Calculate training metrics
    train_target_f1 = precision_recall_fscore_support(train_target_labels, train_target_preds, average='macro')[2]
    train_severity_f1 = precision_recall_fscore_support(train_severity_labels, train_severity_preds, average='macro')[2]
    print(f"Epoch {epoch + 1}: Train Loss = {total_loss / len(train_loader):.4f}, "
          f"Target F1 = {train_target_f1:.4f}, Severity F1 = {train_severity_f1:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    val_target_preds, val_severity_preds, val_target_labels, val_severity_labels = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_target = batch['label_target'].to(device)
            labels_severity = batch['label_severity'].to(device)

            target_logits, severity_logits = model(input_ids, attention_mask)
            loss_target = criterion(target_logits.float(), labels_target.long())
            loss_severity = criterion(severity_logits.float(), labels_severity.long())

            loss = loss_target + loss_severity
            val_loss += loss.item()

            val_target_preds.extend(torch.argmax(target_logits, dim=1).cpu().numpy())
            val_severity_preds.extend(torch.argmax(severity_logits, dim=1).cpu().numpy())
            val_target_labels.extend(labels_target.cpu().numpy())
            val_severity_labels.extend(labels_severity.cpu().numpy())

    # Calculate validation metrics
    val_target_f1 = precision_recall_fscore_support(val_target_labels, val_target_preds, average='macro')[2]
    val_severity_f1 = precision_recall_fscore_support(val_severity_labels, val_severity_preds, average='macro')[2]
    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss = {avg_val_loss:.4f}, "
          f"Target F1 = {val_target_f1:.4f}, Severity F1 = {val_severity_f1:.4f}")

    # Save the best model
    avg_val_f1 = (val_target_f1 + val_severity_f1) / 2
    if avg_val_f1 > best_val_f1:
        best_val_f1 = avg_val_f1
        torch.save(model.state_dict(), f"models_b/bert_multitask_best.pt")
        print(f"Saved Best Model (Epoch {epoch + 1})")
