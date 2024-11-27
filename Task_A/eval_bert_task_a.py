import pandas as pd
from sklearn.metrics import f1_score, recall_score, accuracy_score
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define MultiTaskBERT model
class MultiTaskBERT(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(MultiTaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.hate_classifier = nn.Linear(self.bert.config.hidden_size, 1)
        self.fake_classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        hate_logits = self.hate_classifier(pooled_output)
        fake_logits = self.fake_classifier(pooled_output)
        return hate_logits, fake_logits

# Load the trained model and tokenizer
model = MultiTaskBERT()
model.load_state_dict(torch.load("./model_checkpoints/bert_multitask_epoch_3.pt"))
model = model.to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the dataset
file_path = "test_data_task_a.csv"  # Replace with the actual path to your dataset
df = pd.read_csv(file_path)

# Loss function
loss_fn = nn.BCEWithLogitsLoss()



# Initialize variables for metrics
hate_loss_sum = 0.0
fake_loss_sum = 0.0
hate_preds = []
hate_labels = []
fake_preds = []
fake_labels = []

# Iterate through the dataset
for _, row in df.iterrows():
    text = row['clean_tweets']
    hate_label_true = row['Hate']
    fake_label_true = row['Fake']

    # Tokenize and predict
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    hate_label_true_tensor = torch.tensor([[hate_label_true]], dtype=torch.float32).to(device)
    fake_label_true_tensor = torch.tensor([[fake_label_true]], dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        hate_logits, fake_logits = model(input_ids, attention_mask)
    
    # Calculate loss
    hate_loss = loss_fn(hate_logits, hate_label_true_tensor)
    fake_loss = loss_fn(fake_logits, fake_label_true_tensor)
    hate_loss_sum += hate_loss.item()
    fake_loss_sum += fake_loss.item()

    # Collect predictions
    hate_prob = torch.sigmoid(hate_logits).item()
    fake_prob = torch.sigmoid(fake_logits).item()
    hate_preds.append(1 if hate_prob >= 0.5 else 0)
    fake_preds.append(1 if fake_prob >= 0.5 else 0)
    hate_labels.append(hate_label_true)
    fake_labels.append(fake_label_true)

# Calculate metrics
hate_f1 = f1_score(hate_labels, hate_preds, average='macro')
fake_f1 = f1_score(fake_labels, fake_preds, average='macro')
hate_recall = recall_score(hate_labels, hate_preds, average='macro')
fake_recall = recall_score(fake_labels, fake_preds, average='macro')
hate_accuracy = accuracy_score(hate_labels, hate_preds)
fake_accuracy = accuracy_score(fake_labels, fake_preds)

# Summarize results
results = {
    "Hate Loss": hate_loss_sum / len(df),
    "Fake Loss": fake_loss_sum / len(df),
    "Hate F1 Score (Macro)": hate_f1,
    "Fake F1 Score (Macro)": fake_f1,
    "Hate Recall": hate_recall,
    "Fake Recall": fake_recall,
    "Hate Accuracy": hate_accuracy,
    "Fake Accuracy": fake_accuracy
}

# Print results
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")
