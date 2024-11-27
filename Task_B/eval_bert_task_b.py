import torch
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score
warnings.filterwarnings('ignore')


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




def evaluate_model(model, tokenizer, data, max_length=128, batch_size=16, device="cpu"):
    """
    Evaluate the model's performance on a given dataset, including accuracy.
    """
    # Fill missing values and ensure proper mappings
    data['Target'] = data['Target'].fillna("N/A")
    data['Severity'] = data['Severity'].fillna("N/A")

    target_mapping = {"N/A": 0, "I": 1, "O": 2, "R": 3}
    severity_mapping = {"N/A": 0, "L": 1, "M": 2, "H": 3}

    # Filter invalid labels
    valid_targets = set(target_mapping.keys())
    valid_severities = set(severity_mapping.keys())
    data = data[data['Target'].isin(valid_targets) & data['Severity'].isin(valid_severities)]

    # Map string labels to integers
    true_target_labels = data['Target'].map(target_mapping).tolist()
    true_severity_labels = data['Severity'].map(severity_mapping).tolist()

    print("Mapped Target labels:", set(true_target_labels))
    print("Mapped Severity labels:", set(true_severity_labels))

    # Tokenize the tweets
    encoded_data = tokenizer(
        data['clean_tweets'].tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Move data to the specified device
    input_ids = encoded_data['input_ids'].to(device)
    attention_mask = encoded_data['attention_mask'].to(device)

    # Create DataLoader for batching
    dataset = torch.utils.data.TensorDataset(input_ids, attention_mask)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # Put the model in evaluation mode
    model.eval()
    model.to(device)

    # Placeholder for predictions
    target_predictions = []
    severity_predictions = []

    # Define loss function
    criterion = nn.CrossEntropyLoss()
    total_loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
            batch_input_ids, batch_attention_mask = batch
            target_logits, severity_logits = model(batch_input_ids, batch_attention_mask)

            # Convert logits to class predictions
            target_preds = torch.argmax(torch.softmax(target_logits, dim=1), dim=1)
            severity_preds = torch.argmax(torch.softmax(severity_logits, dim=1), dim=1)

            # Extract labels for the current batch
            batch_target_labels = torch.tensor(
                true_target_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size],
                dtype=torch.long,
                device=device
            )
            batch_severity_labels = torch.tensor(
                true_severity_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size],
                dtype=torch.long,
                device=device
            )

            # Calculate loss
            target_loss = criterion(target_logits, batch_target_labels)
            severity_loss = criterion(severity_logits, batch_severity_labels)
            total_loss += (target_loss.item() + severity_loss.item())

            target_predictions.extend(target_preds.cpu().numpy())
            severity_predictions.extend(severity_preds.cpu().numpy())

    # Metrics calculation
    metrics = {}
    metrics["Target"] = classification_report(true_target_labels, target_predictions, target_names=list(target_mapping.keys()))
    metrics["Severity"] = classification_report(true_severity_labels, severity_predictions, target_names=list(severity_mapping.keys()))

    # Macro F1 and Recall
    metrics["Target_F1"] = f1_score(true_target_labels, target_predictions, average='macro')
    metrics["Target_Recall"] = recall_score(true_target_labels, target_predictions, average='macro')
    metrics["Severity_F1"] = f1_score(true_severity_labels, severity_predictions, average='macro')
    metrics["Severity_Recall"] = recall_score(true_severity_labels, severity_predictions, average='macro')

    # Accuracy
    metrics["Target_Accuracy"] = accuracy_score(true_target_labels, target_predictions)
    metrics["Severity_Accuracy"] = accuracy_score(true_severity_labels, severity_predictions)

    # Loss per instance
    metrics["Loss"] = total_loss / len(data)

    return metrics




new_data = pd.read_csv("data_task_b_cleaned_test.csv")  # Replace with your dataset path


# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = MultiTaskBERT()  # Use the same MultiTaskBERT class
model.load_state_dict(torch.load("models_b/bert_multitask_best.pt"))  # Replace with your saved model path
updated_data = evaluate_model(model, tokenizer, new_data, device="cuda" if torch.cuda.is_available() else "cpu")
print(updated_data)
