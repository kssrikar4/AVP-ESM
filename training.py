import concurrent.futures
import os, random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.manifold import TSNE
import umap
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import EsmTokenizer, EsmForSequenceClassification, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from Bio import SeqIO
from tqdm import tqdm
from tqdm.autonotebook import tqdm as notebook_tqdm
from Bio.Align import substitution_matrices
from torch.nn.functional import softmax

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed()
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")
def plot(positive_file, negative_file):
    pos_lengths = [len(record.seq) for record in SeqIO.parse(positive_file, "fasta")]
    neg_lengths = [len(record.seq) for record in SeqIO.parse(negative_file, "fasta")]
    plt.figure(figsize=(10, 6))
    sns.histplot(pos_lengths, color='skyblue', label='Positive Samples', kde=True)
    sns.histplot(neg_lengths, color='salmon', label='Negative Samples', kde=True)
    plt.title('Distribution of Sequence Lengths')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
positive_file_path = "positive.fasta"
negative_file_path = "negative.fasta"
def load(positive_file, negative_file):
    positive_sequences = list(SeqIO.parse(positive_file, "fasta"))
    negative_sequences = list(SeqIO.parse(negative_file, "fasta"))
    pos_seqs = [str(record.seq) for record in positive_sequences]
    neg_seqs = [str(record.seq) for record in negative_sequences]
    sequences = pos_seqs + neg_seqs
    labels = [1] * len(pos_seqs) + [0] * len(neg_seqs)
    return sequences, labels
class ProteinSequenceDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            seq,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
sequences, labels = load(positive_file_path, negative_file_path)
train_seqs, test_seqs, train_labels, test_labels = train_test_split(
    sequences, labels, test_size=0.2, random_state=42, stratify=labels
)
model_name = "facebook/esm2_t6_8m_ur50d"
tokenizer = EsmTokenizer.from_pretrained(model_name)
max_length = 512
full_dataset = ProteinSequenceDataset(train_seqs, train_labels, tokenizer, max_length)
test_dataset = ProteinSequenceDataset(test_seqs, test_labels, tokenizer, max_length)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
initial_labeled_size = 1000
query_size = 2000
num_active_learning_iterations = int((len(full_dataset) - initial_labeled_size) / query_size)
training_epochs = 3
all_indices = np.arange(len(full_dataset))
np.random.shuffle(all_indices)
labeled_indices = all_indices[:initial_labeled_size].tolist()
unlabeled_indices = all_indices[initial_labeled_size:].tolist()
def evaluate(model, dataloader):
    model.eval()
    predictions, true_labels, confidences = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            probs = softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            confidences.extend(probs.cpu().numpy())
    accuracy = (np.array(predictions) == np.array(true_labels)).mean()
    f1 = f1_score(true_labels, predictions, average='macro')
    return accuracy, f1, np.array(confidences)
def train(model, optimizer, scheduler, dataloader, epochs):
    model.train()
    for epoch in notebook_tqdm(range(epochs), desc="Training"):
        for batch in dataloader:
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} completed, Loss: {loss.item():.4f}")
def save(filepath, model, optimizer, scheduler, labeled_indices, unlabeled_indices, iteration, history):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'labeled_indices': labeled_indices,
        'unlabeled_indices': unlabeled_indices,
        'iteration': iteration,
        'history': history
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")
def load_checkpoint(filepath, model, optimizer, scheduler, device):
    if not os.path.exists(filepath):
        print(f"No checkpoint found at {filepath}. Starting from scratch.")
        return None, None, None, None, None
    print(f"Loading checkpoint from {filepath}")
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    labeled_indices = checkpoint['labeled_indices']
    unlabeled_indices = checkpoint['unlabeled_indices']
    iteration = checkpoint['iteration']
    history = checkpoint['history']
    print(f"Resumed from iteration {iteration + 1} with {len(labeled_indices)} labeled samples.")
    return labeled_indices, unlabeled_indices, iteration, history, checkpoint
checkpoint_path = "active_esm_checkpoint.pth"
model = EsmForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
total_steps = (initial_labeled_size + query_size * num_active_learning_iterations) * training_epochs
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
labeled_indices, unlabeled_indices, start_iteration, history, _ = load_checkpoint(
    checkpoint_path, model, optimizer, scheduler, device
)
if labeled_indices is None:
    print(f"No checkpoint found. Starting fresh.")
    all_indices = np.arange(len(full_dataset))
    np.random.shuffle(all_indices)
    labeled_indices = all_indices[:initial_labeled_size].tolist()
    unlabeled_indices = all_indices[initial_labeled_size:].tolist()
    start_iteration = 0
    history = {'labeled_size': [], 'validation_accuracy': [], 'validation_f1': []}
    print(f"Starting active learning with an initial labeled set of {len(labeled_indices)} samples.")
    print(f"There will be {num_active_learning_iterations} active learning iterations, each querying {query_size} samples.")
for iteration in range(start_iteration, num_active_learning_iterations + 1):
    print(f"\n--- Active Learning Iteration {iteration+1}/{num_active_learning_iterations + 1} ---")
    labeled_subset = Subset(full_dataset, labeled_indices)
    labeled_loader = DataLoader(labeled_subset, batch_size=8, shuffle=True)
    train(model, optimizer, scheduler, labeled_loader, training_epochs)
    val_indices = np.random.choice(labeled_indices, size=int(len(labeled_indices)*0.1), replace=False)
    val_subset = Subset(full_dataset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    val_acc, val_f1, _ = evaluate(model, val_loader)
    history['labeled_size'].append(len(labeled_indices))
    history['validation_accuracy'].append(val_acc)
    history['validation_f1'].append(val_f1)
    print(f"Labeled data size: {len(labeled_indices)}")
    print(f"Validation Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}")
    save(checkpoint_path, model, optimizer, scheduler, labeled_indices, unlabeled_indices, iteration, history)
    if not unlabeled_indices:
        print("All data has been labeled. Ending active learning loop.")
        break
    unlabeled_subset = Subset(full_dataset, unlabeled_indices)
    unlabeled_loader = DataLoader(unlabeled_subset, batch_size=32, shuffle=False)
    _, _, unlabeled_confidences = evaluate(model, unlabeled_loader)
    uncertainty = 1 - np.max(unlabeled_confidences, axis=1)
    uncertain_indices = np.argsort(uncertainty)[-query_size:]
    queried_global_indices = np.array(unlabeled_indices)[uncertain_indices]
    labeled_indices.extend(queried_global_indices.tolist())
    unlabeled_indices = np.delete(unlabeled_indices, uncertain_indices).tolist()
print("\nActive learning loop finished.")
output_dir = "./active_esm_8m"
print(f"Saving the final model to {output_dir}")
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("\n--- Final Evaluation on Test Set ---")
test_acc, test_f1, test_confidences = evaluate(model, test_loader)
print(f"Final Test Accuracy: {test_acc:.4f}")
print(f"Final Test F1 Score: {test_f1:.4f}")
test_preds = np.argmax(test_confidences, axis=1)
test_true_labels = test_dataset.labels
report = classification_report(test_true_labels, test_preds, target_names=["Negative", "Positive"])
print("\nFinal Classification Report:\n", report)
plt.figure(figsize=(10, 6))
plt.plot(history['labeled_size'], history['validation_accuracy'], '-o', label='Validation Accuracy')
plt.plot(history['labeled_size'], history['validation_f1'], '-o', label='Validation F1 Score')
plt.title('Active Learning Performance')
plt.xlabel('Number of Labeled Samples')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()
cm = confusion_matrix(test_true_labels, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.title('Final Confusion Matrix on Test Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
all_logits, y_true, y_pred, y_conf = [], [], [], []
model.eval()
with torch.no_grad():
    for batch in val_loader:
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        max_probs, preds = torch.max(probs, dim=1)
        y_conf.extend(max_probs.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(inputs["labels"].cpu().numpy())
print(classification_report(y_true, y_pred, target_names=["Non-Virulent", "Virulent"]))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Virulent", "Virulent"], yticklabels=["Non-Virulent", "Virulent"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
uncertain = [i for i, p in enumerate(y_conf) if p < 0.6]
print(f"Uncertain predictions (confidence < 0.6): {len(uncertain)}")
def get_cls_embeddings(dataloader):
    model.eval()
    cls_embeddings, labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Getting embeddings"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            outputs = model(**inputs, output_hidden_states=True)
            cls = outputs.hidden_states[-1][:, 0, :]
            cls_embeddings.append(cls.cpu())
            labels.extend(batch['labels'].cpu().numpy())
    return torch.cat(cls_embeddings).numpy(), np.array(labels)
cls_embeds, cls_labels = get_cls_embeddings(test_loader)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
reduced_tsne = tsne.fit_transform(cls_embeds)
plt.figure(figsize=(8,6))
plt.scatter(reduced_tsne[:,0], reduced_tsne[:,1], c=cls_labels, cmap="coolwarm", alpha=0.7)
plt.title("t-SNE of CLS Embeddings (Final Model)")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.grid(True)
plt.colorbar(label="Label")
plt.show()
umap_reducer = umap.UMAP(n_components=2, random_state=42)
reduced_umap = umap_reducer.fit_transform(cls_embeds)
plt.figure(figsize=(8,6))
plt.scatter(reduced_umap[:,0], reduced_umap[:,1], c=cls_labels, cmap="coolwarm", alpha=0.7)
plt.title("UMAP of CLS Embeddings")
plt.xlabel("UMAP Dim 1")
plt.ylabel("UMAP Dim 2")
plt.grid(True)
plt.colorbar(label="Label")
plt.show()

# Note: The same code is used to train the 35m and 150m models, but the batch size and query_size are decreased to accommodate its larger memory footprint and save training time.

