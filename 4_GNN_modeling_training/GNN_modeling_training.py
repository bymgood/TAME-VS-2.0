#!/usr/bin/env python
# coding: utf-8
# Author: Yuemin Bian

print('Module 4_2: GNN prediction model training')

import numpy as np
import pandas as pd
import argparse
import pickle
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from imblearn.metrics import geometric_mean_score, make_index_balanced_accuracy
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from gnn_model import GNN 

# # Parse input
parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
parser.add_argument('--device', type=int, default=0, required=False,
                        help='which gpu to use if any (default: 0)')
parser.add_argument('--batch_size', type=int, default=512, required=False,
                        help='input batch size for training (default: 512)')
parser.add_argument('--epochs', type=int, default=200, required=False,
                        help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.001, required=False,
                        help='learning rate (default: 0.001)')
parser.add_argument('--decay', type=float, default=1e-4, required=False,
                        help='weight decay (default: 1e-4)')
parser.add_argument('--num_layer', type=int, default=5, required=False,
                        help='number of GNN message passing layers (default: 5)')
parser.add_argument('--emb_dim', type=int, default=512, required=False,
                        help='embedding dimensions (default: 512)')
parser.add_argument('--dropout_ratio', type=float, default=0.2, required=False,
                        help='dropout ratio (default: 0.2)')
parser.add_argument('--JK', type=str, default='concat', required=False,
                        help='how the node features across layers are combined. concat, sum, max or last')
parser.add_argument('--gnn_type', type=str, default='gcn', required=False,
                        help='gnn type. gcn, gin, gat, graphsage')
parser.add_argument('--num_workers', type=int, default=8, required=False, 
                        help='number of workers for dataset loading')
parser.add_argument('-u', required=True, help='The uniprot id of your target')                        

args = parser.parse_args()


# # Make functions
def load_data(batch_size=512, num_workers=8):

    with open(args.u+'_gnn_train_data.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    with open(args.u+'_gnn_val_data.pkl', 'rb') as f:
        val_dataset = pickle.load(f)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

    return train_loader, val_loader
    
class Trainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6)
        self.criterion = self.get_loss_criterion(train_loader.dataset)
        self.train_losses = []
        self.train_metrics = []
        self.val_metrics = []

    def get_loss_criterion(self, dataset):
        pos_count = sum(1 for data in dataset if data.y.item() == 1)
        pos_ratio = pos_count / len(dataset)
        class_weights = torch.tensor([1.0, (1 - pos_ratio) / pos_ratio]).to(device)
        return torch.nn.CrossEntropyLoss(weight=class_weights)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_samples = 0
        for batch in tqdm(self.train_loader, desc="Training"):
            batch = batch.to(device)
            self.optimizer.zero_grad()
            out = self.model(batch)
            loss = self.criterion(out, batch.y.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            self.optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs
        return total_loss / total_samples

    def evaluate(self, loader, desc="Evaluating"):
        self.model.eval()
        y_true, y_pred, y_score = [], [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                batch = batch.to(device)
                out = self.model(batch)
                pred = out.argmax(dim=1)
                y_true.extend(batch.y.cpu().numpy().flatten())
                y_pred.extend(pred.cpu().numpy())
                y_score.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
        metrics = self.compute_metrics(y_true, y_pred, y_score)
        return metrics

    def compute_metrics(self, y_true, y_pred, y_score):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        gmean = geometric_mean_score(y_true, y_pred)
        iba_gmean = make_index_balanced_accuracy(alpha=0.1, squared=True)(geometric_mean_score)
        iba = iba_gmean(y_true, y_pred)
        return {
            "accuracy": sum(1 for y_p, y_t in zip(y_pred, y_true) if y_p == y_t) / len(y_true),
            "auc_roc": roc_auc_score(y_true, y_score),
            "f1": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "specificity": specificity,
            "gmean": gmean,
            "iba": iba
        }
    
    def save_model(self):
        model_path = f'gnn_model.pth'  
        torch.save(self.model, model_path)
        print(f'Model saved to {model_path}')

    def run(self):
        for epoch in range(1, args.epochs + 1):
            epoch_loss = self.train_epoch()
            train_metric = self.evaluate(self.train_loader, desc="Evaluating Train")
            val_metric = self.evaluate(self.val_loader, desc="Evaluating Val")
            self.scheduler.step(val_metric["auc_roc"])
            self.train_losses.append(epoch_loss)
            self.train_metrics.append(train_metric)
            self.val_metrics.append(val_metric)
            print(f'Epoch: {epoch:03d}, '
                  f'Train Loss: {epoch_loss:.4f}, '
                  f'Train Acc: {train_metric["accuracy"]:.4f}, '
                  f'Val Acc: {val_metric["accuracy"]:.4f}, '
                  f'Val AUC: {val_metric["auc_roc"]:.4f}, '
                  f'Val Precision: {val_metric["precision"]:.4f}, '
                  f'Val Recall: {val_metric["recall"]:.4f}, '
                  f'Val Specificity: {val_metric["specificity"]:.4f}, '
                  f'Val F1: {val_metric["f1"]:.4f}, '
                  f'Val G-Mean: {val_metric["gmean"]:.4f}, '
                  f'Val IBA: {val_metric["iba"]:.4f}')
            
            if epoch == args.epochs:
                self.save_model()      



def plot_training_dashboard(trainer):
    train_metrics = trainer.train_metrics
    val_metrics = trainer.val_metrics
    train_losses = trainer.train_losses

    train_accuracies = [m["accuracy"] for m in train_metrics]
    val_accuracies = [m["accuracy"] for m in val_metrics]
    train_aucs = [m["auc_roc"] for m in train_metrics]
    val_aucs = [m["auc_roc"] for m in val_metrics]
    train_precisions = [m["precision"] for m in train_metrics]
    val_precisions = [m["precision"] for m in val_metrics]
    train_recalls = [m["recall"] for m in train_metrics]
    val_recalls = [m["recall"] for m in val_metrics]
    train_specificities = [m["specificity"] for m in train_metrics]
    val_specificities = [m["specificity"] for m in val_metrics]
    train_gmeans = [m["gmean"] for m in train_metrics]
    val_gmeans = [m["gmean"] for m in val_metrics]
    train_ibas = [m["iba"] for m in train_metrics]
    val_ibas = [m["iba"] for m in val_metrics]

    plt.figure(figsize=(20, 15), facecolor='white')

    TRAIN_COLOR = '#2c7bb6' 
    VAL_COLOR = '#d7191c'   
    GM_COLOR = '#008837'     
    IBA_COLOR = '#7b3294'   
    PRECISION_COLOR = '#2c7bb6'  
    RECALL_COLOR = '#fdae61'     
    SPECIFICITY_COLOR = '#5e3c99' 

    line_style = {
        'train': {'linewidth': 2, 'alpha': 0.9, 'linestyle': '-'},
        'val': {'linewidth': 2, 'alpha': 0.9, 'linestyle': '--'}
    }
    fill_style = {'alpha': 0.2, 'interpolate': True}

    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'legend.fontsize': 10
    })

    # 1. Training Loss
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(train_losses, label='Training', color=TRAIN_COLOR, **line_style['train'])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss', pad=12)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(framealpha=0.9)

    # 2. Accuracy
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(train_accuracies, label='Train', color=TRAIN_COLOR, **line_style['train'])
    ax2.plot(val_accuracies, label='Validation', color=VAL_COLOR, **line_style['val'])
    ax2.fill_between(range(len(train_accuracies)), 
                    train_accuracies, val_accuracies, 
                    where=(np.array(train_accuracies) > np.array(val_accuracies)),
                    color='green', **fill_style)
    ax2.fill_between(range(len(train_accuracies)), 
                    train_accuracies, val_accuracies,
                    where=(np.array(train_accuracies) <= np.array(val_accuracies)),
                    color='red', **fill_style)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Model Accuracy', pad=12)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend(framealpha=0.9)

    # 3. AUC-ROC 
    window_size = 3
    smoothed_train_auc = np.convolve(train_aucs, np.ones(window_size)/window_size, mode='valid')
    smoothed_val_auc = np.convolve(val_aucs, np.ones(window_size)/window_size, mode='valid')

    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(smoothed_train_auc, label=f'Train (smoothed {window_size}ep)', 
             color=TRAIN_COLOR, **line_style['train'])
    ax3.plot(smoothed_val_auc, label=f'Validation (smoothed {window_size}ep)',
             color=VAL_COLOR, **line_style['val'])
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('AUC-ROC')
    ax3.set_title('ROC Area Under Curve', pad=12)
    ax3.grid(True, linestyle=':', alpha=0.7)
    ax3.legend(framealpha=0.9)

    # 4. Precision-Recall-Specificity
    ax4 = plt.subplot(3, 3, (4,6))
    ax4.plot(train_precisions, label='Train Precision', color=PRECISION_COLOR, **line_style['train'])
    ax4.plot(val_precisions, label='Val Precision', color=PRECISION_COLOR, **line_style['val'])
    ax4.plot(train_recalls, label='Train Recall', color=RECALL_COLOR, **line_style['train'])
    ax4.plot(val_recalls, label='Val Recall', color=RECALL_COLOR, **line_style['val'])
    ax4.plot(train_specificities, label='Train Specificity', color=SPECIFICITY_COLOR, **line_style['train'])
    ax4.plot(val_specificities, label='Val Specificity', color=SPECIFICITY_COLOR, **line_style['val'])
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Score')
    ax4.set_title('Precision-Recall-Specificity Tradeoff', pad=12)
    ax4.grid(True, linestyle=':', alpha=0.7)
    ax4.legend(ncol=2, framealpha=0.9)

    # 5. G-Mean & IBA
    ax5 = plt.subplot(3, 3, (7,9))
    ax5.plot(train_gmeans, label='Train G-Mean', color=GM_COLOR, **line_style['train'])
    ax5.plot(val_gmeans, label='Val G-Mean', color=GM_COLOR, **line_style['val'])
    ax5.plot(train_ibas, label='Train IBA', color=IBA_COLOR, **line_style['train'])
    ax5.plot(val_ibas, label='Val IBA', color=IBA_COLOR, **line_style['val'])
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Score')
    ax5.set_title('Imbalance-aware Metrics', pad=12)
    ax5.grid(True, linestyle=':', alpha=0.7)
    ax5.legend(framealpha=0.9)

    plt.tight_layout(pad=3.0, w_pad=2.0, h_pad=2.0)
    plt.suptitle('Model Training Metrics Dashboard', y=1.02, fontsize=16, fontweight='bold')
    plt.savefig('training_metrics_dashboard.png', dpi=300, bbox_inches='tight')
    
    
    
# use function

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

train_loader, val_loader = load_data(
    batch_size=args.batch_size,
    num_workers=args.num_workers,
)

model = GNN(num_layer=args.num_layer, emb_dim=args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type, output_dim=2)
trainer = Trainer(model, train_loader, val_loader)

trainer.run()

train_metrics=trainer.train_metrics
val_metrics=trainer.val_metrics
train_df = pd.DataFrame(train_metrics)
train_df.to_csv("train_metrics.csv", index=False)
val_df = pd.DataFrame(val_metrics)
val_df.to_csv("val_metrics.csv", index=False)

plot_training_dashboard(trainer)
