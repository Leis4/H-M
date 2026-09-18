# =======================================================
# =======================================================
# IMPORTS
# =======================================================
# =======================================================
import os
import duckdb
import numpy as np
import pandas as pd
import optuna
import json

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision.transforms import v2

import timm

from src.Preprocessor.preprocessor import data_preprocessor
from src.Analyze_data.export_analyse import load_features_from_db

# =======================================================
# LOAD PREPARED DATA
# =======================================================
df = load_features_from_db()
(X_train_proc, X_test_proc,
 y_train, y_test,
 paths_train, paths_test,
 ids_train, ids_test,
 preprocessor) = data_preprocessor(df)

num_features = X_train_proc.shape[1]


# =======================================================
# DATA TO TENSOR DATASET
# =======================================================
class HandMDataset(Dataset):
    def __init__(self, table, label, ids, path, transform=None):
        self.table = np.array(table, dtype=np.float32)
        self.label = np.array(label, dtype=np.float32)
        self.ids = np.array(ids)
        self.path = np.array(path)
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        table_tens = torch.tensor(self.table[idx], dtype=torch.float32)
        label_tens = torch.tensor([self.label[idx]], dtype=torch.float32)

        image_path = self.path[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            img_tens = self.transform(image)
        else:
            img_tens = ToTensor()(image)

        art_id = self.ids[idx]

        return table_tens, img_tens, label_tens, art_id


# =======================================================
# MODEL ARCHITECTURE
# =======================================================
class HM_Model(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.table_net = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        table_dim = 64

        self.image_net = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        img_features_dim = self.image_net.num_features

        full_num_dim = table_dim + img_features_dim

        self.regressor = nn.Sequential(
            nn.Linear(full_num_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(256, 1),
        )

    def forward(self, tab, image):
        img_feats = self.image_net(image)
        tab_feats = self.table_net(tab)

        combined = torch.cat((img_feats, tab_feats), dim=1)
        price_pred = self.regressor(combined)

        return price_pred


# =======================================================
# IMAGE TRANSFORMERS
# =======================================================
IMAGE_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

img_train_transform = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandAugment(num_ops=2, magnitude=5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

test_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(size=int(IMAGE_SIZE * 1.15)),
    v2.CenterCrop(size=(IMAGE_SIZE, IMAGE_SIZE)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

# =======================================================
# DATALOADERS & TRAINING LOGIC
# =======================================================
train_dataset = HandMDataset(X_train_proc, y_train, ids_train, paths_train, transform=img_train_transform)
test_dataset = HandMDataset(X_test_proc, y_test, ids_test, paths_test, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for tab_batch, img_batch, price_batch, _ in dataloader:
        tab_batch = tab_batch.to(device)
        img_batch = img_batch.to(device)
        price_batch = price_batch.to(device)

        optimizer.zero_grad()

        preds = model(tab_batch, img_batch)
        loss = criterion(preds, price_batch)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * tab_batch.size(0)

    return running_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for tab_batch, img_batch, price_batch, _ in dataloader:
            tab_batch = tab_batch.to(device)
            img_batch = img_batch.to(device)
            price_batch = price_batch.to(device)

            preds = model(tab_batch, img_batch)
            loss = criterion(preds, price_batch)

            running_loss += loss.item() * tab_batch.size(0)

    return running_loss / len(dataloader.dataset)


# =======================================================
# INITIALIZATION & DEVICE SETUP
# =======================================================
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(device)
# =======================================================
# =======================================================


# def objective(trial):
#     weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
#     lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
#     model = HM_Model(num_features=num_features).to(device)
#     criterion = nn.SmoothL1Loss()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#
#     for epoch in range(4):
#         train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
#         val_loss = evaluate(model, test_loader, criterion, device)
#
#     del model
#     del optimizer
#     del criterion
#     torch.cuda.empty_cache()
#
#     return val_loss
#
#
# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=10)
# json_file_path = r'D:\Sultan_PC\01_Projects\PyCharmProjects\Project_2_1\src\Models\Best_params\PyTorch\best_params.json'
#
#
# print(f"best loss: {study.best_value:.6f}")
# for key, value in study.best_params.items():
#     print(f"  {key}: {value}")
#
# with open(json_file_path, 'w', encoding='utf-8') as f:
#     json.dump(study.best_params, f, indent=4)
# print('final process')


#RESULTS OF OPTUNA
# [I 2026-09-17 14:33:19,586] Trial 0 finished with value: 0.00011282763781927234 and parameters: {'weight_decay': 0.0022813137282213327, 'lr': 0.00048110206956416754}. Best is trial 0 with value: 0.00011282763781927234.
# [I 2026-09-17 15:10:06,897] Trial 1 finished with value: 0.0001672413780828732 and parameters: {'weight_decay': 0.07432804147323774, 'lr': 0.003671629022790669}. Best is trial 0 with value: 0.00011282763781927234.
# [I 2026-09-17 15:47:04,402] Trial 2 finished with value: 0.00012724268624933634 and parameters: {'weight_decay': 0.002389876325113686, 'lr': 0.0007106794969368067}. Best is trial 0 with value: 0.00011282763781927234.
# [I 2026-09-17 16:24:05,882] Trial 3 finished with value: 0.0014869724699020067 and parameters: {'weight_decay': 0.0017255464449637012, 'lr': 0.006327262083826735}. Best is trial 0 with value: 0.00011282763781927234.
# [I 2026-09-17 17:00:55,645] Trial 4 finished with value: 0.0001712652466 5158916 and parameters: {'weight_decay': 0.00431984802737235, 'lr': 0.02640142830394947}. Best is trial 0 with value: 0.00011282763781927234.
# [I 2026-09-17 17:37:37,553] Trial 5 finished with value: 7.910011025687337e-05 and parameters: {'weight_decay': 0.027378980178833747, 'lr': 0.0054520159969189755}. Best is trial 5 with value: 7.910011025687337e-05.
# [I 2026-09-17 18:14:33,029] Trial 6 finished with value: 6.0355299777374284e-05 and parameters: {'weight_decay': 0.0011090291187001431, 'lr': 0.0002294229333690208}. Best is trial 6 with value: 6.0355299777374284e-05.
# [I 2026-09-17 18:51:26,403] Trial 7 finished with value: 7.322684611307875e-05 and parameters: {'weight_decay': 0.001180888756026524, 'lr': 0.00043055091057622513}. Best is trial 6 with value: 6.0355299777374284e-05.




# =======================================================
# FINAL PYTORCH VS CATBOOST
# =======================================================

best_lr = 0.0002294229333690208
best_wd = 0.0011090291187001431

final_model = HM_Model(num_features=num_features).to(device)
final_criterion = nn.SmoothL1Loss()
final_optimizer = torch.optim.AdamW(final_model.parameters(), lr=best_lr, weight_decay=best_wd)

FINAL_EPOCHS = 3
for epoch in range(FINAL_EPOCHS):
    train_one_epoch(final_model, train_loader, final_criterion, final_optimizer, device)
    print(f"Epoch progress [{epoch + 1}/{FINAL_EPOCHS}]")

# =======================================================
# R2 SCORING
# =======================================================
final_model.eval()

all_predictions = []
all_true_prices = []

with torch.no_grad():
    for tab_batch, img_batch, price_batch, _ in test_loader:
        tab_batch = tab_batch.to(device)
        img_batch = img_batch.to(device)

        preds = final_model(tab_batch, img_batch)

        all_predictions.extend(preds.view(-1).tolist())
        all_true_prices.extend(price_batch.view(-1).tolist())

pytorch_r2 = r2_score(all_true_prices, all_predictions)
catboost_r2 = 0.8097
mae = np.mean(np.abs(np.array(all_true_prices) - np.array(all_predictions)))

print(f"  PyTorch R2 Score:  {pytorch_r2:.4f}")
print(f"  CatBoost R2 Score: {catboost_r2:.4f}")
print(f"\n MAE: {mae:.6f}")

