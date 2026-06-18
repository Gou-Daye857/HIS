# module_fl/fl_client.py
import flwr as fl
import torch
import argparse
from collections import OrderedDict
from fl_models import BareMetalUnet
from fl_dataset import HAM10000NodeDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HospitalClient(fl.client.NumPyClient):
    def __init__(self, node_id, db_path, local_epochs):
        self.node_id = node_id
        self.local_epochs = local_epochs
        self.model = BareMetalUnet(num_classes=7).to(DEVICE)
        self.dataset = HAM10000NodeDataset(db_path)
        self.loader = DataLoader(self.dataset, batch_size=4, shuffle=True, drop_last=True)
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_seg = nn.BCEWithLogitsLoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

        running_loss = 0.0
        running_corrects = 0
        running_dice = 0.0
        total_samples = 0
        total_batches = 0

        for epoch in range(self.local_epochs):
            for imgs, masks, lbls in self.loader:
                imgs, masks, lbls = imgs.to(DEVICE), masks.to(DEVICE), lbls.to(DEVICE)
                optimizer.zero_grad()
                cls_out, seg_out = self.model(imgs)

                loss_cls = self.criterion_cls(cls_out, lbls)
                loss_seg = self.criterion_seg(seg_out, masks)
                loss = 0.5 * loss_cls + 0.5 * loss_seg
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(cls_out, 1)
                running_corrects += torch.sum(preds == lbls.data).item()
                total_samples += imgs.size(0)

                seg_preds = (torch.sigmoid(seg_out) > 0.5).float()
                intersection = (seg_preds * masks).sum().item()
                union = seg_preds.sum().item() + masks.sum().item()
                running_dice += (2.0 * intersection) / (union + 1e-8)
                total_batches += 1

        epoch_loss = running_loss / total_batches if total_batches > 0 else 0.0
        epoch_acc = running_corrects / total_samples if total_samples > 0 else 0.0
        epoch_dice = running_dice / total_batches if total_batches > 0 else 0.0

        # 🚨 确保必须包含 loss, accuracy, dice
        metrics = {
            "loss": float(epoch_loss),
            "accuracy": float(epoch_acc),
            "dice": float(epoch_dice)
        }

        return self.get_parameters(config={}), len(self.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return float(0.0), len(self.dataset), {"accuracy": 0.0}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node_id", type=str, required=True)
    parser.add_argument("--db_path", type=str, required=True)
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    client = HospitalClient(args.node_id, args.db_path, args.epochs)
    fl.client.start_numpy_client(server_address=args.server_address, client=client)