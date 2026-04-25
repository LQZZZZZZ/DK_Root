import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

from models.loss import SupConLoss


class GCELoss(nn.Module):
    """Generalized cross-entropy loss for noisy-label classification."""

    def __init__(self, q=0.7, reduction="mean"):
        super(GCELoss, self).__init__()
        self.q = q
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        pt = probs[range(len(targets)), targets].clamp(min=1e-9, max=1.0)
        loss = (1.0 - pt.pow(self.q)) / self.q
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def Trainer(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device,
            logger, config, experiment_log_dir, training_mode, seed):
    logger.debug("Training started ....")
    criterion = nn.CrossEntropyLoss() if config.loss_type == "CE" else GCELoss(q=config.GCE_q)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, "min")
    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    initial_checkpoint = {
        "model_state_dict": model.state_dict(),
        "temporal_contr_model_state_dict": temporal_contr_model.state_dict(),
    }
    torch.save(initial_checkpoint, os.path.join(experiment_log_dir, "saved_models", "ckp_initial.pt"))
    logger.debug("Initial model checkpoint saved as 'ckp_initial.pt'.")

    for epoch in range(1, config.num_epoch + 1):
        train_results = model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer,
                                    criterion, train_dl, config, device, training_mode, logger)
        train_loss, train_acc, train_top2_acc, _, _, _, train_metrics = train_results
        valid_loss, valid_acc, valid_top2_acc, valid_metrics = model_evaluate(
            model, temporal_contr_model, valid_dl, device, training_mode, config, logger, "[Valid]", seed
        )
        if training_mode not in ["self_supervised", "SupCon"]:
            scheduler.step(valid_loss)
        log_training_results(logger, epoch, train_loss, train_acc, train_top2_acc, valid_loss, valid_acc,
                             valid_top2_acc, training_mode, train_metrics, valid_metrics)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "temporal_contr_model_state_dict": temporal_contr_model.state_dict(),
    }
    torch.save(checkpoint, os.path.join(experiment_log_dir, "saved_models", "ckp_last.pt"))

    if training_mode not in ["self_supervised", "SupCon"]:
        test_loss, test_acc, test_top2_acc, test_metrics = model_evaluate(
            model, temporal_contr_model, test_dl, device, training_mode, config, logger, "[Test]", seed
        )
        logger.debug("\nEvaluate on the test set:")
        logger.debug(f"Test loss      : {test_loss:2.4f}\t | Test Accuracy      : {test_acc:2.4f}"
                     f" | Test Top2 Accuracy: {test_top2_acc:2.4f}")
        if test_metrics.get("macro_f1") is not None:
            logger.debug(f"Test Macro-F1  : {test_metrics['macro_f1']:.4f}\t | "
                         f"Test Macro-Prec : {test_metrics['macro_precision']:.4f}\t | "
                         f"Test Macro-Recall: {test_metrics['macro_recall']:.4f}")
    logger.debug("\nTraining is done.")


def log_training_results(logger, epoch, train_loss, train_acc, train_top2_acc, valid_loss, valid_acc, valid_top2_acc,
                         training_mode=None, train_metrics=None, valid_metrics=None):
    logger.debug(f"\nEpoch : {epoch}\n"
                 f"Train Loss     : {float(train_loss):2.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\t | \tTrain Top2 Accuracy : {train_top2_acc:2.4f}\n"
                 f"Valid Loss     : {float(valid_loss):2.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}\t | \tValid Top2 Accuracy : {valid_top2_acc:2.4f}")
    if training_mode not in ["self_supervised", "SupCon", None]:
        if train_metrics and train_metrics.get("macro_f1") is not None:
            logger.debug(f"Train Macro-F1 : {train_metrics['macro_f1']:.4f}\t | \t"
                         f"Train Macro-Prec : {train_metrics['macro_precision']:.4f}\t | \t"
                         f"Train Macro-Recall: {train_metrics['macro_recall']:.4f}")
        if valid_metrics and valid_metrics.get("macro_f1") is not None:
            logger.debug(f"Valid Macro-F1 : {valid_metrics['macro_f1']:.4f}\t | \t"
                         f"Valid Macro-Prec : {valid_metrics['macro_precision']:.4f}\t | \t"
                         f"Valid Macro-Recall: {valid_metrics['macro_recall']:.4f}")


def model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, config,
                device, training_mode, logger):
    total_loss = []
    number_per_batch = []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    class_predicted = defaultdict(int)
    class_top2_correct = defaultdict(int)
    all_true = []
    all_pred = []
    model.train()
    temporal_contr_model.train()

    for data, labels, aug1, aug2, _ in train_loader:
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
        number_per_batch.append(len(data))
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        if training_mode in ["self_supervised", "SupCon"]:
            _, features1 = model(aug1)
            _, features2 = model(aug2)
            features1 = F.normalize(features1.view(features1.size(0), -1), dim=1)
            features2 = F.normalize(features2.view(features2.size(0), -1), dim=1)
            loss = SupConLoss(device)(torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1), labels)
        elif "supervised" in training_mode or "ft" in training_mode or "train_linear_" in training_mode:
            predictions, _ = model(data)
            loss = criterion(predictions, labels)
            preds = predictions.detach().argmax(dim=1)
            top2_preds = predictions.detach().topk(min(2, predictions.shape[1]), dim=1).indices
            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())
            for label, pred, top2_pred in zip(labels, preds, top2_preds):
                label_id = label.item()
                pred_id = pred.item()
                class_total[label_id] += 1
                class_correct[label_id] += int(label_id == pred_id)
                class_predicted[pred_id] += 1
                class_top2_correct[label_id] += int(label_id in top2_pred.cpu().numpy())
        else:
            predictions, _ = model(data)
            loss = criterion(predictions, labels)

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()

    total_loss_tensor = torch.tensor(total_loss)
    number_tensor = torch.tensor(number_per_batch)
    avg_loss = (total_loss_tensor * number_tensor).sum() / number_tensor.sum()
    if training_mode in ["self_supervised", "SupCon"]:
        return avg_loss, 0, 0, {}, {}, {}, {}

    total = sum(class_total.values())
    total_acc = sum(class_correct.values()) / total if total else 0.0
    total_top2_acc = sum(class_top2_correct.values()) / total if total else 0.0
    class_acc = {label: class_correct[label] / class_total[label] for label in sorted(class_total) if class_total[label] > 0}
    train_metrics = _classification_metrics(all_true, all_pred, config, logger, "[Train]")
    return avg_loss, total_acc, total_top2_acc, class_acc, class_total, class_predicted, train_metrics


def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode, config, logger, data_type, seed):
    model.eval()
    temporal_contr_model.eval()
    criterion = nn.CrossEntropyLoss() if config.loss_type == "CE" else GCELoss(q=config.GCE_q)
    total_loss_list = []
    id_sample_counts = []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    class_top2_correct = defaultdict(int)
    all_true = []
    all_pred = []

    with torch.no_grad():
        for data, labels, aug1, _, _ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)
            aug1 = aug1.float().to(device)
            predictions, _ = model(aug1 if training_mode == "supervised_after_cgan" else data)
            if training_mode in ["self_supervised", "SupCon"]:
                continue
            id_mask = labels < config.num_classes
            if id_mask.sum() > 0:
                loss = criterion(predictions[id_mask], labels[id_mask])
                count = id_mask.sum().item()
                total_loss_list.append(loss.item() * count)
                id_sample_counts.append(count)
            probs = torch.softmax(predictions, dim=1)
            raw_preds = predictions.detach().argmax(dim=1)
            for index in range(len(labels)):
                cur_label = labels[index].item()
                max_prob = probs[index].max().item()
                final_pred = config.num_classes if max_prob < config.ood_threshold else raw_preds[index].item()
                all_true.append(cur_label)
                all_pred.append(final_pred)
                if cur_label < config.num_classes:
                    class_total[cur_label] += 1
                    class_correct[cur_label] += int(final_pred == cur_label)
                    top2 = predictions[index].detach().topk(min(2, predictions.shape[1]), dim=0).indices.cpu().tolist()
                    class_top2_correct[cur_label] += int(cur_label in top2)

    if training_mode in ["self_supervised", "SupCon"]:
        return 0, 0, 0, {}
    avg_loss = torch.tensor(total_loss_list).sum() / torch.tensor(id_sample_counts, dtype=torch.float).sum() if id_sample_counts else 0.0
    total = sum(class_total.values())
    overall_acc = sum(class_correct.values()) / total if total else 0.0
    overall_top2_acc = sum(class_top2_correct.values()) / total if total else 0.0
    metrics_dict = _classification_metrics(all_true, all_pred, config, logger, data_type)
    return avg_loss, overall_acc, overall_top2_acc, metrics_dict


def _classification_metrics(all_true, all_pred, config, logger, prefix):
    metrics_dict = {}
    if not all_true:
        return metrics_dict
    label_names = [
        "Uplink Interference", "Uplink Weak Coverage", "Downlink Interference",
        "Downlink Weak Coverage", "Traffic Channel Overload", "Control Channel Overload",
    ]
    labels_range = list(range(config.num_classes))
    id_pairs = [(true, pred if pred < config.num_classes else -1) for true, pred in zip(all_true, all_pred) if true < config.num_classes]
    if not id_pairs:
        return metrics_dict
    id_true, id_pred = zip(*id_pairs)
    cm = confusion_matrix(id_true, id_pred, labels=labels_range)
    logger.debug(f"{prefix} Confusion Matrix:\n{cm}")
    report_str = classification_report(id_true, id_pred, labels=labels_range,
                                       target_names=label_names[:config.num_classes], digits=4, zero_division=0)
    report_dict = classification_report(id_true, id_pred, labels=labels_range,
                                        target_names=label_names[:config.num_classes], digits=4,
                                        output_dict=True, zero_division=0)
    logger.debug(f"\n{prefix} Classification Report:\n{report_str}")
    metrics_dict["macro_f1"] = report_dict.get("macro avg", {}).get("f1-score", 0.0)
    metrics_dict["macro_precision"] = report_dict.get("macro avg", {}).get("precision", 0.0)
    metrics_dict["macro_recall"] = report_dict.get("macro avg", {}).get("recall", 0.0)
    metrics_dict["weighted_f1"] = report_dict.get("weighted avg", {}).get("f1-score", 0.0)
    metrics_dict["confusion_matrix"] = cm.tolist()
    return metrics_dict


def gen_pseudo_labels(model, dataloader, device, experiment_log_dir, training_mode):
    model.eval()
    softmax = nn.Softmax(dim=1)
    all_pseudo_labels = []
    all_data = []
    with torch.no_grad():
        for data, _, _, _, _ in dataloader:
            data = data.float().to(device)
            predictions, _ = model(data)
            pseudo_labels = softmax(predictions).max(1, keepdim=True)[1].squeeze()
            all_pseudo_labels.append(pseudo_labels.cpu())
            all_data.append(data.cpu())
    data_save = {
        "samples": torch.cat(all_data, dim=0),
        "labels": torch.cat(all_pseudo_labels, dim=0).long(),
    }
    file_name = "train.pt" if training_mode == "gen_pseudo_labels_by_super" else "pseudo_train_data.pt"
    torch.save(data_save, os.path.join(experiment_log_dir, file_name))
    print("Pseudo labels generated.")
