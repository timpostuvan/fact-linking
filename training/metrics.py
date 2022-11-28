def calculate_confusion_matrix(predictions, true_labels):
    tp = ((predictions == 1) & (true_labels == 1)).sum().item()
    tn = ((predictions == 0) & (true_labels == 0)).sum().item()
    fp = ((predictions == 1) & (true_labels == 0)).sum().item()
    fn = ((predictions == 0) & (true_labels == 1)).sum().item()
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

def calculate_f1_score(tp, tn, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) \
        if (precision + recall) > 0 else 0.0
    return f1_score, precision, recall
