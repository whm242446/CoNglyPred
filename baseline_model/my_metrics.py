from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix, f1_score, roc_curve, auc, precision_recall_curve

def get_metrics(all_labels, all_preds, probabilities):
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    mcc = matthews_corrcoef(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds)
    
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    specificity = tn / (tn + fp)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"Sensitivity (Recall): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    fpr, tpr, _ = roc_curve(all_labels, probabilities)
    roc_auc = auc(fpr, tpr)
    
    precision_pr, recall_pr, _ = precision_recall_curve(all_labels, probabilities)
    aupr = auc(recall_pr, precision_pr)
    
    print(f"AUC: {roc_auc:.4f}")
    print(f"AUPR: {aupr:.4f}")

    return accuracy, precision, mcc, recall, specificity, f1, roc_auc, aupr, fpr, tpr, recall_pr, precision_pr