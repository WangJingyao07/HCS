def top1_accuracy(logits, targets):
    return (logits.argmax(dim=-1) == targets).float().mean().item()

def exact_match(pred_str, gold_str):
    return int(pred_str.strip().lower() == gold_str.strip().lower())
