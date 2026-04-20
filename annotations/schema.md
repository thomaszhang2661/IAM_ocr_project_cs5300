# Annotation Audit Schema

## audit.csv — one row per sample

| Column | Who writes | Values | Description |
|--------|-----------|--------|-------------|
| `idx` | system | int | HuggingFace dataset index |
| `split` | system | train/test/val | Dataset split |
| `ground_truth` | IAM | string | Original IAM annotation |
| `doubao_verdict` | Doubao VLM | CORRECT / INCORRECT / AMBIGUOUS | Model's judgment |
| `doubao_reason` | Doubao VLM | string | e.g. "disc→dise" |
| `doubao_corrected` | Doubao VLM | string | Model's full corrected line (or empty) |
| `human_verdict` | You | confirmed_correct / confirmed_error / skip | Your final judgment |
| `human_corrected` | You | string | Your corrected text (fill only if doubao_corrected is wrong) |
| `final_annotation` | computed | string | What training uses (see logic below) |

## final_annotation logic

```
if human_verdict == "confirmed_correct":
    final_annotation = ground_truth          # original was right
elif human_verdict == "confirmed_error":
    if human_corrected != "":
        final_annotation = human_corrected   # your manual fix
    else:
        final_annotation = doubao_corrected  # accept model suggestion
else:  # skip or not reviewed
    final_annotation = ground_truth          # default: keep original
```

## Training splits

- **original train**: use ground_truth for all training samples
- **clean train**: use final_annotation for all training samples
  - samples where human_verdict = confirmed_error → corrected text
  - all others → ground_truth (unchanged)

Compare CER on test set: original_train vs clean_train → measures annotation noise impact
