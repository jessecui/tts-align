# Eval Comparison

_30 held-out prompts. Same reward pipeline as training (Whisper-small WER, UTMOS, ECAPA-TDNN speaker sim, composite). Lower is better for WER and catastrophic-failure; higher is better for everything else._

| method | n | mean WER | mean UTMOS | mean spk_sim | mean composite | catastrophic failure (WER>30%) |
|---|---|---|---|---|---|---|
| base | 30 | 0.068 | 4.36 | 0.580 | 0.871 | 13.3% |
| dpo | 30 | 0.073 | 4.39 | 0.617 | 0.875 | 10.0% |
| kto | 30 | 0.049 | 4.40 | 0.623 | 0.887 | 6.7% |
| grpo | 30 | 0.062 | 4.37 | 0.589 | 0.876 | 13.3% |
