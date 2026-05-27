# Eval Comparison

_30 held-out prompts. Same reward pipeline as training (Whisper-small WER, UTMOS, ECAPA-TDNN speaker sim, composite). Lower is better for WER and catastrophic-failure; higher is better for everything else._

| method | n | mean WER | mean UTMOS | mean spk_sim | mean composite | catastrophic failure (WER>30%) |
|---|---|---|---|---|---|---|
| base | 30 | 0.068 | 4.36 | 0.580 | 0.871 | 13.3% |
| dpo | 30 | 0.061 | 4.38 | 0.586 | 0.876 | 10.0% |
| kto | 30 | 0.057 | 4.38 | 0.619 | 0.882 | 10.0% |
