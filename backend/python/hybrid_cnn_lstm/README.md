# Hybrid CNN–LSTM Feature-Fusion Module (Layer 4.5)

Implements the central contribution of the manuscript *"Hybrid CNN–LSTM Edge AIoT
System with State-of-the-Art Model Benchmarking for Intelligent Rice Pest
Detection and Microclimate Prediction"* (BEEI): a **multimodal model with an
adaptive feature-fusion mechanism** that unifies the previously separate
EfficientNet-B0 (spatial) and LSTM (temporal) branches.

```
image 224×224×3   ─► EfficientNet-B0 ─► GAP ─► Dense(n,ReLU) = R_visual
sensors T=6 × 8   ─► LSTM64→BN→LSTM32→BN→Drop.2 ─► Dense(n,ReLU) = R_environment
    ( 8 = [temperature, humidity, ph, light_intensity, vpd] + hour_sin + hour_cos + temp·humidity )
R_total = w₁·R_visual + w₂·R_environment          (Eq. 1,  w₁+w₂ = 1)
R_total ─► Dropout(0.2) ─► softmax  → pest class (K=10)      (Eq. 3)
        ─► Dropout(0.2) ─► softmax  → plant health (3)
        ─► Dropout(0.2) ─► linear   → microclimate / VPD (5)
```

`w₁ = 0.6`, `w₂ = 0.4` are **fixed hyper-parameters** selected by grid search +
5-fold cross-validation (`tune_fusion_weights.py`), matching the manuscript.

## Manuscript cross-reference

| Manuscript | File / symbol |
|---|---|
| Eq. 1 – adaptive feature fusion | `fusion_model.AdaptiveFeatureFusion` |
| Eq. 3 – soft-max decision head | `fusion_model.build_hybrid_model` (`pest`, `health` heads) |
| Eq. 4 – LSTM cell | `fusion_model.build_lstm_branch` |
| sec. 3.3 step 5 – cyclical time + `temp × humidity` feature engineering | `data_pipeline.engineer_sensor_window` |
| Eq. 9 – 95% confidence interval `μ ± 1.96·σ/√k` | `data_pipeline.ci95` (reported by every CV script) |
| sec. 3.3 – grid-search + 5-fold CV for w₁/w₂ | `tune_fusion_weights.py` |
| Table 2 – Algorithm 1 (17 steps) | `algorithm1.py` |
| sec. 3.8 / Eq. 13 – actuation success rate | `actuation_eval.py` |
| Table 6 – baseline vs hybrid comparison | `evaluate_hybrid.py` |
| Training config (Table 3): Adam 1e-3, batch 32, dropout 0.2, seed 42, 5-fold CV | `train_hybrid.py` |

## Paired data

Real synchronised image↔sensor pairs are not shipped with the repo, so
`make_paired_dataset.py` generates a **reproducible synthetic** T=6 micro-climate
window for every rice-leaf image (seed 42) and aligns them on a common timestamp
(|Δt| = 0 ≤ 5 min). Disease state is coupled to atmospheric stress so the fused
model can exploit genuine cross-modal signal — reproducing the Table 6 gain of
fusion over the unimodal baselines. This mirrors how `backend/python/rnn/`
already evaluates the LSTM on a seeded synthetic series.

## Run order

```bash
pip install -r requirements.txt

# from backend/python/  (so "python -m hybrid_cnn_lstm.<x>" resolves the package)
python -m hybrid_cnn_lstm.make_paired_dataset      # -> paired_manifest.csv + *.npy
python -m hybrid_cnn_lstm.tune_fusion_weights      # -> fusion_weights.json  (grid search w₁)
python -m hybrid_cnn_lstm.train_hybrid             # -> hybrid_cnn_lstm_model.h5 + scaler
python -m hybrid_cnn_lstm.evaluate_hybrid          # -> table6_hybrid_comparison.md
python -m hybrid_cnn_lstm.algorithm1 --limit 10    # -> algorithm1_run_log.json (dry-run)
python -m hybrid_cnn_lstm.actuation_eval           # -> actuation_success_rate.json (Eq. 13)

# enable step-16 transmit (Laravel POST /api/inference + MQTT broker):
python -m hybrid_cnn_lstm.algorithm1 \
    --laravel-url https://petaniasik.my.id/api/inference \
    --iot-id jTZids5M --iot-token <shared-secret> \
    --mqtt-host broker.local
```

Step 16 target endpoint (`backend/laravel/`):

| Method / route | Purpose |
|---|---|
| `POST /api/inference` | ingest one fused inference (auth: `iot_id` + `iot_token`); stores `hybrid_inferences`, re-publishes to MQTT `taniverse/{iot_id}/inference` |
| `GET /api/inference/{iot_id}` | latest fused inference (dashboard / edge) |

The Laravel `RecommendationService::getActuation()` mirrors Algorithm 1 step 15, and the dashboard's **"Aksi Aktuasi Otomatis"** card shows the live decision + fused pest/health/VPD.

All scripts accept `--data-dir` / `--out-dir` and use `seed = 42`.
