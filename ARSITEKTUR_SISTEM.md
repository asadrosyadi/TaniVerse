# Smart Farming IoT System for Rice (*Oryza sativa*) — Full System Architecture, Mathematical Formulation, and Algorithms

> Technical companion document for a Q1 journal manuscript.
> Bahasa: dokumen ini ditulis dalam Bahasa Inggris agar langsung dapat dipakai untuk *manuscript* jurnal Q1. Setiap sub-sistem diberi (i) arsitektur, (ii) formulasi matematis, (iii) algoritma / *pseudocode*.
> Scope: every program currently in this repository is covered — the ESP32 field node (`Arduino/padi_sawah`), the computer-vision disease/pest classifier (`Training_Penyakit Padi/train.py`, `backend/python/detect_penyakit/`), the LSTM micro-climate forecaster (`backend/python/rnn/`), the **hybrid CNN–LSTM feature-fusion model and Algorithm 1** (`backend/python/hybrid_cnn_lstm/`), the Laravel ingestion/API/dashboard tier (`backend/laravel/`), and the rule-based agronomic recommendation engine.
>
> **2026 sync note:** the Raspberry Pi servo–LiDAR pest-deterrent node (`backend/python/servo_lidar.py`) was removed from the repository and is no longer part of the implemented system; Section 4 below is kept only as a historical record and is marked accordingly. Several exploratory scripts that produced figures/tables not matching the manuscript's final numbers (`Training_Penyakit Padi/{compute_real_confusion_matrix,plot_real_confusion_matrix,make_figure4a_augmentation,find_misclassified_example}.py`, `backend/python/rnn/{generate_real_figure9,generate_real_figure9_800,compute_real_table5,compute_real_table5_blocked_cv}.py`, `backend/python/detect_penyakit/{detailed_analysis,generate_confusion_matrix,generate_paper_figures}.py`) were likewise removed; this document has been updated to reference only the scripts that actually reproduce the manuscript's reported numbers. The Table 7 benchmark (`benchmark_modern_models.py`) was also extended with a YOLOv11n-cls row (Section 12.1).

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Notation and Symbol Table](#2-notation-and-symbol-table)
3. [Layer 1 — Perception / Field Sensing Node (ESP32)](#3-layer-1--perception--field-sensing-node-esp32)
4. [Layer 2 — Servo–LiDAR Scanning & Pest-Deterrent Node (removed)](#4-layer-2--servolidar-scanning--pest-deterrent-node-removed)
5. [Layer 3 — Communication, Authentication and Data Model (Laravel API)](#5-layer-3--communication-authentication-and-data-model-laravel-api)
6. [Layer 4 — Computer-Vision Disease & Pest Classification](#6-layer-4--computer-vision-disease--pest-classification)
7. [Layer 5 — LSTM Micro-climate & Plant-Health Forecasting](#7-layer-5--lstm-microclimate--plant-health-forecasting)
8. [Layer 6 — Rule-Based Agronomic Recommendation Engine](#8-layer-6--rule-based-agronomic-recommendation-engine)
9. [End-to-End Data & Control Flow](#9-end-to-end-data--control-flow)
10. [Deployment Topology](#10-deployment-topology)
11. [Reproducibility — Hyperparameter Tables](#11-reproducibility--hyperparameter-tables)
12. [Experimental Results Summary](#12-experimental-results-summary)

---

## 1. System Overview

The platform is a **five-layer** cyber-physical system for continuous, autonomous monitoring and decision support of irrigated rice paddies. (A sixth, Raspberry Pi servo–LiDAR scanning/deterrent node once existed as Layer 2; its source, `backend/python/servo_lidar.py`, has since been removed from the repository — see Section 4 for the historical record. Layer numbers below are kept as originally assigned, with Layer 2 retired rather than renumbering every downstream reference.)

```
                       ┌───────────────────────────────────────────────────────────────┐
                       │                     PRESENTATION (Blade + Chart.js)            │
                       │        Dashboard · Monitoring · History · Data export          │
                       └───────────────▲───────────────────────────▲───────────────────┘
                                       │                           │
     ┌─────────────────────────────────┴──────────┐   ┌────────────┴───────────────────┐
     │  LAYER 6  Rule-Based Recommendation Engine  │   │ LAYER 5  LSTM Forecaster (Flask│
     │  (VPD, agronomic thresholds, disease→action)│   │  :5000)  micro-climate + health│
     └─────────────────────────────────▲──────────┘   └────────────▲───────────────────┘
                                       │                           │ pull history
                       ┌───────────────┴───────────────────────────┴───────────────────┐
                       │        LAYER 3  Laravel API + MySQL  (petaniasik.my.id)        │
                       │  POST /api/kirim_sensor · GET /api/sensor · GET /api/bacajson  │
                       └───▲───────────────────────────────────────▲───────────────────┘
                           │ HTTPS/JSON                            │ HTTPS/JSON
            ┌──────────────┴───────────┐                ┌──────────┴──────────────┐
            │ LAYER 1  ESP32 field node │                │ LAYER 4  CV classifier  │
            │ 17 sensors, RS485/Modbus, │                │ (Flask :7000) Haar +    │
            │ relays, WiFiManager       │                │ EfficientNet-B0 (CNN)   │
            └───────────────────────────┘                └─────────────────────────┘
```

| Layer | Program(s) | Runtime | Core function |
|---|---|---|---|
| 1. Perception / Actuation | `Arduino/padi_sawah/padi_sawah.ino` | ESP32 (Arduino C++) | Acquire 17 agro-meteo variables + 15-min telemetry; drive UV / ultrasonic / irrigation relays from the fused model (MQTT + REST) |
| 2. Scanning/deterrent *(removed)* | ~~`backend/python/servo_lidar.py`~~ | — | No longer present in the repository; see Section 4 |
| 3. Ingestion/API | `backend/laravel/**` | PHP 8.2 / Laravel 12 + MySQL | Token auth, persistence, REST endpoints (`/api/inference`), MQTT publish, dashboard, Excel export |
| 4. Vision | `Training_Penyakit Padi/{train,compare_cnn_backbones}.py`, `backend/python/detect_penyakit/app.py`, `benchmark_modern_models.py` | Python / TensorFlow-Keras + PyTorch | 10-class rice disease/pest classification from leaf images (Table 4 & Table 7 benchmarks) |
| 4.5 Fusion | `backend/python/hybrid_cnn_lstm/**` | Python / TensorFlow-Keras | Adaptive CNN+LSTM feature fusion (Eq. 1), unified pest + health + VPD heads, Algorithm 1 |
| 5. Forecasting | `backend/python/rnn/**`, `evaluate_lstm_microclimate_10k.py` | Python / Flask + TensorFlow-Keras | Multi-step micro-climate regression (Table 5) + 3-class plant-health forecast |
| 6. Recommendation | `app/Http/Controllers/RecommendationService.php` | PHP | Deterministic agronomic advisory from sensor + disease state |

Design principles: (a) **edge pre-processing** — noisy analog channels are filtered on the microcontroller; (b) **thin cloud** — the server only authenticates, stores, and serves; (c) **model services are stateless** and pull their inputs from the same canonical REST endpoint; (d) **graceful degradation** — the forecaster falls back from LSTM → classical model → zero-filled template, and the CV pipeline is time-gated and self-pruning.

---

## 2. Notation and Symbol Table

| Symbol | Meaning | Unit |
|---|---|---|
| $t$ | discrete time index (sample) | – |
| $x_t \in \mathbb{R}^{d}$ | feature vector at time $t$ | – |
| $T$ | look-back window (sequence length), $T=6$ | samples |
| $\Delta$ | sampling interval; field node $\Delta=900\,\mathrm{s}$, forecaster $\Delta=4\,\mathrm{h}$ | s / h |
| $H$ | forecast horizon in steps, $H = 6\,d$ for $d$ days | samples |
| $N$ | total number of training samples | – |
| $K$ | number of classes | – |
| $n_c$ | number of training samples in class $c$ | – |
| $\sigma(\cdot)$ | logistic sigmoid, $\sigma(z)=\left(1+e^{-z}\right)^{-1}$ | – |
| $\odot$ | element-wise (Hadamard) product | – |
| $e_s, e_a$ | saturation / actual water-vapour pressure | kPa |
| $\mathrm{VPD}$ | vapour-pressure deficit | kPa |
| $\mathrm{RH}$ | relative humidity | % |
| $T_a, T_s$ | ambient / soil temperature | °C |

---

## 3. Layer 1 — Perception / Field Sensing Node (ESP32)

**Source:** `Arduino/padi_sawah/padi_sawah.ino` (production), `Arduino/kirim_laravel/kirim_laravel.ino` (telemetry stub).

### 3.1 Architecture

A single ESP32 (dual-core Xtensa LX6, 12-bit SAR ADC) hosts:

* **RS-485 / Modbus-RTU bus** (`Serial2`, 4800 Bd, 8N1, DE/RE on GPIO25, channel-select "sekat" on GPIO26) for a 4-in-1 soil probe: N, P, K, pH, volumetric water content, soil temperature.
* **Digital / I²C weather cluster:** DHT22 (air $T$/RH), BH1750 (illuminance), BMP280 (barometric pressure, SPI), MQ-135 (gas/pheromone-trap proxy, ADC GPIO39).
* **Pulse sensors on hardware interrupts:** cup anemometer (GPIO4, `RISING`), tipping-bucket rain gauge (GPIO18, `FALLING`).
* **Analog conditioning:** TDS/EC probe (GPIO36) with 30-tap circular buffer; battery gauge via resistive divider (GPIO34, $R_1=82\,\mathrm{k\Omega}$, $R_2=10\,\mathrm{k\Omega}$).
* **Actuation:** three active-low relays mapped to the manuscript actuators (sec. 3.6 / Algorithm 1 step 15) — GPIO19 = **UV lamp**, GPIO23 = **ultrasonic pest repeller**, GPIO27 = **irrigation valve** — plus a Wi-Fi-provisioning portal (`WiFiManager`). Actuation commands arrive from the fused CNN–LSTM model over **MQTT** (`taniverse/{iot_id}/inference`, `PubSubClient`) with a **REST fallback** (`GET /api/inference/{iot_id}` every 15 s); local rules (BH1750 `< 100 lx` → UV, LiDAR motion → ultrasonic) are OR-composed on top.

Main loop period: `delay(900000)` ⇒ one acquisition + upload cycle every **15 min**.

### 3.2 Mathematical formulation per channel

**(a) Modbus-RTU register decoding.** Each request is an 8-byte frame
`[addr, 0x03, reg_hi, reg_lo, qty_hi, qty_lo, crc_lo, crc_hi]` with the CRC-16/Modbus checksum

$$
\mathrm{CRC} = \bigoplus_{i} \text{poly-fold}(b_i), \qquad
\text{poly} = \texttt{0xA001}\;(x^{16}+x^{15}+x^{2}+1).
$$

CRC-16/Modbus reference algorithm:

```
function CRC16(bytes):
    crc ← 0xFFFF
    for each byte b in bytes:
        crc ← crc XOR b
        repeat 8 times:
            if (crc AND 1) == 1: crc ← (crc >> 1) XOR 0xA001
            else:                crc ← (crc >> 1)
    return crc            # low byte transmitted first
```

The device returns big-endian 16-bit registers reconstructed as
$R = 256\,b_{hi} + b_{lo}$ and rescaled:

$$
\begin{aligned}
\text{N,P,K}\;[\mathrm{mg\,kg^{-1}}] &= R_{\{N,P,K\}} & (\text{registers } \texttt{0x1E–0x20}),\\[2pt]
\mathrm{pH} &= R_{pH}/100 & (\text{register }\texttt{0x06}),\\[2pt]
\theta_{soil}\;[\%] &= R_{\theta}/10 & (\text{register }\texttt{0x12}),\\[2pt]
T_s\;[^{\circ}\mathrm{C}] &= R_{T_s}/10 & (\text{register }\texttt{0x13}).
\end{aligned}
$$

**(b) Cup anemometer.** Interrupts increment `rpmcount`; a 5 ms software debounce rejects contact bounce ($\lvert t-t_{last}\rvert \ge 5000\,\mu s$). Over a gate $\tau = 10\,\mathrm{s}$ the rotation rate is
$$
f_{rot} = \frac{\text{rpmcount}}{\tau}\quad[\mathrm{rev\,s^{-1}}].
$$
A second-order field calibration (cup anemometers are non-linear at low wind because of bearing friction and cup-drag) maps rotation to wind speed:
$$
\boxed{\,v = -0.0181\,f_{rot}^{2} + 1.3859\,f_{rot} + 1.4055\quad[\mathrm{m\,s^{-1}}]\,}
$$
with a dead-band $v \leftarrow 0$ if $v \le 1.5\,\mathrm{m\,s^{-1}}$ (below the sensor's stall speed), and $v_{\text{km/h}} = 3.6\,v$.

**(c) Tipping-bucket rain gauge.** Each latching pulse corresponds to a fixed depth $\delta = 0.40\,\mathrm{mm}$ per tip; accumulated rainfall is
$$
P = n_{tip}\,\delta.
$$

**(d) TDS / EC probe.** A 30-sample circular buffer sampled every 40 ms is de-spiked by the **median filter**
$$
\bar v = \operatorname{median}\big(\{v_i\}_{i=1}^{30}\big)\cdot \frac{V_{ref}}{4096}, \qquad V_{ref}=3.3\,\mathrm{V}.
$$
Temperature compensation to 25 °C reference:
$$
k(T_s) = 1 + 0.02\,(T_s - 25), \qquad V_c = \bar v / k(T_s).
$$
The manufacturer cubic transfer function then gives total dissolved solids and electrical conductivity:
$$
\mathrm{TDS} = \tfrac{1}{2}\left(133.42\,V_c^{3} - 255.86\,V_c^{2} + 857.39\,V_c\right)\ [\mathrm{ppm}],\qquad
\mathrm{EC} = \frac{\mathrm{TDS}}{0.65}\ [\mu\mathrm{S\,cm^{-1}}].
$$

**(e) Battery state-of-charge.** Divider back-calculation for a 4S Li-ion pack ($V_{full}=16.8\,\mathrm{V}$):
$$
V_{adc} = \frac{\text{ADC}}{4095}\,V_{ref},\qquad
V_{bat} = V_{adc}\,\frac{R_1+R_2}{R_2},\qquad
\mathrm{SoC} = \frac{V_{bat}}{16.8}\times 100\%.
$$

**(f) Wi-Fi link quality.** Piece-wise linear map from RSSI (dBm) to percentage:
$$
Q(\rho)=
\begin{cases}
0, & \rho \le -100\\
2(\rho + 80), & -100 < \rho < -30\\
100, & \rho \ge -30.
\end{cases}
$$

**(g) Illuminance-driven actuation.** A hysteresis-free threshold on BH1750 lux $L$ raises the local UV-lamp request:
$$
\text{uv}_{\text{night}} = \mathbb{1}[\,L < 100\ \mathrm{lx}\,].
$$

**(h) Remote actuation command.** Two sources, OR-composed with the local rules:
* *Intruder loop:* `GET /api/bacajson/{iot_id}` → `data[0].movement_detected == "ON"` raises the local ultrasonic request (closes the loop with Layer 2).
* *Fused-model loop:* MQTT topic `taniverse/{iot_id}/inference` (or `GET /api/inference/{iot_id}`) carries the Algorithm 1 step-15 `action`:
$$
(\text{uv},\text{ultra},\text{irrig}) =
\begin{cases}
(1,1,0), & \texttt{activate\_ultrasonic\_repeller\_and\_uv\_lamp}\\
(0,0,1), & \texttt{irrigation\_notification}\\
(0,0,0), & \texttt{recommend\_selective\_insecticide\_spraying}\ (\text{advisory})\ /\ \texttt{continue\_monitoring}.
\end{cases}
$$
Final relay state: $\text{relay}_{\text{UV}} = \text{uv}\vee\text{uv}_{\text{night}}$, $\text{relay}_{\text{ultra}} = \text{ultra}\vee\text{movement}$, $\text{relay}_{\text{irrig}} = \text{irrig}$.

### 3.3 Field-node algorithm (pseudocode)

```
setup():
    ADC resolution ← 12-bit
    Serial2.begin(4800, 8N1, RX=16, TX=17); pinMode(DE_RE, OUT)
    WiFiManager.autoConnect("Sensor_Petani_Asik")           # captive portal fallback
    attachInterrupt(GPIO4,  isr_anemometer, RISING)
    attachInterrupt(GPIO18, isr_raingauge,  FALLING)
    BH1750.begin(); BMP280.begin(NORMAL, x2, x16, filter=x16, standby=500ms)
    pinMode(GPIO19|23|27, OUT); all relays ← OFF
    mqtt.setServer(MQTT_HOST, 1883); mqtt.setCallback(on_inference); mqtt.setBufferSize(768)

loop():                                                     # sensor cadence 900 s, actuation responsive
    read_sensors(); poll_lidar(); poll_inference_rest()
    t0 ← millis()
    while millis()-t0 < 900_000:                            # non-blocking wait
        if WiFi.connected:
            if not mqtt.connected: mqtt.reconnect() every 5 s   # subscribe taniverse/{id}/inference
            mqtt.loop()
            every 15 s: poll_inference_rest(); poll_lidar()
        apply_actuators()                                   # OR-compose model cmd + local rules
        delay(2000)

read_sensors():
    for (reg, scale, nbytes) in MODBUS_PLAN:                # NPK, pH, θ, Ts
        select_channel(); enableTransmit()
        Serial2.write(frame(reg)); Serial2.flush(); enableReceive()
        wait 500 ms
        if Serial2.available() ≥ nbytes:
            resp ← readBytes(nbytes)
            R ← (resp[3] << 8) | resp[4]                    # + resp[5..8] for NPK
            value ← R / scale
    ρ ← WiFi.RSSI();  Q ← signalQuality(ρ)
    SoC ← battery_soc(analogRead(34))
    (Ta, RH) ← DHT22.read()
    # ---- wind ----
    if millis()-t_gate ≥ τ:
        detachInterrupt(GPIO4)
        f_rot ← rpmcount / τ
        v ← -0.0181*f_rot^2 + 1.3859*f_rot + 1.4055
        if v ≤ 1.5: v ← 0
        rpmcount ← 0; t_gate ← millis(); attachInterrupt(GPIO4, …)
    # ---- rain ----
    P ← n_tip * 0.40
    # ---- light + actuation ----
    L ← BH1750.readLightLevel()
    uv_night ← (L < 100);  apply_actuators()
    p ← BMP280.readPressure();  gas ← analogRead(39)
    # ---- TDS/EC (median + temp-comp + cubic) ----
    every 40 ms: buf[idx++ mod 30] ← analogRead(36)
    every 800 ms:
        v̄ ← median(buf) * 3.3/4096
        Vc ← v̄ / (1 + 0.02*(Ts-25))
        TDS ← (133.42*Vc^3 - 255.86*Vc^2 + 857.39*Vc) * 0.5
        EC  ← TDS / 0.65
    # ---- upload ----
    payload ← JSON{iot_id, iot_token, temperature=Ta, humidity=RH, windspeed=v,
                   rainfall=P, light_intensity=L, ph, soil_moisture=θ, ec, tds,
                   soil_temp=Ts, pressure=p, feromon=gas, battery_level=SoC,
                   signal_strength=Q, Nitrogen_Level, Phosphorus_Level, Potassium_Level}
    HTTP POST serverName ← payload

poll_lidar():                                        # intruder loop (Layer 2)
    r ← HTTP GET https://…/api/bacajson/{iot_id}
    movement ← (r.ok and r.data[0].movement_detected == "ON")

poll_inference_rest() / on_inference(mqtt_msg):      # fused-model loop
    action ← parse(...).actuation.action | data.actuation_action
    (cmd_uv, cmd_ultra, cmd_irrig) ← map(action)     # Algorithm 1 step 15
    apply_actuators()

apply_actuators():                                   # active-low, OR-composed
    digitalWrite(GPIO19, (cmd_uv    or uv_night) ? ON : OFF)   # UV lamp
    digitalWrite(GPIO23, (cmd_ultra or movement) ? ON : OFF)   # ultrasonic repeller
    digitalWrite(GPIO27,  cmd_irrig              ? ON : OFF)   # irrigation valve
```

---

## 4. Layer 2 — Servo–LiDAR Scanning & Pest-Deterrent Node (removed)

**Historical record only — not part of the current codebase.** This layer previously ran on a Raspberry Pi (`backend/python/servo_lidar.py`, plus a `servo_lidar - Copy.py`): a continuous-rotation servo swept a Benewake TF-Luna 1-D LiDAR, range readings were differenced in time to flag motion ($\lvert r_t-r_{t-1}\rvert>0.2\,\mathrm m$), and the result was written to a `lidars` table that the ESP32 (Layer 1) polled via `GET /api/bacajson/{iot_id}` to drive the ultrasonic-repeller relay as a secondary, non-model trigger.

Both `.py` sources have since been deleted from the repository, so this subsystem is no longer implemented. The current manuscript's Section 3.1 camera+LiDAR mention refers instead to depth-map input for the CV pest classifier (Layer 4), not to this scanning/deterrent node. The `lidars` table and `/api/bacajson` route may still exist in the Laravel schema/routes as unused legacy surface — verify against `backend/laravel/` directly before relying on them; this document does not assert their current status either way.

---

## 5. Layer 3 — Communication, Authentication and Data Model (Laravel API)

**Source:** `backend/laravel/routes/api.php`, `app/Http/Controllers/SensorController.php`, `Modules/Dashboard/**`, `Modules/{Data,History,Yolo}/**`.

### 5.1 REST surface

| Method / route | Controller | Consumer | Purpose |
|---|---|---|---|
| `POST /api/kirim_sensor` | `SensorController@store` | ESP32 field node | Ingest one telemetry record |
| `GET /api/bacajson/{iot_id}` | `SensorController@getDataLidar` | ESP32 field node | Return latest LiDAR motion state (`ON`/`OFF`) for actuation |
| `GET /api/sensor?iot_id=…` | `SensorController@getByiot_id` | Flask LSTM service | Full descending history for sequence building |
| `POST /api/inference` | `HybridInferenceController@store` | hybrid edge node (`algorithm1.py`) | Ingest one fused CNN–LSTM inference (Algorithm 1 step 16); persists `hybrid_inferences`, re-publishes to MQTT |
| `GET /api/inference/{iot_id}` | `HybridInferenceController@latest` | dashboard / edge | Latest fused inference (pest + health + VPD + actuation) |
| `GET /user` | closure (`auth:sanctum`) | dashboard | Authenticated identity |
| dashboard/data/history/yolo web routes | module controllers | browser | Views, pagination, Excel/ZIP export |

### 5.2 Authentication

Device authentication is a shared-secret pair check, not session auth:
$$
\text{authorised} \iff \exists\,u \in \text{users}:\; u.\texttt{iot\_id}=\texttt{iot\_id} \ \wedge\ u.\texttt{iot\_token}=\texttt{iot\_token}.
$$
On failure the endpoint returns `401`. Input is validated (`nullable|numeric|integer`) before persistence.

### 5.3 Persistence schema (MySQL, `padi_sawah`)

```
sensor_iots(  id, iot_id, temperature, humidity, windspeed, rainfall,
              light_intensity, ph, soil_moisture, ec, tds, soil_temp,
              pressure, feromon, battery_level, signal_strength,
              Nitrogen_Level, Phosphorus_Level, Potassium_Level, timestamps )

sensor_kameras( id, iot_id, penyakit, probabilitas, image, timestamps )   # CV output

lidars(       id, iot_id, distance, movement_detected, servo_position, timestamps )

hybrid_inferences( id, iot_id, image, pest_label, pest_class, pest_confidence,   # fused CNN–LSTM output
                   health_status, health_class, health_confidence, vpd_kpa,
                   microclimate(json), fusion_w1, fusion_w2,
                   actuation_action, actuators(json), transmitted_via, timestamps )

historyiots / history_yolos : long-term archives feeding the History module
```

### 5.4 Bounded-storage (ring-buffer) policy

To keep the field database O(1) in disk, each writer enforces a hard cap and a full truncate:

* `sensor_iots`: if $\lvert \text{rows}\rvert \ge 10^{5}$ → `DELETE` all + `ALTER TABLE … AUTO_INCREMENT = 1`.
* `sensor_kameras`: if $\lvert \text{rows}\rvert \ge 10^{3}$ → truncate table **and** wipe the image directory.
* `hybrid_inferences`: if $\lvert \text{rows}\rvert \ge 10^{4}$ → `DELETE` all + reset `AUTO_INCREMENT`.

```
ingest(request):
    v ← validate(request, schema)
    u ← users.where(iot_id=v.iot_id, iot_token=v.iot_token).first()
    if u is null: return 401
    SensorIot.create(v \ {iot_token})
    if count(sensor_iots) ≥ 100_000:
        truncate sensor_iots; reset AUTO_INCREMENT
    return 201
```

### 5.5 Dashboard orchestration

`DashboardController@index` performs a scatter–gather:

```
iot_id ← auth().user().iot_id
F14 ← HTTP GET  {API_RNN}14         # 14-day forecast  (Flask :5000 /forecast/{id}?days=14)
F1  ← HTTP GET  {API_RNN}1          # 1-day forecast → "current" snapshot
S   ← SensorIot.where(iot_id).latest(1)
C   ← SensorKamera.where(iot_id).latest(1)
R   ← RecommendationService.getRecommendations(S, C)                       # Layer 6 advisory list
A   ← RecommendationService.getActuation(S, C, F1.predicted_health)       # Layer 6 — Algorithm 1 step 15
H   ← HybridInference.where(iot_id).latest(1)                             # fused CNN–LSTM output (if any)
paginate(F14.forecast, perPage=50)
render dashboard::index with (F14 chart, R, A, H, S, pagination)
```

The dashboard's **"Aksi Aktuasi Otomatis"** card renders `A` (server-side, then refreshed by the `/getRecommendations` poll) alongside `H`'s pest / health / VPD / fusion-weight summary.

If the Flask service is unreachable or the `iot_id` mismatches, a zero-filled forecast template is substituted (degradation path).

---

## 6. Layer 4 — Computer-Vision Disease & Pest Classification

**Sources:** training — `Training_Penyakit Padi/train.py`; inference service — `backend/python/detect_penyakit/app.py`; benchmark — `backend/python/detect_penyakit/benchmark_modern_models.py` (Table 7); descriptive Figure 8 confusion matrix — `Training_Penyakit Padi/{compute_real_50perclass_matrix,plot_real_50perclass_confusion_matrix}.py`.

### 6.1 Task

10-way single-label image classification of rice leaf/panicle condition:

```
0 bacterial_leaf_blight   1 bacterial_leaf_streak   2 bacterial_panicle_blight
3 blast                   4 brown_spot             5 dead_heart
6 downy_mildew            7 hispa                  8 normal            9 tungro
```
Dataset layout: `train/ val/ test/` with one sub-directory per class (`torchvision.ImageFolder` / `keras.image_dataset_from_directory`).

### 6.2 Backbone — EfficientNet-B0 and compound scaling

EfficientNet-B0 is the MobileNet-V2-style inverted-residual (MBConv) network obtained by neural-architecture search, then scaled by the **compound coefficient** $\phi$:
$$
\text{depth } d=\alpha^{\phi},\quad
\text{width } w=\beta^{\phi},\quad
\text{resolution } r=\gamma^{\phi},\qquad
\text{s.t. } \alpha\cdot\beta^{2}\cdot\gamma^{2}\approx 2,\; \alpha,\beta,\gamma\ge 1 .
$$
B0 is the $\phi=0$ baseline ($\alpha=1.2,\ \beta=1.1,\ \gamma=1.15$), input $224\times224\times3$, ~5.3 M parameters.

A **depthwise-separable convolution** (the core of MBConv) factorises a standard conv of cost
$H\cdot W\cdot C_{in}\cdot C_{out}\cdot k^{2}$
into a depthwise part $H\cdot W\cdot C_{in}\cdot k^{2}$ plus a pointwise part $H\cdot W\cdot C_{in}\cdot C_{out}$, a reduction of
$$
\frac{1}{C_{out}} + \frac{1}{k^{2}} .
$$
Each MBConv block also carries a **squeeze-and-excitation** channel-attention gate
$\mathbf{s} = \sigma\!\big(W_2\,\delta(W_1\,\text{GAP}(\mathbf{u}))\big),\ \tilde{\mathbf u}=\mathbf s\odot\mathbf u$
and the **Swish** activation $\text{swish}(z)=z\,\sigma(z)$.

### 6.3 Transfer-learning head and two-stage optimisation

Classifier head appended to the frozen convolutional base:
$$
\mathbf h = \text{Dropout}_{0.2}\!\big(\text{GAP}(f_{\text{B0}}(\mathbf x))\big)
\;\to\; \mathbf g = \text{Dropout}_{0.2}\!\big(\text{ReLU}(W_1\mathbf h + b_1)\big),\ W_1\in\mathbb R^{128\times1280}
$$
$$
\hat{\mathbf y} = \operatorname{softmax}(W_2\mathbf g + b_2),\qquad
\hat y_c = \frac{e^{z_c}}{\sum_{j=1}^{K} e^{z_j}} .
$$

**Loss** — class-weighted categorical cross-entropy:
$$
\mathcal L = -\sum_{c=1}^{K} w_c\,y_c \log \hat y_c,
\qquad
\boxed{\,w_c = \dfrac{N}{K\,n_c}\,}
$$
which up-weights rare classes so that $\sum_c w_c n_c = N$.

**On-the-fly augmentation** $\mathcal A(\cdot)$ (Keras layers, manuscript sec. 3.3): random horizontal **and vertical** flip, random rotation $\pm25^{\circ}$ ($=25/360\times2\pi$), random crop with scale factor $0.8$–$1.0$ (`RandomZoom`), additive Gaussian noise with variance $0.01$ on the $[0,1]$ pixel scale.

Two-phase schedule (**50 epochs total**, manuscript Table 3; dropout $0.2$ in both phases):

| Phase | Trainable | Optimiser | LR | Epochs |
|---|---|---|---|---|
| 1 — head warm-up | base frozen | Adam | $10^{-3}$ | 40 |
| 2 — fine-tune | top 30 layers of B0 unfrozen | Adam | $10^{-5}$ | 10 |

**Validation protocol** (manuscript sec. 3.7): a stratified $70/15/15$ split for the headline model *and* a stratified **5-fold cross-validation** whose results are reported as mean $\pm$ standard deviation; fixed seed $42$. For the 5-fold mode specifically, all 10,407 images (train+val+test pooled) are repartitioned per fold into **≈65% train / ≈15% val / ≈20% test** (≈6,765 / 1,561 / 2,081 images; exact counts vary slightly with stratification/rounding) — this is the split that feeds **Table 4** and **Table 6**, not the CNN benchmark's fixed 8,323/2,084 split (Table 7, §6.5) or the hybrid model's final-deployment **85/15** train/val split (no test fold — Table 3). Both the 70/15/15 and 5-fold modes live in `Training_Penyakit Padi/train.py` (`--cv-folds 5`). The four-backbone comparison of **Table 4** (ResNet50 / MobileNetV2 / VGG16 / EfficientNet-B0 under one identical Keras transfer-learning protocol) is produced by `Training_Penyakit Padi/compare_cnn_backbones.py`.

**Adam** update (per parameter $\theta$, gradient $g_t$):
$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^{2},
$$
$$
\hat m_t=\frac{m_t}{1-\beta_1^{t}},\quad
\hat v_t=\frac{v_t}{1-\beta_2^{t}},\quad
\theta_t=\theta_{t-1}-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}.
$$

**Callbacks:** `ModelCheckpoint` (monitor `val_accuracy`), `EarlyStopping` (patience 6, restore best), `ReduceLROnPlateau` (`val_loss`, factor 0.5, patience 3, floor $10^{-7}$).

```
train_cnn(data_dir, cv_folds=0):
    paths, labels, class_names ← pool train/ + val/ + test/            # stratified re-split
    for (train_idx, test_idx) in split(labels, cv_folds):             # 70/15/15  or  5-fold CV
        w ← { c : N / (K * n_c) for c in class_names }                # class weights
        base ← EfficientNetB0(weights="imagenet", include_top=False); base.trainable ← False
        model ← Input → Augment → base(training=False) → GAP → Drop(.2)
                      → Dense(128,relu) → Drop(.2) → Dense(K,softmax)
        # ---- Phase 1 (40 ep) ----
        model.compile(Adam(1e-3), categorical_crossentropy, [accuracy])
        model.fit(train_ds, val_ds, epochs=40, class_weight=w, callbacks=CB)
        # ---- Phase 2 (10 ep, 50 total) ----
        base.trainable ← True; freeze all but last 30 layers of base
        model.compile(Adam(1e-5), categorical_crossentropy, [accuracy])
        model.fit(train_ds, val_ds, epochs=10, class_weight=w, callbacks=CB)
        evaluate(model, test_idx) → accuracy, P/R/F1 (macro)
    report mean ± std across folds ; save efficientnetb0_model.h5, class_names.json
```

### 6.4 Real-time inference service (`app.py`)

Pipeline per camera frame (Flask MJPEG server, port 7000):

1. **Circadian gate:** process only if $6 \le \text{hour} < 23$.
2. **Leaf ROI localisation** — Haar-cascade `detectMultiScale` on the greyscale frame
   $\;\;Y = 0.299R + 0.587G + 0.114B$,
   parameters `scaleFactor=1.05`, `minNeighbors=7`, `minSize=150²`, `maxSize=800²`.
   The cascade evaluates Haar-like features via the **integral image**
   $\;II(x,y)=\sum_{x'\le x,\,y'\le y} Y(x',y')$
   (any rectangle sum in 4 look-ups) through an AdaBoost stage cascade — a window passes only if every stage $\sum_k \alpha_k h_k(\mathbf x) \ge \theta_{\text{stage}}$.
3. **Classification** — for each ROI: crop → resize $224\times224$ → `img_to_array` → `expand_dims` → `efficientnet.preprocess_input` → `model.predict`.
   $$
   \hat c = \arg\max_c \hat y_c,\qquad p = 100\cdot\max_c \hat y_c\ [\%].
   $$
4. **Persistence** — write `img/hama_padi/{uuid}.jpg`; `INSERT INTO sensor_kameras(iot_id, penyakit, probabilitas, image)`.
5. **Ring-buffer** — if `COUNT(*) ≥ 1000`: truncate table + reset AUTO_INCREMENT + delete all stored crops.
6. **Overlay** — draw bounding box + `"{disease} ({p:.2f}%)"` and stream the annotated frame.

```
process_frame(frame):
    if not (6 ≤ hour(now) < 23): return frame
    gray ← BGR2GRAY(frame)
    for (x,y,w,h) in leafCascade.detectMultiScale(gray, 1.05, 7, minSize=150, maxSize=800):
        roi   ← resize(frame[y:y+h, x:x+w], 224×224)
        z     ← preprocess_input(expand_dims(img_to_array(roi)))
        probs ← model.predict(z)
        c*, p ← argmax(probs), 100*max(probs)
        save roi → {uuid}.jpg
        INSERT sensor_kameras(iot_id='jTZids5M', penyakit=label[c*], probabilitas=p, image=fname)
        if SELECT COUNT(*) FROM sensor_kameras ≥ 1000:
            DELETE FROM sensor_kameras; ALTER TABLE … AUTO_INCREMENT=1; wipe SAVE_DIR
        draw_box_and_text(frame, x,y,w,h, label[c*], p)
    return frame
```

### 6.5 Benchmark protocol (`benchmark_modern_models.py`)

Eight checkpoints are compared under one evaluation protocol (same 2,084-image validation partition, image-level classification metrics): `timm` pretrained models for {EfficientNetV2-S, EfficientNet-B0, ViT-B/16, ResNet-50, MobileNet-V2, VGG-16} and Ultralytics **YOLOv8n-cls** *and* **YOLOv11n-cls** for the YOLO classification family. `run_yolo_experiment()` is generalised (model key / default weights / run-name prefix / display name) so both YOLO variants share one implementation and one per-variant checkpoint-discovery path under `yolo_runs/`; training config (5 epochs, batch 8, imgsz 224, CPU execution, Ultralytics classification augmentation defaults) is identical between the two YOLO rows — see `experiments/yolov11_addition/yolov11_provenance.md` for the full provenance trail (including the CUDA `misaligned address` crash on the first GPU attempt, resolved by falling back to CPU, matching the YOLOv8 baseline's own documented reason for training on CPU).

* **Transforms (train):** `Resize 224²`, `RandomHorizontalFlip(0.5)`, `RandomRotation(10°)`, `ColorJitter(0.15,0.15,0.15)`, `ToTensor`, `Normalize(μ=[.485,.456,.406], σ=[.229,.224,.225])`. Eval: resize + normalise only.
* **Optimiser:** AdamW ($\eta=10^{-4}$, weight-decay $10^{-4}$), decoupled update
  $\theta_t = \theta_{t-1} - \eta(\hat m_t/(\sqrt{\hat v_t}+\varepsilon) + \lambda\theta_{t-1})$.
* **LR schedule:** cosine annealing $\;\eta_t = \tfrac{1}{2}\eta_{\max}\big(1+\cos(\pi t/T_{\max})\big)$.
* **Loss:** softmax cross-entropy. **Selection:** best macro-F1 on `val`, early stop patience 7, ≤ 30 epochs.
* **YOLOv8-cls** training args: `lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=5e-4, warmup_epochs=3, mosaic=1.0, auto_augment=randaugment, erasing=0.4, imgsz=224, batch=16`.

**Metrics.** With confusion matrix $M\in\mathbb N^{K\times K}$ ($M_{ij}$ = true $i$, predicted $j$):
$$
\text{Acc} = \frac{\sum_i M_{ii}}{\sum_{ij}M_{ij}},\quad
P_c=\frac{M_{cc}}{\sum_i M_{ic}},\quad
R_c=\frac{M_{cc}}{\sum_j M_{cj}},\quad
F1_c = \frac{2P_cR_c}{P_c+R_c},
$$
$$
\text{macro-}F1=\frac1K\sum_c F1_c,\qquad
\text{weighted-}F1=\sum_c \frac{n_c}{N}F1_c .
$$

```
run_torch_experiment(model_name):
    model ← timm.create_model(model_name, pretrained=True, num_classes=K)
    opt   ← AdamW(lr=1e-4, wd=1e-4);  sched ← CosineAnnealingLR(T_max=epochs)
    best_f1 ← -1;  patience ← 0
    for epoch in 1..30:
        train_one_epoch(model, train_loader, CE, opt)
        y_true,y_pred ← evaluate(model, val_loader)
        f1 ← macro_F1(y_true, y_pred);  sched.step()
        if f1 > best_f1: best_f1←f1; patience←0; save_ckpt()
        else: patience←1+patience;  if patience ≥ 7: break
    load best_ckpt
    y_true,y_pred ← evaluate(model, test_loader)
    dump metrics(accuracy, P/R/F1 macro+weighted) ; save confusion_matrix.png
```

---

## 6.6 Layer 4.5 — Hybrid CNN–LSTM Feature Fusion & Algorithm 1

**Sources:** `backend/python/hybrid_cnn_lstm/` — `fusion_model.py` (model factory + `AdaptiveFeatureFusion` layer), `tune_fusion_weights.py` (grid search), `train_hybrid.py`, `evaluate_hybrid.py` (Table 6), `algorithm1.py` (Table 2), `make_paired_dataset.py` (paired manifest), `data_pipeline.py`.

The manuscript's central contribution is a **multimodal model** that fuses the spatial (CNN) and temporal (LSTM) branches instead of running them as isolated services.

### 6.6.1 Adaptive feature fusion (Eq. 1)

Each branch is projected to a common dimensionality $n = 128$:
$$
R_{\text{visual}} = \text{ReLU}\!\big(W_v\,\text{GAP}(f_{\text{B0}}(\mathbf x))\big)\in\mathbb R^{n},\qquad
R_{\text{environment}} = \text{ReLU}\!\big(W_e\,\text{LSTM}_{32}(\mathbf s_{1:T})\big)\in\mathbb R^{n},
$$
and combined by a **fixed** convex weighting
$$
\boxed{\,R_{\text{total}} = w_1\,R_{\text{visual}} + w_2\,R_{\text{environment}},\qquad w_1+w_2 = 1\,}
$$
implemented by `AdaptiveFeatureFusion(w1)`. The weights are **hyper-parameters**, not learned: `tune_fusion_weights.py` sweeps $w_1\in\{0,0.1,\dots,1\}$ under stratified 5-fold CV and selects the value maximising mean validation pest-F1 (tie-break: min loss). The manuscript optimum is $w_1 = 0.6,\ w_2 = 0.4$ (`fusion_weights.json`).

### 6.6.2 Decision heads (Eq. 3)

$$
P_{\text{pest}} = \operatorname{softmax}(W_p R_{\text{total}}),\quad
P_{\text{health}} = \operatorname{softmax}(W_h R_{\text{total}}),\quad
\hat{\mathbf m} = W_m R_{\text{total}}\ (\text{linear, VPD} + 4\ \text{micro-climate targets}).
$$
Loss $= \mathcal L_{\text{pest}} + \tfrac12\mathcal L_{\text{health}} + \tfrac12\,\text{MSE}_{\text{micro}}$; Adam $10^{-3}$, batch 32, dropout 0.2, seed 42, stratified 5-fold CV. The **no-fusion baseline** of Table 6 replaces the weighted sum with concatenation + a dense projection (`fusion="concat"`).

### 6.6.3 Paired dataset

Real synchronised image↔sensor pairs are not distributed, so `make_paired_dataset.py` attaches a **reproducible synthetic** $T=6$ micro-climate window (seed 42) to every leaf image, timestamp-aligned ($|\Delta t| = 0 \le 5$ min). Disease state is coupled to atmospheric stress (healthy → low VPD, diseased → high VPD) so fusion carries genuine cross-modal signal — the same synthetic-series methodology already used by Layer 5.

### 6.6.4 Algorithm 1 (Table 2) — `algorithm1.py`

```
for each timestep t:
    F_t, S_t ← capture(camera, IoT)                                  # 2
    S_t ← linear_interpolate(S_t)  if NaN                            # 3
    x_img ← resize224(F_t)                                           # 4
    x_sen ← minmax(S_t) ⊕ cyclical_hour ⊕ (temp×humidity)           # 5
    R_visual      ← CNN(x_img)                                       # 6
    R_environment ← LSTM(x_sen)                                      # 7
    assert |t_img − t_sen| ≤ 5 min                                   # 8
    R_total ← w1·R_visual + w2·R_environment                         # 9
    P ← softmax(R_total)                                             # 10
    pest_class, pest_conf ← argmax(P_pest)                           # 11
    pest_label ← pest_class if pest_conf ≥ τ_p else "No pest detected"   # 12
    health_class, health_conf ← argmax(P_health)                     # 13
    status ← health_class if health_conf ≥ τ_health else "Monitor"   # 14
    if pest_label ≠ "No pest detected":                              # 15
        if pest_class == "hispa": actuate(ultrasonic_repeller, uv_lamp)
        else: recommend(selective_insecticide_spraying)
    elif status == "High Stress": notify(irrigation)
    else: continue_monitoring
    transmit(payload → Laravel REST + MQTT broker)                   # 16
```
$\tau_p = \tau_{\text{health}} = 0.5$. Step 16 is a `POST` to `--laravel-url` and an MQTT publish to `taniverse/{iot_id}/inference` (both optional; `--dry-run` by default). This is the model-level counterpart of the Layer 6 rule engine — Layer 6 stays the deterministic agronomic advisory on the raw sensor record.

### 6.6.5 Table 6 protocol — `evaluate_hybrid.py`

Four rows (CNN-only, LSTM-only, CNN+LSTM concat, proposed fusion) × {Accuracy, F1, VPD RMSE, latency ms/frame}, each as mean ± std over stratified 5-fold CV; latency is the median of single-sample `predict` calls.

---

## 7. Layer 5 — LSTM Micro-climate & Plant-Health Forecasting

**Sources:** offline regression study (reproduces manuscript Table 5 / Figure 9 exactly) — `backend/python/rnn/{evaluate_lstm_microclimate_10k,generate_real_figure9_2025dates,rebuild_figure9_labeled,make_synthetic_microclimate_dataset}.py`, `display_results.py`; worked single-example paragraph (sec. 4.2) — `run_real_vpd_example.py`; Figure 4(b) JSON panel — `make_figure4b_json_output.py`; online service — `backend/python/rnn/run.py`, `app/__init__.py`, `app/routes/api.py`, `app/services/{model_service,forecast_service,data_service}.py`, `app/utils/data_processor.py`; artefacts — `config/plant_health_lstm_model.h5`, `feature_scaler.pkl`, `label_encoder.pkl`, `feature_columns.pkl`, `model_config.pkl`.

There are **two LSTM models** sharing one architecture family:

| | Regression model (`evaluate_lstm_microclimate_10k.py`) | Production plant-health model (`config/*.pkl/.h5`) |
|---|---|---|
| Input | $T=6$ × 5 features `[Ta, RH, pH, L, VPD]` | $T=6$ × **34** engineered features (`model_config.pkl`) |
| Output | 5-D real vector (next-step $[Ta,RH,pH,L,\mathrm{VPD}]$) | 3-class softmax `{Healthy, Moderate Stress, High Stress}` |
| Loss | MSE | categorical cross-entropy |
| Head activation | linear | softmax |

### 7.1 Vapour-Pressure Deficit (physical feature)

Saturation vapour pressure by the Tetens/Magnus equation and the resulting deficit:
$$
e_s(T) = 0.6108\,\exp\!\left(\frac{17.27\,T}{T+237.3}\right)\ [\mathrm{kPa}],\qquad
e_a = e_s\,\frac{\mathrm{RH}}{100},\qquad
\boxed{\ \mathrm{VPD} = e_s - e_a = e_s\!\left(1-\tfrac{\mathrm{RH}}{100}\right)\ }
$$
VPD is both an LSTM input/target and a recommendation trigger (Layer 6).

### 7.2 Sequence construction

Given a resampled multivariate series $\{x_t\}_{t=1}^{n}$ at $\Delta = 4\,\mathrm h$ (so $T=6$ spans 24 h), sliding windows are
$$
X^{(i)} = (x_i, x_{i+1}, \dots, x_{i+T-1}) \in \mathbb R^{T\times d},\qquad
y^{(i)} = x_{i+T},\qquad i = 1,\dots,n-T.
$$
Min–max normalisation per feature $j$ (fitted on train only):
$$
\tilde x_{j} = \frac{x_j - \min_j}{\max_j - \min_j}\in[0,1],
$$
inverted after prediction: $\hat x_j = \tilde{\hat x}_j(\max_j-\min_j)+\min_j$.

### 7.3 Feature engineering for the 34-D production model

From `data_processor.process_for_lstm` / `process_for_prediction`:

* **Cyclical time encoding** (avoids the 23→0 discontinuity):
  $$
  \text{Hour}_{\sin}=\sin\!\frac{2\pi\,\text{h}}{24},\ \text{Hour}_{\cos}=\cos\!\frac{2\pi\,\text{h}}{24};\quad
  \text{Day: } /31,\quad \text{Month: } /12 .
  $$
* **Interaction terms:** $\text{Moisture\_Temp}= \theta\cdot T_s/100$, $\text{Humidity\_Temp}= \mathrm{RH}\cdot T_a/100$.
* **Nutrient descriptors:** $\text{NPK\_Balance} = (N+P+K)/3$, ratios $N/P$, $N/K$ (and $\text{NPK\_Ratio}_{\{N,P,K\}} = \{N,P,K\}/\text{NPK\_Balance}$ in the classical path).
* **Local dynamics:** rolling mean $\bar x_t^{(w)}=\frac1w\sum_{k=0}^{w-1}x_{t-k}$ ($w=3$) and change rate (percent change) $\rho_t = (x_t-x_{t-1})/x_{t-1}$.
* **24 h aggregates & trend (classical path):** $\bar x_{24h}$ = mean of last 6 readings; $\text{trend} = \dfrac{x_{t}-x_{t-5}}{\Delta t\,[\mathrm s]}$.
* **Quantile stress flags (classical path):** with historical quartiles,
  $\text{Moisture\_Stress}= \mathbb 1[\theta < Q_{25}(\theta)]$,
  $\text{Temperature\_Stress}= \mathbb 1[T_s>Q_{75}(T_s)\ \vee\ T_s<Q_{25}(T_s)]$,
  $\text{Light\_Stress}= \mathbb 1[L<Q_{25}(L)]$.

### 7.4 LSTM cell equations

For input $x_t\in\mathbb R^{d}$, previous hidden $h_{t-1}$ and cell $c_{t-1}$:
$$
\begin{aligned}
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f) &&\text{(forget gate)}\\
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i) &&\text{(input gate)}\\
\tilde c_t &= \tanh(W_c x_t + U_c h_{t-1} + b_c) &&\text{(candidate)}\\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde c_t &&\text{(cell state)}\\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o) &&\text{(output gate)}\\
h_t &= o_t \odot \tanh(c_t) &&\text{(hidden state)}
\end{aligned}
$$

**Stacked architecture** (both variants, `build_lstm_regression_model`, defined in `evaluate_lstm_microclimate.py` and reused by `evaluate_lstm_microclimate_10k.py`):
$$
\mathbf x_{1:T}
\xrightarrow{\text{LSTM}_{64},\,\text{seq}}
\text{BN}
\xrightarrow{\text{LSTM}_{32}}
\text{BN}\to\text{Drop}_{0.2}
\to \text{ReLU}(W_{32}\cdot)\to\text{BN}\to\text{Drop}_{0.2}
\to
\begin{cases}
W_{5}\cdot \ (\text{linear}) & \text{regression}\\
\operatorname{softmax}(W_{3}\cdot) & \text{health}
\end{cases}
$$

**Batch normalisation:** $\hat z = \gamma\,\dfrac{z-\mu_{\mathcal B}}{\sqrt{\sigma_{\mathcal B}^{2}+\varepsilon}}+\beta$.
**Dropout** ($p=0.2$): $\tilde h = h\odot \text{Bernoulli}(1-p)/(1-p)$.
Optimiser `adam`; regression loss $\text{MSE}=\frac1n\sum(y-\hat y)^2$, metric MAE.

### 7.5 Regression evaluation metrics

Per parameter $k$:
$$
\text{MAE}_k = \frac1n\sum_{t}\lvert y_{t,k}-\hat y_{t,k}\rvert,\qquad
\text{RMSE}_k = \sqrt{\frac1n\sum_{t}(y_{t,k}-\hat y_{t,k})^{2}},
$$
$$
R^2_k = 1 - \frac{\sum_t (y_{t,k}-\hat y_{t,k})^{2}}{\sum_t (y_{t,k}-\bar y_{k})^{2}} .
$$

```
generate_real_figure9_2025dates():                  # backend/python/rnn/, matches manuscript Table 5 exactly
    data ← synthetic_microclimate(n=10_000, seed=42)  # diurnal+seasonal sinusoids + noise; VPD from Tetens/Magnus
    F ← data[[temperature,humidity,ph,light_intensity,vpd]]
    scaler_X, scaler_y ← MinMax().fit(F)
    X, y ← sliding_windows(scaler(F), T=6)             # X:(9994,6,5)  y:(9994,5)
    # ---- chronological holdout : 6,396 train / 1,599 val / 1,999 test (manuscript sec. 3.3.1 / Table 3) ----
    model ← Sequential([LSTM64→BN→LSTM32→BN→Drop.2→Dense32,relu→BN→Drop.2→Dense5,linear])
    model.compile(Adam(1e-3), mse, [mae]); model.fit(X_train, y_train, epochs=100, batch=32, validation_data=(X_val,y_val))
    y_pred ← model.predict(X_test)                     # single fixed split, NOT k-fold — point estimates, no ± std
    for k in [temperature,humidity,ph,light_intensity,vpd]: report MAE_k, RMSE_k, R²_k   # → manuscript Table 5
    save figure9_2025dates_predictions.csv             # consumed by rebuild_figure9_labeled.py for the (a)-(e) panel figure
    # separately: evaluate_lstm_microclimate_10k.rolling_vpd_forecast_r2() gives the 14-day rolling VPD
    # forecast mentioned in sec. 4.5; the manuscript does not quote a specific R² for that rolling run,
    # only the full 1,999-window holdout numbers above (Table 5) — do not conflate the two.
```

`evaluate_lstm_microclimate.py` (no `_10k` suffix) is still imported by `evaluate_lstm_microclimate_10k.py` for `cross_validate_regression`/`rolling_vpd_forecast_r2`, but its own directly-run output uses a different (smaller/earlier) data configuration and does **not** reproduce the manuscript's Table 5 numbers — treat it as a dependency, not a reproduction script.

### 7.6 Online recursive multi-step forecasting (`forecast_service.py`)

The service produces $H = 6d$ steps ($d\le 14$ days, 4 h grid). Because true future drivers are unknown, it **recursively** feeds predictions back and simulates the diurnal evolution of the driver variables with first-order exponential smoothing toward physically-motivated targets.

* **Solar term:** for hour-of-day $\eta$,
  $\lambda(\eta) = \max\!\big(0,\min(1,\sin\tfrac{(\eta-6)\pi}{12})\big)$ for $6\le\eta\le18$, else $0$.
* **Light:** target $L^\* = 200 + 600\,\lambda$ (day) / small night value; update $L\leftarrow 0.7L + 0.3L^\*$.
* **Air temperature:** target $T^\* = 22 + 6\lambda$ (08–16 h) / $22 - 4(1-\lambda)$ (night); $T\leftarrow 0.9T + 0.1T^\*$.
* **Soil temperature (lag):** $T_s \leftarrow T_s + 0.05\,(T - T_s)$.
* **Soil moisture (depletion + auto-irrigation):** $\theta \leftarrow \theta + (\text{trend} - 0.3)$; if $\theta < 20$ then $\theta \leftarrow \theta + 15$ (watering event); clamp $[10,90]$.
* **Humidity (anti-correlated with $T$):** $\mathrm{RH}^\* = 70 - 2(T-20)$; $\mathrm{RH}\leftarrow 0.8\,\mathrm{RH} + 0.2\,\mathrm{RH}^\*$; clamp $[30,90]$.
* **Other channels:** $x\leftarrow x + \text{trend}\,(1 + \mathcal U(-0.1,0.1))$, then domain clamps (pH $\in[4.5,8.5]$, NPK $\in[5,50]$).
* **Historical trend** (when $\ge 6$ points): $\text{trend}_x = \dfrac{x_{t}-x_{t-5}}{\Delta t\,[\mathrm s]}\times 3600$ (per-hour), else defaults.

At each step the (unchanged in code) sequence tensor is re-scored by the LSTM to attach `predicted_health` + class confidences.

```
generate_lstm_forecast(iot_id, days):
    hist ← DataService.plant_data[iot_id] sorted by created_at
    if len(hist) < T: return None
    Xseq ← scale( engineer( hist[-T:] ) )                       # (1, 6, 34)
    cur  ← current_values(hist[-1])
    out  ← [ current_entry + predict_lstm(Xseq) ]
    for i in 1 .. days*6 - 1:
        η ← (start_hour + 4i) mod 24
        λ ← clamp(sin((η-6)π/12), 0, 1) if 6≤η≤18 else 0
        update_forecast_values(cur, η, λ, trends)               # exponential smoothing rules above
        p ← predict_lstm(Xseq)                                  # softmax → argmax → label_encoder
        out.append( {date, forecast_type:'forecast', soil_temp, humidity, soil_moisture,
                     predicted_health:p.class, confidence:p.probs, …} )
    return out

generate_forecast(iot_id, days):
    return generate_lstm_forecast(...) or generate_traditional_forecast(...)   # graceful fallback
```

### 7.7 Classical fallback model

`model_service.predict_traditional` loads a scikit-learn estimator (`plant_health_prediction_model.joblib`) and returns
$\hat c = \arg\max_c \Pr(c\mid \mathbf x)$ with the full class-probability vector from `predict_proba`. Feature construction is `process_for_prediction` (§7.3, aggregates + quantile stress flags). This path is used when the Keras LSTM or its pre-processing artefacts fail to load.

---

## 8. Layer 6 — Rule-Based Agronomic Recommendation Engine

**Source:** `backend/laravel/app/Http/Controllers/RecommendationService.php` (invoked by `DashboardController`, `DataController`); a simpler variant exists at `Modules/Dashboard/app/Service/RecommendationService.php`.

### 8.1 Structure

A deterministic expert system $g:(\,\text{latest sensor record }s,\ \text{latest disease record }v\,)\mapsto$ ordered advisory list. Two rule groups:

**(a) Disease→treatment lookup.** For the CV output $v.\texttt{penyakit}$, a static map $\Phi$ returns a chemical/cultural control package (9 entries, e.g. *blast* → systemic fungicide tricyclazole/azoxystrobin + intermittent irrigation). Formally $r_{\text{disease}} = \Phi(v.\texttt{penyakit})$.

**(b) Threshold rules on the sensor record.** With the VPD from §7.1 evaluated on air $T$ and RH:
$$
r_{\mathrm{VPD}} =
\begin{cases}
\text{“VPD Tinggi” (danger)}, & \mathrm{VPD} \ge 1.5\ \mathrm{kPa}\\
\text{“VPD Rendah” (warning)}, & \mathrm{VPD} \le 0.5\ \mathrm{kPa}\\
\varnothing, & \text{otherwise.}
\end{cases}
$$

Each agronomic variable $p$ has an optimal interval $[\ell_p, u_p]$ (Table below); a rule fires on interval violation:
$$
\text{rule}(p) =
\begin{cases}
\text{low-}p\ \text{advisory}, & s_p < \ell_p\\
\text{high-}p\ \text{advisory}, & s_p > u_p\\
\varnothing, & \ell_p \le s_p \le u_p.
\end{cases}
$$

| Parameter $p$ | $\ell_p$ | $u_p$ | Unit |
|---|---|---|---|
| Air temperature | 22 | 28 | °C |
| Air humidity | 70 | 85 | % |
| Wind speed | – | 3.6 | m s⁻¹ |
| Soil pH | 5.5 | 7.0 | – |
| Soil moisture | 70 | 85 | % |
| EC | 800 | 1500 | µS cm⁻¹ |
| TDS | – | 500 | ppm |
| Soil temperature | 27 | 34 | °C |
| Nitrogen | 20 | 50 | mg kg⁻¹ |
| Phosphorus | 10 | 30 | mg kg⁻¹ |
| Potassium | 100 | 200 | mg kg⁻¹ |

De-duplication is by advisory title (a title fires at most once per refresh); the final list is $\big[\,r_{\text{danger}}\,\big] \Vert \big[\,r_{\text{other}}\,\big]$ (danger-severity items first).

**(c) Automated actuation decision** — `getActuation($s, $v, \text{health})` is a direct port of **Algorithm 1 step 15** (returns exactly one action):
$$
\text{act} =
\begin{cases}
\texttt{activate\_ultrasonic\_repeller\_and\_uv\_lamp}, & v.\texttt{penyakit} = \text{Hispa}\\
\texttt{recommend\_selective\_insecticide\_spraying}, & v.\texttt{penyakit} \notin \{\text{Hispa},\ \text{Normal}\}\\
\texttt{irrigation\_notification}, & \text{health} = \text{High Stress}\ \vee\ \mathrm{VPD} \ge 1.5\ \mathrm{kPa}\\
\texttt{continue\_monitoring}, & \text{otherwise.}
\end{cases}
$$
This is the model-side counterpart to the ESP32's local illumination loop and the LiDAR intruder loop — it is what the hybrid edge node (`algorithm1.py`) posts to `POST /api/inference`, which the controller then re-publishes to the MQTT broker.

### 8.2 Algorithm (pseudocode)

```
getRecommendations(sensorData, cameraData):
    danger ← [] ; other ← [] ; seen ← ∅
    # (a) disease → action
    for v in cameraData:
        if v.penyakit in Φ: danger.append(Φ[v.penyakit]); seen.add(v.penyakit)
    # (b) threshold rules
    for s in sensorData:
        es  ← 0.6108 * exp(17.27*s.T / (s.T + 237.3))
        vpd ← es * (1 - s.RH/100)
        if vpd ≥ 1.5 and "VPD Tinggi" ∉ seen: danger.append(advisory_VPD_high); seen.add(...)
        elif vpd ≤ 0.5 and "VPD Rendah" ∉ seen: danger.append(advisory_VPD_low); seen.add(...)
        for p in {T, RH, wind, pH, moisture, EC, N, P, K}:
            if   s[p] < ℓ[p] and title_low[p] ∉ seen:  append(low advisory);  seen.add(...)
            elif s[p] > u[p] and title_high[p] ∉ seen: append(high advisory); seen.add(...)
    return danger ++ other        # severity-ordered
```

---

## 9. End-to-End Data & Control Flow

```
   ┌─ every 900 s (sensing) + responsive (actuation) ─────────────────────────────────┐
   │ ESP32: read 17 channels ─► JSON ─► POST /api/kirim_sensor ─► sensor_iots           │
   │ ESP32: SUB mqtt taniverse/{id}/inference  ∨  GET /api/inference/{id}               │
   │        ─► action ─► {UV, ultrasonic, irrigation} relays (GPIO19/23/27)             │
   │ ESP32: GET /api/bacajson/{id} ─► movement_detected ─► ultrasonic relay (OR)        │
   └───────────────────────────────────────────────────────────────────────────────────┘
   ┌─ per camera frame (06–23 h) ──────────────────────────────────────────────────────┐
   │ RPi CV: Haar ROI ─► EfficientNet-B0 ─► (penyakit, prob) ─► sensor_kameras + image  │
   └───────────────────────────────────────────────────────────────────────────────────┘
   ┌─ on dashboard request / poll ─────────────────────────────────────────────────────┐
   │ Flask LSTM: GET /api/sensor?iot_id ─► build T=6 window ─► LSTM forecast (H=6d)     │
   │            served at GET /forecast/{iot_id}?days=…                                 │
   │ Laravel DashboardController: gather {sensor_iots latest, sensor_kameras latest,    │
   │            /forecast 14-day + 1-day}  ─► RecommendationService  ─► Blade + charts  │
   └───────────────────────────────────────────────────────────────────────────────────┘
```

**Closed control loops**

1. *Illumination loop (local):* BH1750 lux $< 100$ ⇒ ESP32 raises the UV-lamp relay (no network).
2. ~~*Intruder loop (distributed):* RPi LiDAR motion ⇒ `lidars` row ⇒ Laravel `/api/bacajson` ⇒ ESP32 ultrasonic relay~~ — the LiDAR producer (Layer 2, Section 4) has been removed from the codebase; the ESP32 firmware's `poll_lidar()` call and the `/api/bacajson` route may still exist as unfed legacy surface, but no process currently writes to `lidars`.
3. *Advisory loop (human-in-the-loop):* sensors + CV disease ⇒ recommendation engine ⇒ operator action (irrigation, fungicide, fertiliser).
4. *Fused actuation loop (model-in-the-loop):* hybrid CNN–LSTM (`algorithm1.py`) ⇒ `getActuation` decision ⇒ `POST /api/inference` ⇒ `hybrid_inferences` + MQTT publish `taniverse/{iot_id}/inference` ⇒ **ESP32** subscriber drives the UV-lamp / ultrasonic-repeller / irrigation-valve relays (GPIO19/23/27), REST-polled fallback every 15 s.

---

## 10. Deployment Topology

| Tier | Hardware | Software stack | Notes |
|---|---|---|---|
| Field node | ESP32, RS485 4-in-1 soil probe, DHT22, BH1750, BMP280, MQ-135, cup anemometer, tipping bucket, TDS probe, **3× relay (UV lamp / ultrasonic repeller / irrigation valve)**, 4S Li-ion (+ 200 Wp solar) | Arduino C++ (`WiFiManager`, `ArduinoJson`, `HTTPClient`, **`PubSubClient`**, `Adafruit_BMP280`, `BH1750`, `DHT`) | 15-min sensing cadence; responsive actuation via MQTT sub + REST fallback; captive-portal provisioning; active-low relays |
| Edge node | Raspberry Pi + camera | Python 3 (`pymysql`), Flask + OpenCV + TensorFlow (`app.py`, port 7000) | local MySQL mirror `padi_sawah`; MJPEG stream. *(The servo + TF-Luna LiDAR scanning node once ran alongside this on the same class of hardware; its source has been removed — see Section 4.)* |
| Application server | `petaniasik.my.id` | Laravel 12 (PHP 8.2) + MySQL + `nwidart/laravel-modules` + `spatie/laravel-permission` + `spatie/activitylog`; Flask LSTM service (port 5000, `flask-socketio`); optional MQTT broker | REST + Blade dashboard; RBAC; Excel/ZIP export; `App\Services\MqttPublisher` (php-mqtt client if installed, else built-in QoS-0 socket publisher; `config/mqtt.php`, `MQTT_*` env) |
| Client | Browser | Blade + Bootstrap + Chart.js | dashboard, monitoring, history, data export |

---

## 11. Reproducibility — Hyperparameter Tables

### 11.1 CNN transfer-learning (`train.py`)

| Item | Value |
|---|---|
| Backbone | EfficientNet-B0, ImageNet weights, `include_top=False` |
| Input | 224 × 224 × 3, `efficientnet.preprocess_input` (0–255) |
| Head | GAP → Drop 0.2 → Dense 128 ReLU → Drop 0.2 → Dense 10 softmax |
| Augmentation (sec. 3.3) | flip H+V, rotation ±25° (25/360·2π), random crop scale 0.8–1.0 (`RandomZoom`), Gaussian noise variance 0.01 on [0,1] |
| Class weights | $w_c = N/(K n_c)$ |
| Phase 1 | base frozen, Adam 1e-3, 40 epochs |
| Phase 2 | top 30 layers unfrozen, Adam 1e-5, 10 epochs (**50 total**, manuscript Table 3) |
| Loss / metric | categorical cross-entropy / accuracy |
| Callbacks | early-stop(val_acc, patience 6, restore best), ReduceLROnPlateau(val_loss, 0.5, patience 3, min 1e-7); ckpt(val_acc) in single-split mode |
| Validation | stratified 70/15/15 split **and** stratified 5-fold CV (`--cv-folds 5`) — the 5-fold mode repartitions all 10,407 images per fold as ≈65% train / ≈15% val / ≈20% test (≈6,765/1,561/2,081); mean ± std + 95% CI ($\mu\pm1.96\sigma/\sqrt{5}$, Eq. 9) — this is **Table 4** |
| Batch / seed | 32 / 42 (`keras.utils.set_random_seed`) |

### 11.2 Benchmark (`benchmark_modern_models.py`)

| Item | Value |
|---|---|
| Models | yolov8n-cls, yolo11n-cls, efficientnetv2_rw_s, efficientnet_b0, vit_base_patch16_224, resnet50, mobilenetv2_100, vgg16 |
| Optimiser (torchvision/timm branches) | AdamW, lr 1e-4, weight-decay 1e-4 |
| Scheduler | CosineAnnealingLR, $T_{\max}$ = epochs |
| Epochs / patience (torchvision/timm branches) | ≤ 30 / 7 (monitor val macro-F1) |
| Batch / img size / seed (torchvision/timm branches) | 16 / 224 / 42 |
| Normalisation | ImageNet mean/std |
| YOLOv8n-cls / YOLOv11n-cls (identical protocol) | epochs 5, batch 8, imgsz 224, device cpu, workers 0, optimizer auto (lr0 0.01, lrf 0.01, momentum 0.937, wd 5e-4, warmup 3), mosaic 1.0, randaugment, erasing 0.4, ultralytics 8.4.24 |

### 11.3 LSTM regression — manuscript Table 5 (`evaluate_lstm_microclimate_10k.py` + `generate_real_figure9_2025dates.py`)

| Item | Value |
|---|---|
| Window $T$ / step | 6 / 4 h (24 h look-back), 4 h forecast horizon |
| Features / targets | `[temperature, humidity, ph, light_intensity, vpd]` (5) → next-step (5) |
| Architecture | LSTM 64 (seq) → BN → LSTM 32 → BN → Drop 0.2 → Dense 32 ReLU → BN → Drop 0.2 → Dense 5 linear |
| Optimiser / loss / metric | Adam (lr $10^{-3}$) / MSE / MAE |
| Epochs / batch | 100 / 32 (manuscript Table 3) |
| Scaling | MinMax on X and y, **fitted on the entire 10,000-record series before windowing/partitioning** (manuscript sec. 3.3.1 states this as a documented preprocessing-leakage limitation, not a strength) |
| Data split | chronological holdout: 6,396 train / 1,599 val / 1,999 test windows — a **single fixed split**, not k-fold; Table 5 values are point estimates, not mean ± std |
| Long-horizon eval | 14-day rolling recursive VPD forecast (sec. 4.5) — the manuscript states this is evaluated but does not quote a specific R² for it; do not substitute the Table 5 holdout R² (0.949) for it |

### 11.4 Production plant-health LSTM (`config/model_config.pkl`)

| Item | Value |
|---|---|
| Window $T$ | 6 |
| Features | 34 engineered (raw agro-vars + cyclical time + interactions + NPK descriptors + rolling mean/change rate) |
| Classes | `{0: Healthy, 1: High Stress, 2: Moderate Stress}` |
| Artefacts | `plant_health_lstm_model.h5`, `feature_scaler.pkl`, `label_encoder.pkl`, `feature_columns.pkl` |
| Forecast horizon | up to 14 days (H = 84 steps of 4 h) |

### 11.5 Hybrid CNN–LSTM feature fusion (`hybrid_cnn_lstm/`)

| Item | Value |
|---|---|
| Branches | EfficientNet-B0 → Dense($n$, ReLU) = $R_{\text{visual}}$ ; LSTM 64→BN→LSTM 32→BN→Drop 0.2→Dense($n$, ReLU) = $R_{\text{environment}}$ |
| LSTM input | $T=6 \times 8$ features: `[temperature, humidity, pH, light, VPD]` + `hour_sin` + `hour_cos` + `temp×humidity` (Algorithm 1 step 5, `data_processor`/`engineer_sensor_window`) |
| Fusion dim $n$ | 128 |
| Fusion (Eq. 1) | $R_{\text{total}} = w_1 R_{\text{visual}} + w_2 R_{\text{environment}}$, $w_1+w_2=1$, fixed |
| Weight selection | grid search $w_1\in\{0,0.1,\dots,1\}$ × stratified 5-fold CV → max val pest-F1; manuscript optimum $w_1=0.6$, $w_2=0.4$ |
| Heads | Dense 10 softmax (pest) · Dense 3 softmax (health) · Dense 5 linear (micro-climate/VPD) |
| Loss | $\mathcal L_{\text{pest}} + 0.5\,\mathcal L_{\text{health}} + 0.5\,\text{MSE}_{\text{micro}}$ |
| Optimiser / batch / dropout / seed | Adam $10^{-3}$ / 32 / 0.2 / 42 |
| Validation (ablation) | stratified 5-fold CV on all 10,407 images, ≈65/15/20 per fold; every metric reported as mean ± std **and** 95% CI ($\mu \pm 1.96\sigma/\sqrt{5}$, Eq. 9) — **Table 6** |
| Final deployed model | separate from the ablation above: 85/15 train/val split, no held-out test fold (manuscript Table 3, "Final Fusion Model") |
| No-fusion baseline | concatenation + Dense projection (`fusion="concat"`) |
| Decision thresholds | $\tau_p = \tau_{\text{health}} = 0.5$ (Algorithm 1) |
| Paired data | synthetic timestamp-aligned image↔sensor windows, seed 42, disease-coupled stress |

---

## 12. Experimental Results Summary

### 12.1 Validation comparison of available classification checkpoints — Table 7 (`benchmark_modern_models.py`, `experiments/scopus_q1_comparison/` + `experiments/yolov11_addition/`)

Eight checkpoints evaluated on the same 2,084-image validation partition (image-level metrics; validation also guided checkpoint selection, so these are **not** independent-test numbers) — manuscript **Table 7** / Section 4.4 ("Comparative Evaluation of Available Classification Checkpoints"), *not* the Table 4 numbers (Table 4 is the separate deployment-oriented Keras backbone comparison, §12.3).

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) |
|---|---|---|---|---|---|
| **YOLOv11n-cls** | **0.9093** | **0.9014** | **0.9003** | **0.8990** | **0.9089** |
| EfficientNetV2-S | 0.8690 | 0.8542 | 0.8510 | 0.8491 | 0.8682 |
| YOLOv8n-cls | 0.8652 | 0.8533 | 0.8452 | 0.8482 | 0.8641 |
| EfficientNet-B0 (deployed) | 0.8589 | 0.8479 | 0.8477 | 0.8474 | 0.8586 |
| ResNet-50 | 0.4290 | 0.2910 | 0.2886 | 0.2560 | 0.3676 |
| ViT-B/16 | 0.1387 | 0.0139 | 0.1000 | 0.0244 | 0.0338 |
| VGG-16 | 0.1108 | 0.0688 | 0.1236 | 0.0642 | 0.0745 |
| MobileNet-V2 | 0.0461 | 0.0046 | 0.1000 | 0.0088 | 0.0041 |

YOLOv11n-cls (5 epochs, batch 8, imgsz 224, CPU — identical protocol to the YOLOv8n-cls baseline; see `experiments/yolov11_addition/yolov11_provenance.md`) now tops the table, ahead of EfficientNetV2-S by +2.03 pp accuracy / +0.99 pp macro-F1, added in response to a reviewer request for a newer benchmark model. The two EfficientNet/YOLOv8 checkpoints below it converge to ≈ 0.86–0.87 accuracy and ≈ 0.85 macro-F1; the remaining backbones did not converge under the shared short-schedule / cuDNN-stability protocol (documented failure mode: MobileNet-V2 and ViT each collapsed to predicting a single class for every validation image, `test_loss` NaN for MobileNet-V2). EfficientNet-B0 remains the deployed backbone for its accuracy/FLOP trade-off on the Raspberry Pi, not because it tops this table.

**Note:** this table is unrelated to Figure 8 (the EfficientNet-B0 confusion matrix on a 500-image class-balanced *subset*, 43.40% accuracy, §12.2 of the manuscript / `Training_Penyakit Padi/{compute_real_50perclass_matrix,plot_real_50perclass_confusion_matrix}.py`). An earlier version of this document conflated the two and cited a "before vs. after augmentation" ablation (`generate_confusion_matrix.py`, `detailed_analysis.py`) as reproducing Figure 8; those scripts produced a different, unreferenced analysis and have since been removed from the repository along with their output figures.

### 12.2 LSTM micro-climate regression — manuscript Table 5 (`generate_real_figure9_2025dates.py` → `figure9_2025dates_predictions.csv`, re-plotted with panel labels by `rebuild_figure9_labeled.py`; $T=6$ = 24 h look-back, chronological holdout, $n_{\text{test}}=1{,}999$)

| Parameter | MAE | RMSE | $R^2$ |
|---|---|---|---|
| Air Temperature (°C) | 0.451 | 0.565 | 0.969 |
| Relative Humidity (%) | 1.759 | 2.185 | 0.937 |
| Soil pH | 0.050 | 0.062 | 0.924 |
| Light Intensity (lux) | 27.186 | 32.396 | 0.993 |
| VPD (kPa) | 0.077 | 0.099 | 0.949 |

These are point estimates from a single chronological holdout split (not k-fold, no ± std — see §11.3). Light intensity has the highest $R^2$ (0.993); soil pH the lowest (0.924). VPD RMSE (0.099 kPa) and MAE (0.077 kPa) are the headline numbers quoted in the manuscript's Discussion and Conclusion. The manuscript separately notes a 14-day rolling recursive VPD forecast (sec. 4.5) but does not quote a specific R² for that rolling evaluation — it is a distinct evaluation from the 1,999-window holdout above and the two should not be conflated. The manuscript also explicitly flags that the MinMax scalers were fitted on the entire 10,000-record series before windowing/partitioning, which is a preprocessing-leakage limitation on interpreting these numbers as unseen-data performance (sec. 3.3.1).

### 12.3 CNN backbone comparison — Table 4 (`compare_cnn_backbones.py`)

One identical **Keras two-stage transfer-learning** protocol (frozen EfficientNet-style head, augmentation of sec. 3.3, Adam $10^{-3}\!\to\!10^{-5}$, 40 + 10 epochs, class weights) evaluated by stratified 5-fold CV; mean ± std. This is the deployment-oriented comparison the manuscript reports as Table 4 (distinct from §12.1 / Table 7).

| Model | Accuracy (%) | Precision | Recall | F1-Score |
|---|---|---|---|---|
| ResNet50 | 85.0 ± 0.8 | 0.83 ± 0.01 | 0.82 ± 0.01 | 0.82 ± 0.01 |
| MobileNetV2 | 83.5 ± 0.9 | 0.81 ± 0.01 | 0.80 ± 0.01 | 0.80 ± 0.01 |
| VGG16 | 84.2 ± 0.7 | 0.82 ± 0.01 | 0.81 ± 0.01 | 0.81 ± 0.01 |
| **EfficientNet-B0 (proposed)** | **89.2 ± 0.6** | **0.89 ± 0.01** | **0.89 ± 0.01** | **0.89 ± 0.01** |

Augmentation lifts EfficientNet-B0 from ≈ 85.0 % to ≈ 89.2 % accuracy (blast ↔ brown_spot confusion reduced).

### 12.4 Hybrid CNN–LSTM feature fusion — Table 6 (`evaluate_hybrid.py`)

Stratified 5-fold CV, mean ± std; latency = median single-sample `predict`.

| Model | Accuracy (%) | F1-Score | RMSE (VPD) | Latency (ms/frame) |
|---|---|---|---|---|
| CNN only | 89.2 ± 0.6 | 0.89 ± 0.01 | — | 92 ± 3 |
| LSTM only | — | — | 0.075 ± 0.006 | 84 ± 2 |
| CNN + LSTM (without fusion) | 90.3 ± 0.7 | 0.91 ± 0.01 | 0.069 ± 0.005 | 104 ± 4 |
| **Proposed CNN–LSTM (feature fusion, $w_1{=}0.6$)** | **93.6 ± 0.8** | **0.93 ± 0.01** | **0.061 ± 0.004** | 110 ± 4 |

Adaptive fusion adds ≈ 4.4 % pest accuracy and cuts VPD error ≈ 18.6 % vs. the unimodal baselines while staying under the 150 ms real-time budget.

---

### Appendix A — File → responsibility map

| Path | Role in the architecture |
|---|---|
| `Arduino/padi_sawah/padi_sawah.ino` | Layer 1 firmware: sensing, filtering, telemetry, **UV/ultrasonic/irrigation actuation** from the fused model (MQTT `PubSubClient` sub + REST `/api/inference` fallback + local BH1750/LiDAR rules) |
| `Arduino/kirim_laravel/kirim_laravel.ino` | Layer 1 minimal POST reference (dummy payload) |
| ~~`backend/python/servo_lidar.py`~~ | *Removed from the repository* — formerly Layer 2: servo sweep + TF-Luna + motion → `lidars` (see Section 4) |
| `backend/laravel/routes/api.php` | Layer 3: REST route table |
| `backend/laravel/app/Http/Controllers/SensorController.php` | Layer 3: ingest, auth, ring-buffer, history/lidar reads |
| `backend/laravel/Modules/Dashboard/app/Http/Controllers/DashboardController.php` | Layer 3/6: scatter-gather of sensor + CV + forecast + recommendations + `getActuation` + latest `HybridInference` (shown in the "Aksi Aktuasi Otomatis" and "Deteksi Hama" cards) |
| `backend/laravel/app/Http/Controllers/RecommendationService.php` | Layer 6: VPD + threshold + disease→action rule engine; `getActuation` = Algorithm 1 step 15 |
| `backend/laravel/Modules/Dashboard/app/Http/Controllers/HybridInferenceController.php` | Layer 3: `POST/GET /api/inference` — ingest & serve fused CNN–LSTM output |
| `backend/laravel/Modules/Dashboard/{database/migrations/*_create_hybrid_inferences_table,app/Models/HybridInference}.php` | Layer 3: `hybrid_inferences` schema + model |
| `backend/laravel/app/Services/MqttPublisher.php`, `config/mqtt.php` | Layer 3: MQTT publish (Algorithm 1 step 16); php-mqtt client or built-in QoS-0 fallback |
| `Training_Penyakit Padi/train.py` | Layer 4: EfficientNet-B0 two-stage transfer learning (50 ep, dropout 0.2, stratified 70/15/15 + 5-fold CV) |
| `Training_Penyakit Padi/compare_cnn_backbones.py` | Layer 4: **Table 4** — ResNet50 / MobileNetV2 / VGG16 / EfficientNet-B0 under one Keras TL protocol + 5-fold CV |
| `Training_Penyakit Padi/make_figure4_combined.py` | Layer 4: **Figure 4** — (a) real augmentation panel via `train.build_augmentation` + (b) reads `figure4b_real_forecast_entry.json` from Layer 5 |
| `backend/python/detect_penyakit/app.py` | Layer 4: real-time Haar + CNN inference service (Flask :7000) |
| `backend/python/detect_penyakit/benchmark_modern_models.py` | Layer 4: **Table 7** — 8-checkpoint timm/YOLO comparison protocol (incl. YOLOv11n-cls) |
| `Training_Penyakit Padi/{compute_real_50perclass_matrix,plot_real_50perclass_confusion_matrix}.py` | Layer 4: **Figure 8** — EfficientNet-B0 confusion matrix on the 500-image class-balanced validation subset |
| `backend/python/hybrid_cnn_lstm/fusion_model.py` | Layer 4.5: adaptive feature-fusion model factory + `AdaptiveFeatureFusion` (Eq. 1) |
| `backend/python/hybrid_cnn_lstm/tune_fusion_weights.py` | Layer 4.5: grid search $w_1/w_2$ with 5-fold CV (→ `fusion_weights.json`) |
| `backend/python/hybrid_cnn_lstm/{train_hybrid,evaluate_hybrid}.py` | Layer 4.5: train fused model; reproduce **Table 6** |
| `backend/python/hybrid_cnn_lstm/algorithm1.py` | Layer 4.5: runnable **Algorithm 1** (Table 2) — fuse → threshold → actuate → transmit |
| `backend/python/hybrid_cnn_lstm/actuation_eval.py` | Layer 4.5/6: actuation success rate SR (sec. 3.8, Eq. 13) from the Algorithm 1 run log |
| `backend/python/hybrid_cnn_lstm/{make_paired_dataset,data_pipeline}.py` | Layer 4.5: synthetic timestamp-aligned image↔sensor manifest + tf.data pipeline |
| `backend/python/rnn/evaluate_lstm_microclimate_10k.py` | Layer 5: LSTM multi-output regression study, N=10,000 (matches manuscript sec. 3.2.2); also hosts `rolling_vpd_forecast_r2` used for the 14-day rolling forecast (sec. 4.5) |
| `backend/python/rnn/evaluate_lstm_microclimate.py` | Layer 5: dependency of the above (`cross_validate_regression`, `rolling_vpd_forecast_r2`); its own direct output does not reproduce Table 5 |
| `backend/python/rnn/generate_real_figure9_2025dates.py` | Layer 5: reproduces manuscript **Table 5** / Figure 9 data exactly (chronological holdout, n_test=1,999); writes `figure9_2025dates_predictions.csv` |
| `backend/python/rnn/rebuild_figure9_labeled.py` | Layer 5: re-renders Figure 9 from that CSV with (a)-(e) subplot labels matching the manuscript caption — the authoritative Figure 9 image |
| `backend/python/rnn/run_real_vpd_example.py` | Layer 5: worked single-VPD-example paragraph (manuscript sec. 4.2) |
| `backend/python/rnn/make_figure4b_json_output.py` | Layer 5: Figure 4(b) JSON panel, read by `Training_Penyakit Padi/make_figure4_combined.py` |
| `backend/python/rnn/app/services/forecast_service.py` | Layer 5: recursive multi-step forecasting with diurnal simulation |
| `backend/python/rnn/app/services/model_service.py` | Layer 5: model/scaler/encoder loading, LSTM & classical inference |
| `backend/python/rnn/app/utils/data_processor.py` | Layer 5: feature engineering (cyclical, interactions, rolling, stress flags) |
| `backend/python/rnn/app/__init__.py`, `run.py` | Layer 5: Flask app factory + `/forecast/{iot_id}` route |
```
