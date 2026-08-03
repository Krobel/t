# TinyML B-RAS soft-sensor repository

This repository contains code for an ESP32-S3 and Blues Notecard water-quality monitoring workflow for a biofloc-based recirculating aquaculture system.

It includes:

- Sensor-only firmware for the biofloc reactor and fish tank.
- TinyML-enabled firmware for the biofloc reactor and fish tank.
- Generated embedded model headers for TAN, TOC, DOC, TN, and DN prediction.
- Python model-development scripts.
- Model-export documentation and templates.
- Example input-data formats.

## Repository structure

```text
firmware/
  biofloc_node_with_TinyML/
  biofloc_node_without_TinyML/
  fish_node_with_TinyML/
  fish_node_without_TinyML/
embedded_models/
model_training/
model_export/
data_examples/
docs/
original_uploaded_code/
```

## Firmware summary

| Firmware folder | Node | TinyML  | Main purpose |
|---|---|---:|---:|
| `biofloc_node_with_TinyML` | Biofloc reactor | Yes  | Sensor telemetry plus embedded soft-sensor prediction. |
| `biofloc_node_without_TinyML` | Biofloc reactor | No | Sensor-only baseline with turbidity. |
| `fish_node_with_TinyML` | Fish tank | Yes  | Sensor telemetry plus embedded soft-sensor prediction. |
| `fish_node_without_TinyML` | Fish tank | No | Sensor-only baseline. |

## Embedded model input order

### Biofloc model

The biofloc embedded model uses:

1. DO_B(mg/L)
2. ORP_B(mV)
3. EC_B(mS/cm)
4. pH_B
5. Temp_B(C)
6. Turbidty_B(NTU)

### Fish model

The fish embedded model uses:

1. DO_F(mg/L)
2. ORP_F(mV)
3. pH_F
4. Ec_F(mS/cm)
5. Temp_F(C)


