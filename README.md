# Southern Company Network Lifecycle Dashboard

**Live Demo:** [https://innovate-bp17.onrender.com](https://innovate-bp17.onrender.com/)

This project turns Southern Company's raw network equipment workbook into a repeatable lifecycle planning tool. It combines an ETL pipeline with an interactive Dash application so leadership and engineering teams can see where lifecycle risk sits, what it may cost to address, and which site packages should be refreshed first.

## Quick Start

Prerequisites:

- Python 3.10+
- `UAInnovateDataset-SoCo.xlsx` in the project root

Run:

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:8050` in a browser.

To rebuild the processed data from the workbook:

```bash
python etl_pipeline.py
```

## Current Dashboard Structure

The app is organized around one primary narrative page plus five deep dives.

### 1. Network Risk Assessment

This is the executive landing page. It restores the full project story described in the recovered notes:

- risk-focused KPI row
- expandable cost stack
- support coverage and security interpretation
- formal data-confidence section
- fleet composition, lifecycle, and risk visuals
- geographic risk map and state/county views
- recommended refresh schedule
- AI refresh plan optimizer
- executive decision brief and final recommendation

### 2. Deep-Dive Pages

- `EoS / EoL Timeline`
- `Proximity Analysis`
- `Cost & Risk Analysis`
- `Port Utilization`
- `Exception Management`

The app also includes a floating `Fleet Assistant` chat widget that answers questions from aggregated dashboard statistics when `OPENAI_API_KEY` is configured.

## Core Business Rules

- Only active or reachable devices are included.
- `NA` is authoritative for switches, routers, and voice gateways.
- `CatCtr` is authoritative for access points and WLCs, with `Prime` filling wireless gaps.
- State and site are parsed from hostname prefixes.
- Devices at decom sites are excluded.
- `Unknown` lifecycle rows remain in inventory counts but are excluded from risk-focused views per Southern Company guidance.

## Current Snapshot

Based on the current processed output:

- `22,744` devices processed from the workbook
- `22,743` devices visible in the dashboard after persisted exception handling
- `7,409` lifecycle-known devices in the current risk-focused view
- `15,334` `Unknown` lifecycle devices retained in inventory reporting
- `99.92%` geocoded coverage
- `$21.39M` hardware-only refresh cost in the risk-focused fleet
- `$48.44M` modeled all-in refresh cost in the risk-focused fleet
- `96.0%` replacement mapping coverage in the risk-focused fleet
- `97.3%` staffing-hour coverage in the risk-focused fleet

Lifecycle counts in the current visible risk-focused view:

- `Past EoL`: `1,146`
- `Past EoS`: `6,263`
- `EoS within 1yr`: `0`
- `Current`: `0`

## Default Optimizer Recommendation

With the restored default optimizer settings:

- Budget cap: `12M`
- FTE cap: `10`
- Objective: `Balanced`

Current first wave:

- `8` sites
- `2,162` devices
- about `$8.75M` modeled refresh spend
- about `$9.55M` planning budget including site mobilization reserve
- `10.0` FTE
- about `41.8%` modeled risk coverage

Current recommended sites:

1. `VNP` - Vogtle Nuclear Plant
2. `XGP` - GPC - Corp HQ
3. `PLM` - APC-Miller Steam Plant
4. `QPF` - Plant Franklin - Southern Power
5. `WLV` - SCS Wilsonville
6. `BFG` - Birmingham Fleet Garage
7. `SIN` - Plant Sinclair Dam
8. `SPO` - Sands Place OH

## Files

- `etl_pipeline.py`: workbook ingestion and transformation
- `app.py`: Dash dashboard
- `pipeline_output/unified_devices.csv`: processed dataset
- `pipeline_output/pipeline_metadata.json`: run metadata
- `pipeline_output/exceptions.json`: persisted business exceptions

## Optional Chat Configuration

To enable `Fleet Assistant`, set:

```bash
OPENAI_API_KEY=your_key_here
```

If no API key is configured, the widget remains visible but returns an inline availability message instead of crashing.
