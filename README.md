# Southern Company Network Equipment Lifecycle Dashboard

An automated data analytics pipeline and interactive dashboard for Southern Company's Network Services division. Surfaces equipment aging, risk exposure, cost optimization opportunities, and refresh planning recommendations across thousands of Cisco networking devices.

## Quick Start

### Prerequisites
- Python 3.10+
- The dataset file `UAInnovateDataset-SoCo.xlsx` in the project root

### Install & Run

**Windows:**
```
run.bat
```

**Mac/Linux:**
```bash
chmod +x run.sh
./run.sh
```

**Manual:**
```bash
pip install -r requirements.txt
python etl_pipeline.py
python app.py
```

Open **http://localhost:8050** in your browser.

---

## Architecture Overview

```
UAInnovateDataset-SoCo.xlsx
        |
        v
  ┌─────────────────────────────────────┐
  │         etl_pipeline.py             │
  │                                     │
  │  1. Load all worksheets             │
  │  2. Filter active/reachable only    │
  │  3. Normalize device types          │
  │  4. Deduplicate (CatCtr/Prime win)  │
  │  5. Expand serial stacks            │
  │  6. Parse hostname -> state/site    │
  │  7. Join SOLID location data        │
  │  8. Join ModelData (EoS/EoL/cost)   │
  │  9. Compute lifecycle status & risk │
  │ 10. Apply user exceptions           │
  │ 11. Output unified CSV              │
  └─────────────────────────────────────┘
        |
        v
  pipeline_output/
  ├── unified_devices.csv
  ├── pipeline_metadata.json
  └── exceptions.json
        |
        v
  ┌─────────────────────────────────────┐
  │            app.py                   │
  │     (Dash Interactive Dashboard)    │
  │                                     │
  │  - Executive Summary                │
  │  - Geographic Risk Map              │
  │  - EoS/EoL Timeline                │
  │  - Proximity Analysis               │
  │  - Cost & Risk Analysis             │
  │  - Exception Management             │
  │  - Refresh Priorities               │
  └─────────────────────────────────────┘
```

---

## Data Model

### Source Sheets

| Sheet | Description | Records |
|-------|-------------|---------|
| **SOLID** | Site directory (address, state, zip) | ~3,630 |
| **SOLID-Loc** | Site geocoding (lat/lon, county, affiliate) | ~3,631 |
| **NA** | Network Automation — switches, routers, VGs | ~9,395 |
| **PrimeAP** | Cisco Prime — access points | ~565 |
| **PrimeWLC** | Cisco Prime — wireless LAN controllers | ~7 |
| **CatCtr** | Catalyst Center — APs, WLCs, switches, routers | ~11,679 |
| **Decom** | Sites scheduled for decommission | 6 |
| **ModelData** | Model lifecycle dates, costs, replacement info | 167 |
| **Pricing** | Detailed SKU-level pricing | 133 |
| **Glossary** | Column and term definitions | 98 |

### Unified Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `device_id` | string | Unique hash ID |
| `hostname` | string | Device hostname (normalized) |
| `ip_address` | string | Management IP |
| `device_type` | string | Switch, Router, Access Point, Voice Gateway, or WLC |
| `model` | string | Cisco model number |
| `serial_number` | string | Individual serial (stacks expanded) |
| `state_code` | string | 2-char state from hostname |
| `site_code` | string | 3-char site from hostname |
| `state` | string | Full state abbreviation |
| `affiliate` | string | Owning affiliate (GPC, APC, MPC, etc.) |
| `latitude/longitude` | float | Geocoordinates |
| `county` | string | Physical county |
| `eos_date` | date | End-of-Sale date |
| `eol_date` | date | End-of-Life date |
| `lifecycle_status` | string | Past EoL, Past EoS, EoS within 1yr, Current, Unknown |
| `risk_score` | int | 0-100 composite risk score |
| `risk_tier` | string | Critical, High, Medium, Low |
| `total_refresh_cost` | float | Material + labor cost to refresh |
| `exception_flagged` | bool | User-flagged exception |
| `source` | string | Data source (NA, CatCtr, PrimeAP, PrimeWLC) |

---

## Transformation Rules Applied

### 1. Active Devices Only
- **NA**: `Device Status == "Active"`
- **CatCtr**: `reachabilityStatus == "Reachable"`
- **PrimeAP**: `upTime > 0`
- **PrimeWLC**: `reachability == "REACHABLE"`

### 2. Device Type Normalization
NA types are mapped to canonical categories:
- `L3Switch`, `Application Switch`, `Virtual Switch` → **Switch**
- `Router` → **Router**
- `Voice Gateway` → **Voice Gateway**
- `Firewall`, `Virtual Firewall` → **excluded** (out of scope)
- `Wireless Controller`, `WirelessLC` → removed from NA (CatCtr/Prime authoritative)

### 3. Source-of-Truth Hierarchy
- **Switches, Routers, Voice Gateways**: NA is authoritative
- **Access Points, WLCs**: CatCtr is primary; Prime fills gaps for devices not in CatCtr
- Duplicates in NA for AP/WLC types are dropped

### 4. Hostname Parsing
- First 2 characters → state code
- Characters 3-5 → site code
- Domain suffixes (e.g., `.southernco.com`) stripped before parsing

### 5. Serial Number Expansion
Comma-separated serial numbers (switch stacks, HA pairs) are expanded into individual device records.

### 6. Decommission Filtering
Devices at the 6 decom sites (GSE, GSI, MCM, PSW, RED, XAG) are excluded.

### 7. Additional Rules from Glossary
- AP hostnames starting with "AP" are excluded
- 5508 WLCs are not flagged for lifecycle (no replacement mapped)

---

## Dashboard Views

### 1. Executive Summary
KPI cards showing total devices, past-EoL count, past-EoS count, critical risk count, estimated refresh cost, and active sites. Includes device type distribution, lifecycle status pie chart, risk distribution bar chart, and breakdowns by state and affiliate.

### 2. Geographic Risk Map
Interactive map using scatter markers sized by device count and colored by average risk score. Includes a US choropleth by state and a top-20 county risk ranking.

### 3. EoS/EoL Timeline
Stacked bar charts showing device counts by End-of-Sale and End-of-Life quarterly timeline, segmented by device type. A "Today" reference line highlights what's already past due. Includes a sortable/filterable table of overdue (past EoL) devices.

### 4. Proximity Analysis
DBSCAN-based spatial clustering of sites. Adjustable radius slider (1-25 miles) and minimum site count. Identifies clusters of nearby sites that could be refreshed together as coordinated projects. Shows cluster details with device counts, risk scores, and estimated costs.

### 5. Cost & Risk Analysis
- Total refresh cost breakdown by lifecycle status
- Average cost per device by type
- Risk score vs. refresh cost scatter plot (by site)
- Cost distribution by affiliate
- Cumulative investment timeline (cost ordered by EoL date)

### 6. Exception Management
Search devices by hostname or serial number. Toggle exception flags with reasons. Exceptions are persisted to `pipeline_output/exceptions.json` and respected in all downstream reporting — flagged devices are excluded from all charts and calculations.

### 7. Refresh Priorities
- Top 20 highest-risk sites ranked by average risk score
- Priority models ranked by a composite score (risk x log of count)
- Recommended 4-phase refresh schedule:
  - Phase 1 (Immediate): Past EoL devices
  - Phase 2 (Near-Term): Past EoS devices
  - Phase 3 (Planning): EoS within 1 year
  - Phase 4 (Strategic): In-scope but still current

---

## Automation

### Data Re-ingestion
To process a new data drop:

1. Replace `UAInnovateDataset-SoCo.xlsx` with the updated file (same filename)
2. Click "Refresh Data" in the dashboard sidebar, **or** re-run `python etl_pipeline.py`
3. The pipeline re-executes end-to-end and the dashboard picks up the new data

The pipeline can also be invoked programmatically:
```python
from etl_pipeline import run_pipeline
df, sheets = run_pipeline("path/to/new/UAInnovateDataset-SoCo.xlsx")
```

### Exception Persistence
User-flagged exceptions survive pipeline re-runs. They are stored separately in `pipeline_output/exceptions.json` and applied during each pipeline execution.

---

## Risk Scoring Methodology

The composite risk score (0-100) considers:

| Factor | Points |
|--------|--------|
| Past End-of-Life | 30 base + 5 per year past |
| Past End-of-Sale | 15-20 |
| EoS within 1 year | 10 |
| Flagged as "In Scope" for lifecycle | 10 |
| Has known replacement device | 5 |

Risk tiers:
- **Critical** (75-100): Immediate action needed
- **High** (50-75): Near-term refresh priority
- **Medium** (25-50): Plan and budget
- **Low** (0-25): Monitor

---

## Assumptions Beyond Specifications

1. **Port utilization data** from NA is preserved but not available for CatCtr/Prime sources
2. **"Unknown" lifecycle status** means no matching model in the ModelData sheet — these are assigned Low risk and zero refresh cost
3. **Proximity clustering** uses DBSCAN with haversine metric for geodesic accuracy
4. **Cost estimates** use material_cost + labor_cost from ModelData. Actual project costs may vary based on scope, scheduling, and volume discounts.
5. **5508 WLCs** have no replacement mapped per the glossary note "5508's will not be lifecycled" — they are included in inventory but have $0 refresh cost

---

## File Structure

```
Innovate/
├── UAInnovateDataset-SoCo.xlsx   # Source data
├── etl_pipeline.py               # ETL pipeline
├── app.py                        # Dashboard application
├── requirements.txt              # Python dependencies
├── run.bat                       # Windows launcher
├── run.sh                        # Unix launcher
├── README.md                     # This file
└── pipeline_output/
    ├── unified_devices.csv       # Processed dataset
    ├── pipeline_metadata.json    # Run metadata
    └── exceptions.json           # User exception flags
```
