"""
Southern Company Network Equipment Lifecycle ETL Pipeline
=========================================================
Ingests UAInnovateDataset-SoCo.xlsx, applies cleaning/transformation rules,
and outputs a unified dataset for the dashboard.

Data Flow:
  Raw Excel -> Read Sheets -> Filter Active -> Normalize Types ->
  Deduplicate (CatCtr/Prime override NA) -> Parse Hostname (State/Site) ->
  Join Location + Model Data -> Compute Risk Scores -> Unified Output
"""

import os
import re
import json
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "UAInnovateDataset-SoCo.xlsx"
OUTPUT_DIR = BASE_DIR / "pipeline_output"
EXCEPTIONS_FILE = OUTPUT_DIR / "exceptions.json"

TODAY = pd.Timestamp.now().normalize()

# Device type normalization map (NA types -> canonical)
DEVICE_TYPE_MAP = {
    "Router": "Router",
    "L3Switch": "Switch",
    "Switch": "Switch",
    "Voice Gateway": "Voice Gateway",
    "Application Switch": "Switch",
    "Virtual Switch": "Switch",
    "Wireless Controller": "Wireless LAN Controller",
    "WirelessLC": "Wireless LAN Controller",
    "Firewall": "Firewall",
    "Virtual Firewall": "Firewall",
}

# CatCtr family normalization
CATCTR_FAMILY_MAP = {
    "Unified AP": "Access Point",
    "Switches and Hubs": "Switch",
    "Routers": "Router",
    "Wireless Controller": "Wireless LAN Controller",
    "Managed": "Switch",
}


def load_raw_data(filepath=DATA_FILE):
    """Load all worksheets from the Excel file."""
    print(f"[ETL] Loading data from {filepath}")
    xls = pd.ExcelFile(filepath)
    sheets = {}
    for name in xls.sheet_names:
        sheets[name] = pd.read_excel(xls, sheet_name=name)
        print(f"  -> {name}: {sheets[name].shape[0]} rows, {sheets[name].shape[1]} cols")
    return sheets


def get_decom_sites(sheets):
    """Get set of site codes scheduled for decommission."""
    decom = sheets["Decom"]
    return set(decom["Site Cd"].str.strip().str.upper())


def parse_hostname(hostname):
    """Extract state code (first 2 chars) and site code (next 3 chars) from hostname."""
    if pd.isna(hostname) or len(str(hostname)) < 5:
        return None, None
    h = str(hostname).strip().upper()
    # Remove domain suffixes
    h = h.split(".")[0]
    state_code = h[:2]
    site_code = h[2:5]
    return state_code, site_code


def expand_serial_stacks(df, serial_col="serial_number"):
    """Expand comma-separated serial numbers into separate rows (switch stacks / HA pairs)."""
    rows = []
    for _, row in df.iterrows():
        serials = str(row[serial_col]).split(",") if pd.notna(row[serial_col]) else [row[serial_col]]
        for s in serials:
            new_row = row.copy()
            new_row[serial_col] = str(s).strip() if pd.notna(s) else s
            rows.append(new_row)
    return pd.DataFrame(rows)


def process_na(sheets):
    """Process NA (Network Automation) data."""
    na = sheets["NA"].copy()

    # Filter: only active devices
    na = na[na["Device Status"] == "Active"].copy()

    # Filter: only Cisco (already all Cisco per data)
    # Filter out Firewalls (out of scope per glossary)
    na = na[~na["Device Type"].isin(["Firewall", "Virtual Firewall"])].copy()

    # Normalize device types
    na["device_type_normalized"] = na["Device Type"].map(DEVICE_TYPE_MAP)

    # Remove wireless devices from NA (CatCtr/Prime are source of truth)
    na = na[~na["device_type_normalized"].isin(["Wireless LAN Controller", "Access Point"])].copy()

    # Parse hostname
    parsed = na["Host Name"].apply(lambda h: pd.Series(parse_hostname(h), index=["state_code", "site_code"]))
    na = pd.concat([na, parsed], axis=1)

    # Expand switch stacks (comma-separated serials)
    na["serial_number"] = na["Serial Number"]
    na = expand_serial_stacks(na, "serial_number")

    # Standardize columns
    na_out = na.rename(columns={
        "Host Name": "hostname",
        "Device IP": "ip_address",
        "Device Model": "model",
        "Software Version": "software_version",
        "Free Ports": "free_ports",
        "Total Ports": "total_ports",
        "Ports In Use": "ports_in_use",
    })[["hostname", "ip_address", "device_type_normalized", "model", "serial_number",
        "software_version", "free_ports", "total_ports", "ports_in_use",
        "state_code", "site_code"]].copy()
    na_out.rename(columns={"device_type_normalized": "device_type"}, inplace=True)
    na_out["source"] = "NA"

    print(f"[ETL] NA processed: {len(na_out)} devices")
    return na_out


def process_catctr(sheets):
    """Process Catalyst Center data."""
    cat = sheets["CatCtr"].copy()

    # Filter: reachable only
    cat = cat[cat["reachabilityStatus"] == "Reachable"].copy()

    # Normalize family
    cat["device_type"] = cat["family"].map(CATCTR_FAMILY_MAP)
    cat = cat[cat["device_type"].notna()].copy()

    # Only keep in-scope types (AP, WLC, Switch, Router)
    # Exclude hostnames starting with "AP" for access points per glossary
    ap_mask = (cat["device_type"] == "Access Point") & (cat["hostname"].str.upper().str.startswith("AP", na=False))
    cat = cat[~ap_mask].copy()

    # Parse hostname (strip domain)
    def parse_cat_hostname(h):
        if pd.isna(h):
            return None, None
        h_clean = str(h).split(".")[0].strip().upper()
        return parse_hostname(h_clean)

    parsed = cat["hostname"].apply(lambda h: pd.Series(parse_cat_hostname(h), index=["state_code", "site_code"]))
    cat = pd.concat([cat, parsed], axis=1)

    # Expand serial stacks
    cat["serial_number"] = cat["serialNumber"]
    cat = expand_serial_stacks(cat, "serial_number")

    cat_out = cat.rename(columns={
        "hostname": "hostname_raw",
        "platformId": "model",
        "softwareVersion": "software_version",
    }).copy()
    cat_out["hostname"] = cat_out["hostname_raw"].apply(lambda h: str(h).split(".")[0].strip().upper() if pd.notna(h) else h)
    cat_out["ip_address"] = cat_out["dnsResolvedManagementAddress"]
    cat_out["free_ports"] = np.nan
    cat_out["total_ports"] = np.nan
    cat_out["ports_in_use"] = np.nan
    cat_out["source"] = "CatCtr"

    cat_out = cat_out[["hostname", "ip_address", "device_type", "model", "serial_number",
                        "software_version", "free_ports", "total_ports", "ports_in_use",
                        "state_code", "site_code", "source"]].copy()

    print(f"[ETL] CatCtr processed: {len(cat_out)} devices")
    return cat_out


def process_prime_ap(sheets):
    """Process Prime AP data."""
    ap = sheets["PrimeAP"].copy()

    # Filter: active (upTime > 0)
    ap = ap[ap["upTime"] > 0].copy()

    # Exclude names starting with "AP"
    ap = ap[~ap["name"].str.upper().str.startswith("AP", na=False)].copy()

    parsed = ap["name"].apply(lambda h: pd.Series(parse_hostname(h), index=["state_code", "site_code"]))
    ap = pd.concat([ap, parsed], axis=1)

    ap_out = ap.rename(columns={
        "name": "hostname",
        "ipAddress": "ip_address",
        "model": "model",
        "serialNumber": "serial_number",
        "softwareVersion": "software_version",
    }).copy()
    ap_out["device_type"] = "Access Point"
    ap_out["free_ports"] = np.nan
    ap_out["total_ports"] = np.nan
    ap_out["ports_in_use"] = np.nan
    ap_out["source"] = "PrimeAP"

    ap_out = ap_out[["hostname", "ip_address", "device_type", "model", "serial_number",
                      "software_version", "free_ports", "total_ports", "ports_in_use",
                      "state_code", "site_code", "source"]].copy()

    print(f"[ETL] PrimeAP processed: {len(ap_out)} devices")
    return ap_out


def process_prime_wlc(sheets):
    """Process Prime WLC data."""
    wlc = sheets["PrimeWLC"].copy()

    # Filter: reachable
    wlc = wlc[wlc["reachability"].str.upper() == "REACHABLE"].copy()

    parsed = wlc["deviceName"].apply(lambda h: pd.Series(parse_hostname(h), index=["state_code", "site_code"]))
    wlc = pd.concat([wlc, parsed], axis=1)

    wlc_out = wlc.rename(columns={
        "deviceName": "hostname",
        "ipAddress": "ip_address",
        "manufacturer_part_partNumber": "model",
        "manufacturer_part_serialNumber": "serial_number",
        "softwareVersion": "software_version",
    }).copy()
    wlc_out["device_type"] = "Wireless LAN Controller"
    wlc_out["free_ports"] = np.nan
    wlc_out["total_ports"] = np.nan
    wlc_out["ports_in_use"] = np.nan
    wlc_out["source"] = "PrimeWLC"

    wlc_out = wlc_out[["hostname", "ip_address", "device_type", "model", "serial_number",
                         "software_version", "free_ports", "total_ports", "ports_in_use",
                         "state_code", "site_code", "source"]].copy()

    print(f"[ETL] PrimeWLC processed: {len(wlc_out)} devices")
    return wlc_out


def deduplicate_sources(na_df, catctr_df, prime_ap_df, prime_wlc_df):
    """
    CatCtr/Prime are authoritative for APs and WLCs.
    For APs: CatCtr is primary, Prime fills gaps (devices not in CatCtr).
    For WLCs: same logic.
    NA handles switches, routers, voice gateways.
    """
    # CatCtr APs
    catctr_aps = catctr_df[catctr_df["device_type"] == "Access Point"]
    catctr_wlcs = catctr_df[catctr_df["device_type"] == "Wireless LAN Controller"]
    catctr_other = catctr_df[~catctr_df["device_type"].isin(["Access Point", "Wireless LAN Controller"])]

    # Prime APs not already in CatCtr (by hostname)
    catctr_ap_hostnames = set(catctr_aps["hostname"].str.upper())
    prime_ap_unique = prime_ap_df[~prime_ap_df["hostname"].str.upper().isin(catctr_ap_hostnames)]

    # Prime WLCs not already in CatCtr
    catctr_wlc_hostnames = set(catctr_wlcs["hostname"].str.upper())
    prime_wlc_unique = prime_wlc_df[~prime_wlc_df["hostname"].str.upper().isin(catctr_wlc_hostnames)]

    combined = pd.concat([
        na_df,
        catctr_aps,
        catctr_wlcs,
        catctr_other,
        prime_ap_unique,
        prime_wlc_unique,
    ], ignore_index=True)

    print(f"[ETL] After deduplication: {len(combined)} devices")
    return combined


def join_location_data(devices, sheets):
    """Join SOLID and SOLID-Loc data based on site_code."""
    solid = sheets["SOLID"].copy()
    solid_loc = sheets["SOLID-Loc"].copy()

    # Merge SOLID and SOLID-Loc on Site Code
    location = solid.merge(solid_loc, on="Site Code", how="outer", suffixes=("", "_loc"))
    location["Site Code"] = location["Site Code"].str.strip().str.upper()

    # Rename for join
    location = location.rename(columns={
        "Site Code": "site_code",
        "Site Name": "site_name",
        "Street Address 1": "address",
        "City": "city",
        "State": "state",
        "Zip": "zip_code",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "PhysicalAddressCounty": "county",
        "Call Group": "call_group",
        "Owner": "affiliate",
    })

    devices["site_code"] = devices["site_code"].str.strip().str.upper()
    merged = devices.merge(
        location[["site_code", "site_name", "address", "city", "state", "zip_code",
                   "latitude", "longitude", "county", "call_group", "affiliate"]],
        on="site_code",
        how="left"
    )

    # Fill state from hostname state_code where missing
    state_map = {
        "AL": "AL", "GA": "GA", "MS": "MS", "FL": "FL", "IL": "IL",
        "CA": "CA", "TX": "TX", "VA": "VA", "NC": "NC", "OK": "OK",
        "TN": "TN", "NJ": "NJ", "SC": "SC", "NV": "NV", "KY": "KY",
        "ME": "ME", "CO": "CO", "WA": "WA", "DC": "DC", "NY": "NY",
        "PA": "PA", "OH": "OH", "WI": "WI", "IN": "IN", "MI": "MI",
        "MN": "MN", "MO": "MO", "LA": "LA", "AR": "AR", "MD": "MD",
    }
    merged["state"] = merged["state"].fillna(merged["state_code"].map(state_map))

    print(f"[ETL] After location join: {len(merged)} devices, {merged['latitude'].notna().sum()} geocoded")
    return merged


def join_model_data(devices, sheets):
    """Join ModelData for EoS/EoL dates, costs, and lifecycle info."""
    md = sheets["ModelData"].copy()

    md = md.rename(columns={
        "Model": "model_key",
        "Model Parent": "model_parent",
        "EoS": "eos_date",
        "EoL": "eol_date",
        "Category": "model_category",
        "In Scope?": "in_scope",
        "Repl Device": "replacement_device",
        "Device Cost": "device_cost",
        "DNA Cost": "dna_cost",
        "Staging Cost": "staging_cost",
        "Labor Cost": "labor_cost",
        "Material Cost": "material_cost",
        "Tax&OH": "tax_overhead",
        "DE Hrs": "de_hours",
        "SE Hrs": "se_hours",
        "FOT Hrs": "fot_hours",
    })

    model_cols = ["model_key", "model_parent", "eos_date", "eol_date", "model_category",
                  "in_scope", "replacement_device", "device_cost", "dna_cost", "staging_cost",
                  "labor_cost", "material_cost", "tax_overhead", "de_hours", "se_hours", "fot_hours"]

    merged = devices.merge(
        md[model_cols],
        left_on="model",
        right_on="model_key",
        how="left"
    )

    print(f"[ETL] After model join: {merged['eos_date'].notna().sum()} devices have EoS dates")
    return merged


def compute_lifecycle_status(df):
    """Compute lifecycle risk status for each device."""
    df = df.copy()

    # Lifecycle status
    conditions = [
        df["eol_date"].notna() & (df["eol_date"] <= TODAY),
        df["eos_date"].notna() & (df["eos_date"] <= TODAY) & (df["eol_date"] > TODAY),
        df["eos_date"].notna() & (df["eos_date"] > TODAY) & (df["eos_date"] <= TODAY + pd.DateOffset(years=1)),
        df["eos_date"].notna() & (df["eos_date"] > TODAY + pd.DateOffset(years=1)),
    ]
    choices = ["Past EoL", "Past EoS", "EoS within 1yr", "Current"]
    df["lifecycle_status"] = np.select(conditions, choices, default="Unknown")

    # Risk score (0-100)
    def calc_risk(row):
        score = 0
        if pd.notna(row["eol_date"]):
            if row["eol_date"] <= TODAY:
                years_past = (TODAY - row["eol_date"]).days / 365.25
                score += min(50, 30 + years_past * 5)
            elif row["eos_date"] is not pd.NaT and row["eos_date"] <= TODAY:
                score += 20
        if pd.notna(row["eos_date"]):
            if row["eos_date"] <= TODAY:
                score += 15
            elif row["eos_date"] <= TODAY + pd.DateOffset(years=1):
                score += 10
        if row.get("in_scope") == "Yes":
            score += 10
        # Higher risk for devices with known replacement
        if pd.notna(row.get("replacement_device")):
            score += 5
        return min(100, score)

    df["risk_score"] = df.apply(calc_risk, axis=1)

    # Risk tier
    df["risk_tier"] = pd.cut(df["risk_score"], bins=[-1, 25, 50, 75, 100],
                              labels=["Low", "Medium", "High", "Critical"])

    print(f"[ETL] Lifecycle status distribution:")
    print(df["lifecycle_status"].value_counts().to_string())

    return df


def filter_decom_sites(df, decom_sites):
    """Mark devices at decommissioned sites."""
    df = df.copy()
    df["is_decom_site"] = df["site_code"].isin(decom_sites)
    pre = len(df)
    df = df[~df["is_decom_site"]].copy()
    print(f"[ETL] Filtered {pre - len(df)} devices at decom sites")
    return df


def load_exceptions():
    """Load user-defined exceptions."""
    if EXCEPTIONS_FILE.exists():
        with open(EXCEPTIONS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_exceptions(exceptions):
    """Save exceptions to file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXCEPTIONS_FILE, "w") as f:
        json.dump(exceptions, f, indent=2)


def apply_exceptions(df):
    """Apply exception flags from persisted file."""
    exceptions = load_exceptions()
    df = df.copy()
    df["exception_flagged"] = False
    df["exception_reason"] = ""
    for key, info in exceptions.items():
        mask = df["serial_number"] == key
        if mask.any():
            df.loc[mask, "exception_flagged"] = True
            df.loc[mask, "exception_reason"] = info.get("reason", "")
    flagged = df["exception_flagged"].sum()
    print(f"[ETL] Applied {flagged} exception flags")
    return df


def generate_device_id(df):
    """Generate unique device ID."""
    df = df.copy()
    df["device_id"] = df.apply(
        lambda r: hashlib.md5(
            f"{r.get('hostname','')}-{r.get('serial_number','')}-{r.get('source','')}".encode()
        ).hexdigest()[:12],
        axis=1
    )
    return df


def compute_refresh_cost(df):
    """Compute total refresh cost per device."""
    df = df.copy()
    df["total_refresh_cost"] = df["material_cost"].fillna(0) + df["labor_cost"].fillna(0)
    return df


def run_pipeline(filepath=DATA_FILE):
    """Execute the full ETL pipeline."""
    print("=" * 60)
    print("Southern Company Network Equipment Lifecycle ETL Pipeline")
    print("=" * 60)
    print(f"Run timestamp: {datetime.now().isoformat()}")
    print(f"Data file: {filepath}")
    print()

    # 1. Load raw data
    sheets = load_raw_data(filepath)

    # 2. Get decom sites
    decom_sites = get_decom_sites(sheets)
    print(f"[ETL] Decom sites: {decom_sites}")

    # 3. Process each source
    na_df = process_na(sheets)
    catctr_df = process_catctr(sheets)
    prime_ap_df = process_prime_ap(sheets)
    prime_wlc_df = process_prime_wlc(sheets)

    # 4. Deduplicate
    unified = deduplicate_sources(na_df, catctr_df, prime_ap_df, prime_wlc_df)

    # 5. Filter decom sites
    unified = filter_decom_sites(unified, decom_sites)

    # 6. Join location data
    unified = join_location_data(unified, sheets)

    # 7. Join model/lifecycle data
    unified = join_model_data(unified, sheets)

    # 8. Compute lifecycle status and risk
    unified = compute_lifecycle_status(unified)

    # 9. Compute refresh costs
    unified = compute_refresh_cost(unified)

    # 10. Apply exceptions
    unified = apply_exceptions(unified)

    # 11. Generate device IDs
    unified = generate_device_id(unified)

    # 12. Ensure string columns are clean for serialization
    for col in unified.select_dtypes(include=["object"]).columns:
        unified[col] = unified[col].astype(str).replace("nan", pd.NA).replace("None", pd.NA)

    # 13. Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "unified_devices.csv"
    unified.to_csv(csv_path, index=False)

    # Save metadata
    meta = {
        "run_timestamp": datetime.now().isoformat(),
        "source_file": str(filepath),
        "total_devices": len(unified),
        "by_source": unified["source"].value_counts().to_dict(),
        "by_device_type": unified["device_type"].value_counts().to_dict(),
        "by_lifecycle_status": unified["lifecycle_status"].value_counts().to_dict(),
        "geocoded_count": int(unified["latitude"].notna().sum()),
        "total_refresh_cost": float(unified["total_refresh_cost"].sum()),
    }
    with open(OUTPUT_DIR / "pipeline_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print()
    print("=" * 60)
    print(f"Pipeline complete! {len(unified)} devices processed.")
    print(f"Output: {csv_path}")
    print("=" * 60)

    return unified, sheets


if __name__ == "__main__":
    run_pipeline()
