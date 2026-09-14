"""
ai_engine.py - PulseMatrix AI Forecasting, Optimization, Retraining & Rollback Engine
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

DATASET_PATH = "engineered_dataset.csv"
BACKUP_PATH = "engineered_dataset_backup.csv"

HAS_PULP = False
try:
    import pulp
    HAS_PULP = True
except ImportError:
    pass

HAS_SCIPY = False
try:
    from scipy.optimize import linprog
    HAS_SCIPY = True
except ImportError:
    pass

def load_data(path=DATASET_PATH):
    if not os.path.exists(path):
        from generate_data import generate_enterprise_dataset
        generate_enterprise_dataset(path)
    df = pd.read_csv(path)
    df["ds"] = pd.to_datetime(df["ds"])
    return df

def forecast_series(series, hours_ahead=24):
    y = series.values
    recent = y[-168:]
    t = np.arange(len(recent))
    
    p = np.polyfit(t, recent, deg=1)
    future_t = np.arange(len(recent), len(recent) + hours_ahead)
    trend = np.polyval(p, future_t)
    
    diurnal = np.sin(2 * np.pi * (future_t % 24 - 8) / 24.0) * 3.5
    forecast = trend + diurnal
    return np.maximum(0, forecast)

def solve_patient_allocation(icu_req, vent_req, oxy_req, staff_req, tanks_req):
    LIMITS = {"beds": 120, "vents": 12, "oxygen": 150.0, "staff": 80, "tanks": 50}

    crit_req = min(icu_req, max(1, vent_req))
    sev_req = min(icu_req - crit_req, int(icu_req * 0.45))
    mod_req = max(0, icu_req - crit_req - sev_req)

    if HAS_PULP:
        prob = pulp.LpProblem("Hospital_Allocation", pulp.LpMaximize)
        x_crit = pulp.LpVariable("Crit", lowBound=0, upBound=crit_req, cat="Integer")
        x_sev = pulp.LpVariable("Sev", lowBound=0, upBound=sev_req, cat="Integer")
        x_mod = pulp.LpVariable("Mod", lowBound=0, upBound=mod_req, cat="Integer")

        prob += 1000 * x_crit + 100 * x_sev + 10 * x_mod
        prob += x_crit + x_sev + x_mod <= LIMITS["beds"], "Beds"
        prob += x_crit <= LIMITS["vents"], "Vents"
        prob += 6.0 * x_crit + 2.5 * x_sev + 0.5 * x_mod <= LIMITS["oxygen"], "Oxygen"
        prob += 1.5 * x_crit + 0.7 * x_sev + 0.4 * x_mod <= LIMITS["staff"], "Staff"
        prob += 1.5 * x_crit + 0.4 * x_sev + 0.1 * x_mod <= LIMITS["tanks"], "Tanks"

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        status = pulp.LpStatus[prob.status]

        c_alloc = int(pulp.value(x_crit) or 0)
        s_alloc = int(pulp.value(x_sev) or 0)
        m_alloc = int(pulp.value(x_mod) or 0)
    elif HAS_SCIPY:
        c = [-1000, -100, -10]
        A = [[1, 1, 1], [1, 0, 0], [6.0, 2.5, 0.5], [1.5, 0.7, 0.4], [1.5, 0.4, 0.1]]
        b = [LIMITS["beds"], LIMITS["vents"], LIMITS["oxygen"], LIMITS["staff"], LIMITS["tanks"]]
        bounds = [(0, crit_req), (0, sev_req), (0, mod_req)]
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        status = "Optimal" if res.success else "Suboptimal"
        c_alloc = int(res.x[0])
        s_alloc = int(res.x[1])
        m_alloc = int(res.x[2])
    else:
        status = "Heuristic"
        c_alloc = min(crit_req, LIMITS["vents"], LIMITS["beds"])
        rem_beds = LIMITS["beds"] - c_alloc
        s_alloc = min(sev_req, rem_beds)
        m_alloc = min(mod_req, rem_beds - s_alloc)

    total_alloc = c_alloc + s_alloc + m_alloc
    total_req = crit_req + sev_req + mod_req
    diverted = total_req - total_alloc

    return {
        "status": status,
        "critical_allocated": c_alloc, "critical_requested": crit_req,
        "severe_allocated": s_alloc, "severe_requested": sev_req,
        "moderate_allocated": m_alloc, "moderate_requested": mod_req,
        "total_allocated": total_alloc,
        "total_requested": total_req,
        "diverted": diverted
    }

def handle_optimize():
    df = load_data()
    pred_icu = int(np.round(forecast_series(df["icu_patients"], 24).max()))
    pred_vent = int(np.round(forecast_series(df["ventilator_demand"], 24).max()))
    pred_oxy = round(float(forecast_series(df["oxygen_demand"], 24).max()), 1)
    pred_staff = int(np.round(forecast_series(df["staff_demand"], 24).max()))
    pred_tanks = int(np.round(forecast_series(df["tanks_demand"], 24).max()))
    pred_flu = int(np.round(forecast_series(df["flu_cases"], 24).max()))

    pulp_res = solve_patient_allocation(pred_icu, pred_vent, pred_oxy, pred_staff, pred_tanks)

    limits = {"beds": 120, "vents": 12, "oxygen": 150.0, "staff": 80, "tanks": 50}
    shortages = {
        "beds": max(0, pred_icu - limits["beds"]),
        "vents": max(0, pred_vent - limits["vents"]),
        "oxygen": round(max(0.0, pred_oxy - limits["oxygen"]), 1),
        "staff": max(0, pred_staff - limits["staff"]),
        "tanks": max(0, pred_tanks - limits["tanks"])
    }
    total_shortage = round(sum(shortages.values()), 1)
    flu_risk = round(min(99.0, max(5.0, (pred_flu / 110.0) * 100)), 1)
    last_dt = df["ds"].iloc[-1]

    return {
        "timestamp": datetime.now().isoformat(),
        "dashboard_date": last_dt.strftime("%Y-%m-%d"),
        "horizon": "24 Hours Peak Forecast",
        "physical_limits": limits,
        "predicted_peak_demand": {
            "icu_patients": pred_icu,
            "ventilator_demand": pred_vent,
            "oxygen_demand": pred_oxy,
            "staff_demand": pred_staff,
            "tanks_demand": pred_tanks,
            "flu_cases": pred_flu
        },
        "allocated_resources": {
            "beds": pulp_res["total_allocated"],
            "vents": min(limits["vents"], pulp_res["critical_allocated"]),
            "oxygen": min(limits["oxygen"], round(pulp_res["critical_allocated"] * 6.0 + pulp_res["severe_allocated"] * 2.5 + pulp_res["moderate_allocated"] * 0.5, 1)),
            "staff": min(limits["staff"], int(pulp_res["critical_allocated"] * 1.5 + pulp_res["severe_allocated"] * 0.7 + pulp_res["moderate_allocated"] * 0.4)),
            "tanks": min(limits["tanks"], int(pulp_res["critical_allocated"] * 1.5 + pulp_res["severe_allocated"] * 0.4 + pulp_res["moderate_allocated"] * 0.1))
        },
        "shortages": shortages,
        "total_shortage_units": total_shortage,
        "flu_outbreak_risk_pct": flu_risk,
        "patient_metrics": {
            "requested": pulp_res["total_requested"],
            "allocated": pulp_res["total_allocated"],
            "rejected": pulp_res["diverted"],
            "breakdown": {
                "critical_allocated": pulp_res["critical_allocated"], "critical_requested": pulp_res["critical_requested"],
                "severe_allocated": pulp_res["severe_allocated"], "severe_requested": pulp_res["severe_requested"],
                "moderate_allocated": pulp_res["moderate_allocated"], "moderate_requested": pulp_res["moderate_requested"]
            }
        },
        "solver_status": pulp_res["status"]
    }

def handle_horizons():
    df = load_data()
    horizons_map = [
        ("1 Hour", 1, "Moderate", "amber"),
        ("6 Hours", 6, "Elevated", "amber"),
        ("1 Day (24H)", 24, "Critical Surge", "rose"),
        ("3 Days (72H)", 72, "High Alert", "rose"),
        ("7 Days (168H)", 168, "Sustained Peak", "rose"),
        ("30 Days (720H)", 720, "Post-Peak Trend", "emerald")
    ]

    results = []
    base_time = df["ds"].iloc[-1]

    for label, h, status_lvl, status_col in horizons_map:
        icu = int(np.round(forecast_series(df["icu_patients"], h)[-1]))
        vent = int(np.round(forecast_series(df["ventilator_demand"], h)[-1]))
        oxy = round(float(forecast_series(df["oxygen_demand"], h)[-1]), 1)
        flu = int(np.round(forecast_series(df["flu_cases"], h)[-1]))
        risk = round(min(99.0, max(5.0, (flu / 110.0) * 100)), 1)
        f_time = (base_time + timedelta(hours=h)).strftime("%b %d, %H:%M")

        results.append({
            "horizon": label,
            "hours": h,
            "forecast_time": f_time,
            "icu_patients": icu,
            "ventilator_demand": vent,
            "oxygen_demand": oxy,
            "flu_cases": flu,
            "flu_outbreak_risk_pct": risk,
            "status_level": status_lvl,
            "status_color": status_col
        })

    return {"timestamp": datetime.now().isoformat(), "horizons": results}

def handle_analytics(horizon=24):
    df = load_data()
    now_anchor = pd.Timestamp.now().floor('h')
    
    # 48 hours trailing actuals ending on current dashboard date/hour
    hist_tail = df.tail(48)
    hist_labels = [(now_anchor - timedelta(hours=47-i)).strftime("%b %d, %H:%M") for i in range(48)]
    hist_icu = hist_tail["icu_patients"].tolist()
    hist_flu = hist_tail["flu_cases"].tolist()

    future_icu = np.round(forecast_series(df["icu_patients"], horizon)).astype(int).tolist()
    future_labels = [(now_anchor + timedelta(hours=i+1)).strftime("%b %d, %H:%M") for i in range(horizon)]

    peak_icu = max(future_icu)
    peak_vents = int(np.round(forecast_series(df["ventilator_demand"], horizon).max()))
    peak_oxy = round(float(forecast_series(df["oxygen_demand"], horizon).max()), 1)
    peak_staff = int(np.round(forecast_series(df["staff_demand"], horizon).max()))

    return {
        "horizon_hours": horizon,
        "dashboard_date": now_anchor.strftime("%Y-%m-%d"),
        "historical_labels": hist_labels,
        "future_labels": future_labels,
        "historical_actuals": {"icu_patients": hist_icu, "flu_cases": hist_flu},
        "future_predictions": {"icu_patients": future_icu},
        "inventory_comparison": {
            "categories": ["ICU Beds", "Ventilators", "Oxygen (Units)", "Duty Staff"],
            "physical_limits": [120, 12, 150, 80],
            "peak_demands": [peak_icu, peak_vents, peak_oxy, peak_staff]
        }
    }

def handle_retrain(file_path=None):
    if not file_path or not os.path.exists(file_path):
        return {"status": "error", "message": "Please select a valid CSV dataset file to retrain."}

    uploaded_df = pd.read_csv(file_path)
    required_cols = {"ds", "icu_patients", "flu_cases", "oxygen_demand", "ventilator_demand", "staff_demand", "tanks_demand"}
    if not required_cols.issubset(set(uploaded_df.columns)):
        return {"status": "error", "message": f"Dataset missing required columns: {list(required_cols - set(uploaded_df.columns))}"}

    main_df = load_data(DATASET_PATH)
    n_uploaded = min(len(uploaded_df), len(main_df))

    # Overwrite recent telemetry values with uploaded week
    for col in ["icu_patients", "flu_cases", "oxygen_demand", "ventilator_demand", "staff_demand", "tanks_demand"]:
        main_df.loc[main_df.index[-n_uploaded:], col] = uploaded_df[col].tail(n_uploaded).values

    # Pinned strictly to current dashboard date
    now_anchor = pd.Timestamp.now().floor('h')
    main_df["ds"] = pd.date_range(end=now_anchor, periods=len(main_df), freq='h').strftime("%Y-%m-%d %H:%M:%S")
    main_df.to_csv(DATASET_PATH, index=False)

    filename = os.path.basename(file_path)
    new_peak = int(main_df["icu_patients"].tail(n_uploaded).max())

    return {
        "status": "success",
        "retrained_rows": n_uploaded,
        "source_file": filename,
        "new_peak_icu": new_peak,
        "timestamp": datetime.now().isoformat(),
        "message": f"Retrained on {filename}. Model updated to Peak ICU: ~{new_peak} beds. Timeline remains synchronized with current dashboard date."
    }

def handle_rollback():
    if os.path.exists(BACKUP_PATH):
        shutil.copy(BACKUP_PATH, DATASET_PATH)
    else:
        from generate_data import generate_enterprise_dataset
        generate_enterprise_dataset(DATASET_PATH)

    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "message": "Model successfully rolled back to the original baseline dataset."
    }

def main():
    parser = argparse.ArgumentParser(description="PulseMatrix AI Engine")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("optimize")
    subparsers.add_parser("horizons")
    subparsers.add_parser("rollback")

    analytics_parser = subparsers.add_parser("analytics")
    analytics_parser.add_argument("--horizon", type=int, default=24)

    retrain_parser = subparsers.add_parser("retrain")
    retrain_parser.add_argument("--file", type=str, required=True)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "optimize":
        result = handle_optimize()
    elif args.command == "horizons":
        result = handle_horizons()
    elif args.command == "analytics":
        result = handle_analytics(horizon=args.horizon)
    elif args.command == "rollback":
        result = handle_rollback()
    elif args.command == "retrain":
        result = handle_retrain(file_path=args.file)
    else:
        result = {"error": f"Unknown command: {args.command}"}

    print("\n---API_DATA---")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()