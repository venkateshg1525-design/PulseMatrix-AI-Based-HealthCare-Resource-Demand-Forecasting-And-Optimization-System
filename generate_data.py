"""
generate_data.py - PulseMatrix AI Healthcare Dataset Engine
Generates:
  1. engineered_dataset.csv (20,000 hourly records ending on today's live date)
  2. engineered_dataset_backup.csv (Permanent baseline backup for 1-click rollback)
  3. week1_data.csv (168 hourly records - Mild Easing: Peak ICU ~133 beds)
  4. week2_data.csv (168 hourly records - Slight Surge: Peak ICU ~137 beds)
  5. week3_data.csv (168 hourly records - Stabilized: Peak ICU ~135 beds)
"""

import sys
import numpy as np
import pandas as pd

def generate_enterprise_dataset(output_path="engineered_dataset.csv", n_rows=20000):
    np.random.seed(42)

    # 1. Automatically detects today's live date and current hour
    end_time = pd.Timestamp.now().floor('h')
    timestamps = pd.date_range(end=end_time, periods=n_rows, freq="h")

    day_of_year = timestamps.dayofyear.values
    hour_of_day = timestamps.hour.values
    day_of_week = timestamps.dayofweek.values
    is_weekend = (day_of_week >= 5).astype(float)

    # 2. Seasonality & Circadian Diurnal Wave
    winter_phase = 2 * np.pi * (day_of_year - 25) / 365.25
    winter_wave = np.cos(winter_phase)
    winter_harmonic = 0.30 * np.cos(2 * winter_phase)
    annual_cycle = winter_wave + winter_harmonic
    diurnal_wave = np.sin(2 * np.pi * (hour_of_day - 8) / 24.0)

    # 3. Base Demand Metric Calculations
    flu_base = 20.0 + 8.0 * diurnal_wave
    clipped_cycle = np.maximum(0.0, annual_cycle - 0.15)
    flu_seasonal = 90.0 * np.power(clipped_cycle / 1.15, 1.6)
    flu_noise = np.random.normal(0, 3.5, n_rows)
    flu_cases = np.maximum(3.0, flu_base + flu_seasonal + flu_noise)

    icu_base = 48.0 + 6.0 * diurnal_wave + 0.28 * flu_cases
    icu_noise = np.random.normal(0, 2.5, n_rows)
    icu_patients = np.maximum(12.0, icu_base + icu_noise)

    oxygen_demand = 55.0 + 0.65 * icu_patients + 0.35 * flu_cases + np.random.normal(0, 3.0, n_rows)
    oxygen_demand = np.maximum(25.0, oxygen_demand)

    vent_ratio = 0.11 + 0.04 * np.clip(flu_cases / 120.0, 0.0, 1.0)
    ventilator_demand = vent_ratio * icu_patients + 0.05 * flu_cases + np.random.normal(0, 0.8, n_rows)
    ventilator_demand = np.maximum(1.0, ventilator_demand)

    staff_base = 30.0 + 0.40 * icu_patients + 0.14 * flu_cases + np.random.normal(0, 2.0, n_rows)
    weekend_factor = 1.0 - (0.20 * is_weekend)
    staff_demand = np.maximum(18.0, staff_base * weekend_factor)

    tanks_demand = 0.32 * oxygen_demand + 0.10 * icu_patients + np.random.normal(0, 2.0, n_rows)
    tanks_demand = np.maximum(6.0, tanks_demand)

    # 4. Crisis Injector (Final 14 Days = 336 Hours)
    crisis_hours = 14 * 24
    crisis_start_idx = n_rows - crisis_hours
    crisis_t = np.arange(crisis_hours)

    exp_scaling = (np.exp(3.0 * crisis_t / crisis_hours) - 1.0) / (np.exp(3.0) - 1.0)
    surge_multiplier = 1.0 + 2.2 * exp_scaling

    flu_cases[crisis_start_idx:] *= surge_multiplier
    icu_patients[crisis_start_idx:] *= (1.0 + 1.7 * exp_scaling)
    oxygen_demand[crisis_start_idx:] *= (1.0 + 1.8 * exp_scaling)
    ventilator_demand[crisis_start_idx:] *= (1.0 + 1.9 * exp_scaling)
    staff_demand[crisis_start_idx:] *= (1.0 + 1.2 * exp_scaling)
    tanks_demand[crisis_start_idx:] *= (1.0 + 1.6 * exp_scaling)

    # Baseline Main Dataset
    df = pd.DataFrame({
        "ds": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
        "icu_patients": np.round(icu_patients).astype(int),
        "flu_cases": np.round(flu_cases).astype(int),
        "oxygen_demand": np.round(oxygen_demand, 1),
        "ventilator_demand": np.round(ventilator_demand).astype(int),
        "staff_demand": np.round(staff_demand).astype(int),
        "tanks_demand": np.round(tanks_demand).astype(int)
    })

    # Save active dataset and permanent backup for 1-click rollback
    df.to_csv(output_path, index=False)
    df.to_csv("engineered_dataset_backup.csv", index=False)
    print(f"Main Dataset Generated: {output_path} ({len(df)} records up to {end_time})")
    print("Backup Dataset Created: engineered_dataset_backup.csv")

    # 5. Generate 3 Weekly Telemetry Files (168 rows each)
    last_ts = timestamps[-1]
    
    # Week 1: Mild Easing (Peak ICU ~133)
    t_w1 = pd.date_range(end=last_ts, periods=168, freq="h")
    w1_df = pd.DataFrame({
        "ds": t_w1.strftime("%Y-%m-%d %H:%M:%S"),
        "icu_patients": np.clip(np.random.normal(132.8, 1.0, 168).round(), 131, 134).astype(int),
        "flu_cases": np.clip(np.random.normal(55.0, 2.0, 168).round(), 50, 62).astype(int),
        "oxygen_demand": np.round(np.random.normal(238.0, 3.0, 168), 1),
        "ventilator_demand": np.clip(np.random.normal(18.5, 0.8, 168).round(), 17, 20).astype(int),
        "staff_demand": np.clip(np.random.normal(92.0, 2.0, 168).round(), 88, 96).astype(int),
        "tanks_demand": np.clip(np.random.normal(78.0, 2.0, 168).round(), 74, 84).astype(int)
    })
    w1_df.to_csv("week1_data.csv", index=False)
    print("Generated week1_data.csv (168 rows) - Targets ~133 ICU Beds")

    # Week 2: Slight Surge (Peak ICU ~137)
    w2_df = pd.DataFrame({
        "ds": t_w1.strftime("%Y-%m-%d %H:%M:%S"),
        "icu_patients": np.clip(np.random.normal(136.8, 1.0, 168).round(), 135, 138).astype(int),
        "flu_cases": np.clip(np.random.normal(65.0, 2.0, 168).round(), 58, 70).astype(int),
        "oxygen_demand": np.round(np.random.normal(248.0, 3.0, 168), 1),
        "ventilator_demand": np.clip(np.random.normal(20.5, 0.8, 168).round(), 19, 22).astype(int),
        "staff_demand": np.clip(np.random.normal(98.0, 2.0, 168).round(), 94, 102).astype(int),
        "tanks_demand": np.clip(np.random.normal(84.0, 2.0, 168).round(), 80, 89).astype(int)
    })
    w2_df.to_csv("week2_data.csv", index=False)
    print("Generated week2_data.csv (168 rows) - Targets ~137 ICU Beds")

    # Week 3: Stabilized (Peak ICU ~135)
    w3_df = pd.DataFrame({
        "ds": t_w1.strftime("%Y-%m-%d %H:%M:%S"),
        "icu_patients": np.clip(np.random.normal(134.8, 1.0, 168).round(), 133, 136).astype(int),
        "flu_cases": np.clip(np.random.normal(60.0, 2.0, 168).round(), 54, 65).astype(int),
        "oxygen_demand": np.round(np.random.normal(242.0, 3.0, 168), 1),
        "ventilator_demand": np.clip(np.random.normal(19.5, 0.8, 168).round(), 18, 21).astype(int),
        "staff_demand": np.clip(np.random.normal(95.0, 2.0, 168).round(), 90, 99).astype(int),
        "tanks_demand": np.clip(np.random.normal(81.0, 2.0, 168).round(), 77, 86).astype(int)
    })
    w3_df.to_csv("week3_data.csv", index=False)
    print("Generated week3_data.csv (168 rows) - Targets ~135 ICU Beds")

if __name__ == "__main__":
    generate_enterprise_dataset()