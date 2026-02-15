import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

folder_path = "./data_thes"

print("Finding Excel files in", folder_path)
excel_files = [
	f for f in os.listdir(folder_path)
	if f.lower().endswith((".xlsx", ".xls"))
]

if not excel_files:
	raise FileNotFoundError(f"No Excel files found in {folder_path}")

excel_files.sort()

all_dataframes = []
for filename in excel_files:
	file_path = os.path.join(folder_path, filename)
	print(f"Loading data from {file_path}...")
	df = pd.read_excel(file_path)
	df.drop(["DayCode"], axis=1, inplace=True, errors="ignore")
	all_dataframes.append(df)

print("Concatenating data...")
data = pd.concat(all_dataframes, ignore_index=True)

print("All data loaded and concatenated successfully!")
print("Combined data shape:", data.shape)
print("Combined data columns:", data.columns)
print("Combined data types:\n", data.dtypes)
print("Combined data preview:\n", data.head())

# --- Detect date column ---
date_col = None

for col in data.columns:
	if np.issubdtype(data[col].dtype, np.datetime64):
		date_col = col
		break

if date_col is None:
	for col in data.columns:
		try:
			converted = pd.to_datetime(data[col], errors="coerce")
		except Exception:
			continue
		if converted.notna().sum() > 0.8 * len(converted):
			data[col] = converted
			date_col = col
			break

if date_col is None:
	raise ValueError("Could not detect a date column to compute seasonality.")

print(f"Using '{date_col}' as date column for seasonality analysis.")

# --- Prepare month and allergen-long format ---
data["month"] = data[date_col].dt.month

numeric_cols = data.select_dtypes(include="number").columns.tolist()
if "month" in numeric_cols:
	numeric_cols.remove("month")

if not numeric_cols:
	raise ValueError("No numeric columns found to treat as pollen allergens.")

print("Treating the following columns as pollen allergens:", numeric_cols)

long = data.melt(
	id_vars=[date_col, "month"],
	value_vars=numeric_cols,
	var_name="allergen",
	value_name="value",
)

long = long.dropna(subset=["value"])
long_positive = long[long["value"] > 0]

if long_positive.empty:
	print("Warning: No positive pollen values found; seasonality table will be empty.")

seasonality = (
	long_positive
	.groupby(["allergen", "month"])
	.agg(
		days_with_pollen=("value", "count"),
		total_pollen=("value", "sum"),
		mean_pollen=("value", "mean"),
	)
	.reset_index()
)

def month_to_season(m: int) -> str:
	if m in (12, 1, 2):
		return "winter"
	elif m in (3, 4, 5):
		return "spring"
	elif m in (6, 7, 8):
		return "summer"
	else:
		return "autumn"

seasonality["season"] = seasonality["month"].map(month_to_season)

print("Seasonality table (first rows):")
print(seasonality.head())
output_dir='./results'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "pollen_seasonality.csv")
seasonality.to_csv(output_path, index=False)
print(f"Seasonality per pollen allergen saved to '{output_path}'.")

# --- Percentage contribution of pollens ---
if long_positive.empty:
	print("No positive pollen values; skipping percentage analysis.")
else:
	# Overall percentage share per allergen
	allergen_totals = long_positive.groupby("allergen")["value"].sum().sort_values(ascending=False)
	total_pollen_all = allergen_totals.sum()
	allergen_percent = (allergen_totals / total_pollen_all * 100).reset_index(name="percent_of_total")

	percent_overall_path = os.path.join(output_dir, "pollen_percentage_overall.csv")
	allergen_percent.to_csv(percent_overall_path, index=False)
	print(f"Overall percentage of total pollen per allergen saved to '{percent_overall_path}'.")
	print("Top 10 allergens by percentage:\n", allergen_percent.head(10))

	# Add season to long_positive to study seasonal percentage distributions
	long_positive = long_positive.copy()
	long_positive["season"] = long_positive["month"].map(month_to_season)

	seasonal_totals = (
		long_positive
		.groupby(["season", "allergen"])["value"]
		.sum()
		.reset_index(name="total_pollen")
	)

	# Within each season, compute allergen percentage
	seasonal_totals["season_total"] = seasonal_totals.groupby("season")["total_pollen"].transform("sum")
	seasonal_totals["percent_within_season"] = (
		seasonal_totals["total_pollen"] / seasonal_totals["season_total"] * 100
	)

	percent_season_path = os.path.join(output_dir, "pollen_percentage_by_season.csv")
	seasonal_totals.to_csv(percent_season_path, index=False)
	print(f"Seasonal percentage of pollen per allergen saved to '{percent_season_path}'.")

# --- Plots ---
if seasonality.empty:
	print("Seasonality table is empty; skipping plots.")
else:
	# 1) Monthly mean pollen for the most prevalent allergen
	total_by_allergen = seasonality.groupby("allergen")["total_pollen"].sum()
	top_allergen = total_by_allergen.idxmax()
	print(f"Top allergen by total pollen: {top_allergen}")

	top_monthly = (
		seasonality[seasonality["allergen"] == top_allergen]
		.sort_values("month")
	)

	plt.figure(figsize=(8, 4))
	plt.bar(top_monthly["month"], top_monthly["mean_pollen"], color="tab:green")
	plt.xlabel("Month")
	plt.ylabel("Mean pollen concentration")
	plt.title(f"Monthly mean pollen for {top_allergen}")
	plt.xticks(range(1, 13))
	plt.tight_layout()
	plot1_path = os.path.join(output_dir, "plot_top_allergen_monthly_mean.png")
	plt.savefig(plot1_path, dpi=150)
	plt.close()
	print(f"Saved plot: {plot1_path}")

	# 2) Heatmap of mean pollen by allergen and month
	pivot = seasonality.pivot(index="allergen", columns="month", values="mean_pollen")
	plt.figure(figsize=(10, max(4, 0.3 * len(pivot.index))))
	im = plt.imshow(pivot.fillna(0), aspect="auto", interpolation="nearest", cmap="YlOrRd")
	plt.colorbar(im, label="Mean pollen concentration")
	plt.yticks(range(len(pivot.index)), pivot.index)
	plt.xticks(range(len(pivot.columns)), pivot.columns)
	plt.xlabel("Month")
	plt.ylabel("Allergen")
	plt.title("Mean pollen by allergen and month")
	plt.tight_layout()
	plot2_path = os.path.join(output_dir, "plot_allergens_month_heatmap.png")
	plt.savefig(plot2_path, dpi=150)
	plt.close()
	print(f"Saved plot: {plot2_path}")

	# 3) Seasonal total pollen for the top allergen
	season_summary = (
		seasonality.groupby(["allergen", "season"])["total_pollen"]
		.sum()
		.reset_index()
	)
	top_season = season_summary[season_summary["allergen"] == top_allergen]
	season_order = ["winter", "spring", "summer", "autumn"]
	top_season["season"] = pd.Categorical(top_season["season"], categories=season_order, ordered=True)
	top_season = top_season.sort_values("season")

	plt.figure(figsize=(6, 4))
	plt.bar(top_season["season"], top_season["total_pollen"], color="tab:blue")
	plt.xlabel("Season")
	plt.ylabel("Total pollen")
	plt.title(f"Seasonal total pollen for {top_allergen}")
	plt.tight_layout()
	plot3_path = os.path.join(output_dir, "plot_top_allergen_season_total.png")
	plt.savefig(plot3_path, dpi=150)
	plt.close()
	print(f"Saved plot: {plot3_path}")

	# 4) Overall percentage distribution of pollens by allergen (bar chart)
	plt.figure(figsize=(10, 5))
	plt.bar(allergen_percent["allergen"], allergen_percent["percent_of_total"], color="tab:orange")
	plt.xticks(rotation=90)
	plt.ylabel("Percentage of total pollen (%)")
	plt.xlabel("Allergen")
	plt.title("Overall distribution of pollen percentage by allergen")
	plt.tight_layout()
	plot4_path = os.path.join(output_dir, "plot_allergen_percentage_overall.png")
	plt.savefig(plot4_path, dpi=150)
	plt.close()
	print(f"Saved plot: {plot4_path}")

	# 5) Seasonal percentage distribution for top allergens (stacked bar)
	season_order = ["winter", "spring", "summer", "autumn"]
	seasonal_totals["season"] = pd.Categorical(seasonal_totals["season"], categories=season_order, ordered=True)

	# Focus on top N allergens overall for readability
	N = 8
	top_allergens_list = allergen_totals.head(N).index.tolist()
	seasonal_top = seasonal_totals[seasonal_totals["allergen"].isin(top_allergens_list)]

	pivot_season = seasonal_top.pivot(index="season", columns="allergen", values="percent_within_season").fillna(0)
	pivot_season = pivot_season.reindex(season_order)

	plt.figure(figsize=(8, 5))
	bottom = np.zeros(len(pivot_season.index))
	for allergen in pivot_season.columns:
		values = pivot_season[allergen].values
		plt.bar(pivot_season.index, values, bottom=bottom, label=allergen)
		bottom += values

	plt.ylabel("Percentage within season (%)")
	plt.xlabel("Season")
	plt.title("Seasonal percentage distribution of top pollens")
	plt.legend(title="Allergen", bbox_to_anchor=(1.05, 1), loc="upper left")
	plt.tight_layout()
	plot5_path = os.path.join(output_dir, "plot_allergen_percentage_by_season.png")
	plt.savefig(plot5_path, dpi=150)
	plt.close()
	print(f"Saved plot: {plot5_path}")