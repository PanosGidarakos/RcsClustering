import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

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
	# print(f"Loading data from {file_path}...")
	df = pd.read_excel(file_path)
	df.drop(["DayCode"], axis=1, inplace=True, errors="ignore")
	df=df[df[["DayDate","Urticaceae","Ambrosia","Artemisia","Cupressaceae"]]]
	all_dataframes.append(df)

data = pd.concat(all_dataframes, ignore_index=True)
print("Concatenating data ok")
print()

# print("All data loaded and concatenated successfully!")
# print("Combined data shape:", data.shape)
# print("Combined data columns:", data.columns)
# print("Combined data types:\n", data.dtypes)
# print("Combined data preview:\n", data.head())

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
print()

# --- Prepare month and allergen-long format ---
data["month"] = data[date_col].dt.month

numeric_cols = data.select_dtypes(include="number").columns.tolist()
if "month" in numeric_cols:
	numeric_cols.remove("month")

if not numeric_cols:
	raise ValueError("No numeric columns found to treat as pollen allergens.")

print("Treating the following columns as pollen allergens:", numeric_cols)
print()

long = data.melt(
	id_vars=[date_col, "month"],
	value_vars=numeric_cols,
	var_name="allergen",
	value_name="value",
)

print(long)
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
print()
print()
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
	heat_data = pivot.fillna(0).to_numpy()
	if np.any(heat_data > 0):
		max_val = float(heat_data.max())
		norm = PowerNorm(gamma=0.4, vmin=0, vmax=max_val)
		im = plt.imshow(
			heat_data,
			aspect="auto",
			interpolation="nearest",
			cmap="inferno",
			norm=norm,
		)
	else:
		im = plt.imshow(
			heat_data,
			aspect="auto",
			interpolation="nearest",
			cmap="inferno",
		)
	plt.colorbar(im, label="Mean pollen concentration (enhanced scale)")
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

	# 6) Mixed signal from top allergens
	# Select top 2 allergens to mix
	top_2_allergens = allergen_totals.head(2).index.tolist()
	print(f"\nCreating mixed signal from top allergens: {top_2_allergens}")

	# Filter long_positive for these allergens and pivot to get daily values
	mixed_data = long_positive[long_positive["allergen"].isin(top_2_allergens)].copy()
	
	# Create daily time series for each allergen
	daily_by_allergen = (
		mixed_data
		.groupby([date_col, "allergen"])["value"]
		.sum()
		.unstack(fill_value=0)
	)
	
	# Create the mixed (summed) signal
	daily_by_allergen["MIXED"] = daily_by_allergen[top_2_allergens].sum(axis=1)
	daily_by_allergen = daily_by_allergen.sort_index()

	# Plot individual signals and the mixed signal
	fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
	
	colors = ["tab:blue", "tab:orange", "tab:red"]
	labels = top_2_allergens + ["MIXED (sum)"]
	columns = top_2_allergens + ["MIXED"]
	
	for ax, col, color, label in zip(axes, columns, colors, labels):
		ax.fill_between(daily_by_allergen.index, daily_by_allergen[col], alpha=0.5, color=color)
		ax.plot(daily_by_allergen.index, daily_by_allergen[col], color=color, linewidth=0.8)
		ax.set_ylabel("Pollen concentration")
		ax.set_title(label)
		ax.grid(True, alpha=0.3)
	
	axes[-1].set_xlabel("Date")
	fig.suptitle(f"Mixed Signal: {top_2_allergens[0]} + {top_2_allergens[1]}", fontsize=14, y=1.02)
	plt.tight_layout()
	plot6_path = os.path.join(output_dir, "plot_mixed_signal_timeseries.png")
	plt.savefig(plot6_path, dpi=150, bbox_inches="tight")
	plt.close()
	print(f"Saved plot: {plot6_path}")

	# 7) Monthly comparison of individual vs mixed signal
	monthly_mixed = mixed_data.copy()
	monthly_mixed["month"] = monthly_mixed[date_col].dt.month
	
	monthly_agg = (
		monthly_mixed
		.groupby(["month", "allergen"])["value"]
		.mean()
		.unstack(fill_value=0)
	)
	monthly_agg["MIXED"] = monthly_agg[top_2_allergens].sum(axis=1)
	
	fig, ax = plt.subplots(figsize=(10, 6))
	x = np.arange(1, 13)
	width = 0.25
	
	ax.bar(x - width, monthly_agg[top_2_allergens[0]].reindex(x, fill_value=0), width, 
	       label=top_2_allergens[0], color="tab:blue", alpha=0.8)
	ax.bar(x, monthly_agg[top_2_allergens[1]].reindex(x, fill_value=0), width, 
	       label=top_2_allergens[1], color="tab:orange", alpha=0.8)
	ax.bar(x + width, monthly_agg["MIXED"].reindex(x, fill_value=0), width, 
	       label="MIXED (sum)", color="tab:red", alpha=0.8)
	
	ax.set_xlabel("Month")
	ax.set_ylabel("Mean pollen concentration")
	ax.set_title(f"Monthly Mean: {top_2_allergens[0]} vs {top_2_allergens[1]} vs Mixed")
	ax.set_xticks(x)
	ax.legend()
	ax.grid(True, alpha=0.3, axis="y")
	plt.tight_layout()
	plot7_path = os.path.join(output_dir, "plot_mixed_signal_monthly.png")
	plt.savefig(plot7_path, dpi=150)
	plt.close()
	print(f"Saved plot: {plot7_path}")

	# Save mixed signal data to CSV
	mixed_csv = daily_by_allergen.reset_index()
	mixed_csv_path = os.path.join(output_dir, "mixed_signal_daily.csv")
	mixed_csv.to_csv(mixed_csv_path, index=False)
	print(f"Mixed signal daily data saved to '{mixed_csv_path}'")

	# ========================================================================
	# 8) SPECTRAL MIXING ANALYSIS
	# ========================================================================
	print("\n" + "="*60)
	print("SPECTRAL MIXING ANALYSIS")
	print("="*60)
	
	# Treat each allergen's monthly pattern as a "spectral signature"
	# Build endmember matrix from top N allergens
	N_endmembers = 5
	top_n_allergens = allergen_totals.head(N_endmembers).index.tolist()
	print(f"Using top {N_endmembers} allergens as endmembers: {top_n_allergens}")
	
	# Create monthly spectral signatures (12 bands = 12 months)
	spectral_signatures = {}
	for allergen in top_n_allergens:
		allergen_seasonality = seasonality[seasonality["allergen"] == allergen]
		monthly_values = allergen_seasonality.set_index("month")["mean_pollen"].reindex(range(1, 13), fill_value=0)
		# Normalize to create a spectral signature (unit sum)
		total = monthly_values.sum()
		if total > 0:
			spectral_signatures[allergen] = monthly_values.values / total
		else:
			spectral_signatures[allergen] = monthly_values.values
	
	# Build endmember matrix (12 months x N endmembers)
	endmember_matrix = np.column_stack([spectral_signatures[a] for a in top_n_allergens])
	
	# Plot 1: Spectral signatures for each endmember
	fig, ax = plt.subplots(figsize=(12, 6))
	months = np.arange(1, 13)
	colors_spectral = plt.cm.tab10(np.linspace(0, 1, N_endmembers))
	
	for i, allergen in enumerate(top_n_allergens):
		ax.plot(months, spectral_signatures[allergen], 'o-', 
		        color=colors_spectral[i], linewidth=2, markersize=6, label=allergen)
	
	ax.set_xlabel("Month (spectral band)")
	ax.set_ylabel("Normalized intensity")
	ax.set_title("Spectral Signatures of Top Pollen Allergens")
	ax.set_xticks(months)
	ax.legend(loc="upper right")
	ax.grid(True, alpha=0.3)
	plt.tight_layout()
	plot8_path = os.path.join(output_dir, "plot_spectral_signatures.png")
	plt.savefig(plot8_path, dpi=150)
	plt.close()
	print(f"Saved plot: {plot8_path}")
	
	# ---- Linear Spectral Unmixing ----
	# Create a synthetic mixed signal with known proportions
	true_fractions = np.array([0.4, 0.35, 0.15, 0.07, 0.03])  # Must sum to 1
	true_fractions = true_fractions[:N_endmembers]
	true_fractions = true_fractions / true_fractions.sum()  # Normalize
	
	# Generate synthetic mixed spectrum
	mixed_spectrum = endmember_matrix @ true_fractions
	
	# Add some noise to make it realistic
	np.random.seed(42)
	noise = np.random.normal(0, 0.02, mixed_spectrum.shape)
	mixed_spectrum_noisy = np.clip(mixed_spectrum + noise, 0, None)
	mixed_spectrum_noisy = mixed_spectrum_noisy / mixed_spectrum_noisy.sum()
	
	# Constrained least squares unmixing (fractions >= 0, sum to 1)
	from scipy.optimize import nnls, minimize
	
	def unmix_constrained(mixed, endmembers):
		"""Fully constrained linear unmixing: fractions >= 0 and sum to 1."""
		n_endmembers = endmembers.shape[1]
		
		def objective(f):
			return np.sum((mixed - endmembers @ f)**2)
		
		# Constraints: sum to 1
		constraints = {'type': 'eq', 'fun': lambda f: np.sum(f) - 1}
		# Bounds: 0 <= f <= 1
		bounds = [(0, 1)] * n_endmembers
		# Initial guess
		x0 = np.ones(n_endmembers) / n_endmembers
		
		result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
		return result.x
	
	# Perform unmixing
	estimated_fractions = unmix_constrained(mixed_spectrum_noisy, endmember_matrix)
	
	print("\nLinear Spectral Unmixing Results:")
	print("-" * 40)
	print(f"{'Allergen':<20} {'True %':>10} {'Estimated %':>12} {'Error':>10}")
	print("-" * 40)
	for i, allergen in enumerate(top_n_allergens):
		true_pct = true_fractions[i] * 100
		est_pct = estimated_fractions[i] * 100
		error = abs(true_pct - est_pct)
		print(f"{allergen:<20} {true_pct:>10.1f} {est_pct:>12.1f} {error:>10.2f}")
	
	rmse = np.sqrt(np.mean((true_fractions - estimated_fractions)**2)) * 100
	print("-" * 40)
	print(f"RMSE: {rmse:.2f}%")
	
	# Plot 2: True vs Estimated fractions
	fig, axes = plt.subplots(1, 3, figsize=(15, 5))
	
	# Subplot 1: Bar comparison
	ax = axes[0]
	x = np.arange(N_endmembers)
	width = 0.35
	ax.bar(x - width/2, true_fractions * 100, width, label='True', color='tab:blue', alpha=0.8)
	ax.bar(x + width/2, estimated_fractions * 100, width, label='Estimated', color='tab:orange', alpha=0.8)
	ax.set_xlabel("Endmember")
	ax.set_ylabel("Fraction (%)")
	ax.set_title("Linear Unmixing: True vs Estimated Fractions")
	ax.set_xticks(x)
	ax.set_xticklabels([a[:8] for a in top_n_allergens], rotation=45, ha='right')
	ax.legend()
	ax.grid(True, alpha=0.3, axis='y')
	
	# Subplot 2: Mixed spectrum reconstruction
	ax = axes[1]
	reconstructed = endmember_matrix @ estimated_fractions
	ax.plot(months, mixed_spectrum_noisy, 'ko-', linewidth=2, markersize=8, label='Observed (noisy)')
	ax.plot(months, mixed_spectrum, 'b--', linewidth=1.5, label='True mixed')
	ax.plot(months, reconstructed, 'r-', linewidth=2, label='Reconstructed')
	ax.set_xlabel("Month (spectral band)")
	ax.set_ylabel("Normalized intensity")
	ax.set_title("Spectrum Reconstruction")
	ax.set_xticks(months)
	ax.legend()
	ax.grid(True, alpha=0.3)
	
	# Subplot 3: Pie charts
	ax = axes[2]
	# Create two pie charts side by side using subplots within subplot
	ax.axis('off')
	ax_true = fig.add_axes([0.68, 0.15, 0.14, 0.7])
	ax_est = fig.add_axes([0.84, 0.15, 0.14, 0.7])
	
	colors_pie = plt.cm.Set2(np.linspace(0, 1, N_endmembers))
	ax_true.pie(true_fractions, colors=colors_pie, autopct='%1.0f%%', startangle=90)
	ax_true.set_title("True Mix", fontsize=10)
	ax_est.pie(estimated_fractions, colors=colors_pie, autopct='%1.0f%%', startangle=90)
	ax_est.set_title("Estimated", fontsize=10)
	
	plt.tight_layout()
	plot9_path = os.path.join(output_dir, "plot_spectral_unmixing.png")
	plt.savefig(plot9_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f"Saved plot: {plot9_path}")
	
	# ---- Apply unmixing to REAL mixed data ----
	print("\nApplying unmixing to actual mixed pollen data...")
	
	# Use the actual monthly mixed signal from the data
	actual_mixed_monthly = monthly_agg["MIXED"].reindex(range(1, 13), fill_value=0).values
	actual_mixed_normalized = actual_mixed_monthly / actual_mixed_monthly.sum() if actual_mixed_monthly.sum() > 0 else actual_mixed_monthly
	
	# Unmix using only the top 2 allergens (Cupressaceae and Urticaceae)
	endmember_2 = np.column_stack([spectral_signatures[a] for a in top_2_allergens])
	estimated_real = unmix_constrained(actual_mixed_normalized, endmember_2)
	
	# True fractions based on total pollen
	total_top2 = allergen_totals[top_2_allergens].sum()
	true_real_fractions = (allergen_totals[top_2_allergens] / total_top2).values
	
	print("\nReal Data Unmixing (Top 2 allergens):")
	print("-" * 40)
	print(f"{'Allergen':<20} {'True %':>10} {'Estimated %':>12}")
	print("-" * 40)
	for i, allergen in enumerate(top_2_allergens):
		print(f"{allergen:<20} {true_real_fractions[i]*100:>10.1f} {estimated_real[i]*100:>12.1f}")
	
	# Plot 3: Real data unmixing
	fig, axes = plt.subplots(1, 2, figsize=(12, 5))
	
	ax = axes[0]
	ax.plot(months, actual_mixed_normalized, 'ko-', linewidth=2, markersize=8, label='Actual mixed')
	for i, allergen in enumerate(top_2_allergens):
		ax.plot(months, spectral_signatures[allergen], '--', linewidth=1.5, 
		        label=f'{allergen} signature', alpha=0.7)
	reconstructed_real = endmember_2 @ estimated_real
	ax.plot(months, reconstructed_real, 'r-', linewidth=2, label='Reconstructed')
	ax.set_xlabel("Month")
	ax.set_ylabel("Normalized intensity")
	ax.set_title("Real Mixed Signal Decomposition")
	ax.set_xticks(months)
	ax.legend()
	ax.grid(True, alpha=0.3)
	
	ax = axes[1]
	x = np.arange(2)
	width = 0.35
	ax.bar(x - width/2, true_real_fractions * 100, width, label='True (from totals)', color='tab:blue', alpha=0.8)
	ax.bar(x + width/2, estimated_real * 100, width, label='Estimated (unmixing)', color='tab:orange', alpha=0.8)
	ax.set_ylabel("Fraction (%)")
	ax.set_title("Real Data: True vs Estimated Fractions")
	ax.set_xticks(x)
	ax.set_xticklabels(top_2_allergens)
	ax.legend()
	ax.grid(True, alpha=0.3, axis='y')
	
	plt.tight_layout()
	plot10_path = os.path.join(output_dir, "plot_real_spectral_unmixing.png")
	plt.savefig(plot10_path, dpi=150)
	plt.close()
	print(f"Saved plot: {plot10_path}")
	
	print("\nSpectral mixing analysis complete!")