#!/bin/bash

# Example usage:
# bash iDM/PostProcessing/Swampland_Survival_Rate.sh --chain-prefix pNG_Planck_InitCond_MCMC
# The --threshold flag can be used to set the acceptable O(s1) respectively O(s2) for the de Sitter Conjecture.
# Default is 0.1, meaning dV/V >= 0.1 or ddV/V <= -0.1 is required to satisfy the de Sitter condition.
# bash iDM/PostProcessing/Swampland_Survival_Rate.sh --chain-prefix pNG_Planck_InitCond_MCMC --threshold 0.3
#
# NOTE: This script evaluates hard-cut thresholds for reference and baselinevalidation.
# The actual Cobaya post-processing and minimizer YAML files now use smooth suppression
# for the de Sitter Conjecture (dSC) via a log-amplitude Gaussian penalty, so the number
# of retained samples in actual runs will typically be higher than what this script reports.

# Exit immediately if any command fails.
set -e

# Treat unset variables as errors to prevent silent mistakes.
set -u

# Fail a pipeline if any command in it fails.
set -o pipefail

# Define the directory containing the chain files.
CHAIN_DIR="/project/home/p201176"

# Define the default chain prefix used by Cobaya output files.
CHAIN_PREFIX="hyperbolic_Planck_InitCond_MCMC"

# Define the default de Sitter threshold used in the OR-condition report.
THRESHOLD="0.1"

# Define the fail penalty used in YAML external likelihoods.
FAIL_PENALTY="-1e10"

# Print help text and exit.
print_help() {
	# Print usage header.
	echo "Usage: $(basename "$0") [options]"

	# Print blank line for readability.
	echo

	# Print option: threshold.
	echo "  --threshold <value>      de Sitter threshold t (default: 0.1)"

	# Print option: fail penalty.
	echo "  --fail-penalty <value>   penalty applied when a criterion fails (default: -1e10)"

	# Print option: chain prefix.
	echo "  --chain-prefix <prefix>  chain filename prefix (default: hyperbolic_Planck_InitCond_MCMC)"

	# Print option: chain directory.
	echo "  --chain-dir <path>       directory containing chain files (default: /project/home/p201176)"

	# Print option: help.
	echo "  -h, --help               show this help message"
}

# Parse CLI arguments one token at a time.
while [[ $# -gt 0 ]]; do
	# Dispatch by current flag.
	case "$1" in
		# Parse custom threshold value.
		--threshold)
			# Require a value after the flag.
			[[ $# -ge 2 ]] || { echo "Error: --threshold requires a value."; exit 1; }

			# Store threshold value.
			THRESHOLD="$2"

			# Shift past flag and value.
			shift 2
			;;

		# Parse custom chain filename prefix.
		--chain-prefix)
			# Require a value after the flag.
			[[ $# -ge 2 ]] || { echo "Error: --chain-prefix requires a value."; exit 1; }

			# Store chain prefix.
			CHAIN_PREFIX="$2"

			# Shift past flag and value.
			shift 2
			;;

		# Parse custom chain directory.
		--chain-dir)
			# Require a value after the flag.
			[[ $# -ge 2 ]] || { echo "Error: --chain-dir requires a value."; exit 1; }

			# Store chain directory path.
			CHAIN_DIR="$2"

			# Shift past flag and value.
			shift 2
			;;

		# Parse custom fail-penalty value.
		--fail-penalty)
			# Require a value after the flag.
			[[ $# -ge 2 ]] || { echo "Error: --fail-penalty requires a value."; exit 1; }

			# Store fail-penalty value.
			FAIL_PENALTY="$2"

			# Shift past flag and value.
			shift 2
			;;

		# Show help and exit on request.
		-h|--help)
			# Print usage details.
			print_help

			# Exit successfully after help.
			exit 0
			;;

		# Reject unknown arguments.
		*)
			# Print unknown option error.
			echo "Error: unknown option '$1'"

			# Print usage summary.
			print_help

			# Exit with failure code.
			exit 1
			;;
	esac
done

# Validate that threshold is numeric.
awk -v x="${THRESHOLD}" 'BEGIN { exit !(x + 0 == x) }' || { echo "Error: --threshold must be numeric."; exit 1; }

# Validate that threshold is strictly positive.
awk -v x="${THRESHOLD}" 'BEGIN { exit !(x > 0) }' || { echo "Error: --threshold must be > 0."; exit 1; }

# Validate that fail-penalty is numeric.
awk -v x="${FAIL_PENALTY}" 'BEGIN { exit !(x + 0 == x) }' || { echo "Error: --fail-penalty must be numeric."; exit 1; }

# Define the file glob from the selected chain prefix.
CHAIN_GLOB="${CHAIN_PREFIX}.[0-9]*.txt"

# Change to the chain directory so relative globs resolve correctly.
cd "${CHAIN_DIR}"

# Ensure at least one chain file matches the requested pattern.
compgen -G "${CHAIN_GLOB}" > /dev/null || { echo "Error: no files matched pattern '${CHAIN_GLOB}' in '${CHAIN_DIR}'."; exit 1; }

# Expand the matching chain files into an array for downstream checks.
CHAIN_FILES=( ${CHAIN_GLOB} )

# Select the first chain file as header reference.
FIRST_CHAIN_FILE="${CHAIN_FILES[0]}"

# Extract a column index by name from a chain header line.
get_col_index() {
	# Read target column name from first function argument.
	local target_name="$1"

	# Read input file path from second function argument.
	local input_file="$2"

	# Parse the first header line and return 1-based field index for target name.
	awk -v key="${target_name}" 'NR==1 { for (i = 1; i <= NF; i++) if ($i == key) { print i; exit 0 } exit 1 }' < <(head -n 1 "${input_file}" | sed 's/^#//')
}

# Resolve dynamic column index for sample weight.
COL_WEIGHT="$(get_col_index "weight" "${FIRST_CHAIN_FILE}")" || { echo "Error: could not find column 'weight' in ${FIRST_CHAIN_FILE}"; exit 1; }

# Resolve dynamic column index for minus-log-posterior.
COL_MINUSLOGPOST="$(get_col_index "minuslogpost" "${FIRST_CHAIN_FILE}")" || { echo "Error: could not find column 'minuslogpost' in ${FIRST_CHAIN_FILE}"; exit 1; }

# Resolve dynamic column index for Distance Conjecture diagnostic.
COL_PHI_RANGE="$(get_col_index "phi_scf_range" "${FIRST_CHAIN_FILE}")" || { echo "Error: could not find column 'phi_scf_range' in ${FIRST_CHAIN_FILE}"; exit 1; }

# Resolve dynamic column index for dSC c1 diagnostic.
COL_DV_MIN="$(get_col_index "dV_V_scf_min" "${FIRST_CHAIN_FILE}")" || { echo "Error: could not find column 'dV_V_scf_min' in ${FIRST_CHAIN_FILE}"; exit 1; }

# Resolve dynamic column index for dSC c3 diagnostic.
COL_DDV_MAX="$(get_col_index "ddV_V_scf_max" "${FIRST_CHAIN_FILE}")" || { echo "Error: could not find column 'ddV_V_scf_max' in ${FIRST_CHAIN_FILE}"; exit 1; }

# Resolve dynamic column index for dSC c2 diagnostic.
COL_DDV_AT_DV_MIN="$(get_col_index "ddV_V_at_dV_V_min" "${FIRST_CHAIN_FILE}")" || { echo "Error: could not find column 'ddV_V_at_dV_V_min' in ${FIRST_CHAIN_FILE}"; exit 1; }

# Resolve dynamic column index for dSC c4 diagnostic.
COL_DV_AT_DDV_MAX="$(get_col_index "dV_V_at_ddV_V_max" "${FIRST_CHAIN_FILE}")" || { echo "Error: could not find column 'dV_V_at_ddV_V_max' in ${FIRST_CHAIN_FILE}"; exit 1; }

# Resolve dynamic column index for SWGC diagnostic.
COL_SWGC_MIN="$(get_col_index "swgc_expr_min" "${FIRST_CHAIN_FILE}")" || { echo "Error: could not find column 'swgc_expr_min' in ${FIRST_CHAIN_FILE}"; exit 1; }

# Define required column names in the order used for consistency checks.
REQUIRED_COLUMN_NAMES=("weight" "minuslogpost" "phi_scf_range" "dV_V_scf_min" "ddV_V_scf_max" "ddV_V_at_dV_V_min" "dV_V_at_ddV_V_max" "swgc_expr_min")

# Define reference indices matching REQUIRED_COLUMN_NAMES order.
REFERENCE_COLUMN_INDICES=("${COL_WEIGHT}" "${COL_MINUSLOGPOST}" "${COL_PHI_RANGE}" "${COL_DV_MIN}" "${COL_DDV_MAX}" "${COL_DDV_AT_DV_MIN}" "${COL_DV_AT_DDV_MAX}" "${COL_SWGC_MIN}")

# Check every matching chain file for consistent required-column positions.
for chain_file in "${CHAIN_FILES[@]}"; do
	# Iterate over required names and validate index stability.
	for i in "${!REQUIRED_COLUMN_NAMES[@]}"; do
		# Read expected column name at current index.
		col_name="${REQUIRED_COLUMN_NAMES[$i]}"

		# Read reference index for this column name.
		ref_idx="${REFERENCE_COLUMN_INDICES[$i]}"

		# Detect index for this column in current file.
		file_idx="$(get_col_index "${col_name}" "${chain_file}")" || { echo "Error: could not find column '${col_name}' in ${chain_file}"; exit 1; }

		# Abort if current file index differs from reference file index.
		[[ "${file_idx}" == "${ref_idx}" ]] || { echo "Error: column '${col_name}' index mismatch in ${chain_file} (expected ${ref_idx}, got ${file_idx})."; exit 1; }
		done
done

# Run AWK to compute weighted survival rates for swampland constraints.
awk -v selected_t="${THRESHOLD}" \
	-v fail_penalty="${FAIL_PENALTY}" \
	-v col_weight="${COL_WEIGHT}" \
	-v col_minuslogpost="${COL_MINUSLOGPOST}" \
	-v col_phi_range="${COL_PHI_RANGE}" \
	-v col_dv_min="${COL_DV_MIN}" \
	-v col_ddv_max="${COL_DDV_MAX}" \
	-v col_ddv_at_dv_min="${COL_DDV_AT_DV_MIN}" \
	-v col_dv_at_ddv_max="${COL_DV_AT_DDV_MAX}" \
	-v col_swgc_min="${COL_SWGC_MIN}" '

# Initialize global accumulators and selected-threshold counters.
BEGIN {
	# Initialize total weighted sum.
	W = 0

	# Initialize total row count.
	n = 0

	# Initialize weighted sum of minus-log-posterior.
	mlpW = 0

	# Initialize running minimum of minus-log-posterior.
	mlpMin = 1e99

	# Initialize running maximum of minus-log-posterior.
	mlpMax = -1e99

	# Initialize weighted count for de Sitter condition c1 at selected threshold.
	c1w = 0

	# Initialize weighted count for de Sitter condition c2 at selected threshold.
	c2w = 0

	# Initialize weighted count for de Sitter condition c3 at selected threshold.
	c3w = 0

	# Initialize weighted count for de Sitter condition c4 at selected threshold.
	c4w = 0

	# Initialize weighted count for 4-way OR pass at selected threshold.
	orw = 0

	# Initialize weighted count for branch A pass (c1 or c2).
	branchAw = 0

	# Initialize weighted count for branch B pass (c3 or c4).
	branchBw = 0

	# Initialize weighted count for c2 rescue events where c1 fails.
	rescue2w = 0

	# Initialize weighted count for c4 rescue events where c3 fails.
	rescue4w = 0

	# Initialize weighted count of Distance Conjecture violations.
	phiBad = 0

	# Initialize weighted count of SWGC violations.
	swgcBad = 0

	# Initialize unweighted count of Distance Conjecture violations.
	nPhiBad = 0

	# Initialize unweighted count of SWGC violations.
	nSwgcBad = 0

	# Initialize unweighted count of de Sitter OR pass at selected threshold.
	nOrPass = 0

	# Initialize unweighted count of branch A pass (c1 or c2).
	nBranchAPass = 0

	# Initialize unweighted count of branch B pass (c3 or c4).
	nBranchBPass = 0

	# Initialize unweighted count of c2 rescue events where c1 fails.
	nRescue2 = 0

	# Initialize unweighted count of c4 rescue events where c3 fails.
	nRescue4 = 0

	# Initialize weighted count of samples passing both hard cuts.
	wHardPass = 0

	# Initialize unweighted count of samples passing both hard cuts.
	nHardPass = 0

	# Initialize weighted sum after all YAML-equivalent cuts.
	wAllPass = 0

	# Initialize unweighted count after all YAML-equivalent cuts.
	nAllPass = 0

	# Initialize sum of normalized importance weights after all cuts.
	sumImp = 0

	# Initialize sum of squared normalized importance weights after all cuts.
	sumImp2 = 0

	# Initialize max retained weight after all cuts.
	maxAllPassW = 0

	# Initialize sum of squared retained weights after all cuts.
	w2AllPass = 0

	# Initialize minimum YAML-aligned importance factor across all samples.
	impMin = 1e99

	# Initialize maximum YAML-aligned importance factor across all samples.
	impMax = -1e99
}

# Skip comment lines that start with # in Cobaya chain files.
/^#/ {
	next
}

# Process each data row and update all statistics.
{
	# Read sample weight from detected weight column.
	wt = $(col_weight)

	# Read minus-log-posterior from detected minuslogpost column.
	mlp = $(col_minuslogpost)

	# Read scalar field excursion range from detected phi_scf_range column.
	phi = $(col_phi_range)

	# Read dV/V minimum diagnostic from detected dV_V_scf_min column.
	dv = $(col_dv_min)

	# Read ddV/V maximum diagnostic from detected ddV_V_scf_max column.
	ddvmax = $(col_ddv_max)

	# Read ddV/V at dV/V minimum from detected ddV_V_at_dV_V_min column.
	ddvatmin = $(col_ddv_at_dv_min)

	# Read dV/V at ddV/V maximum from detected dV_V_at_ddV_V_max column.
	dvatmax = $(col_dv_at_ddv_max)

	# Read SWGC expression minimum from detected swgc_expr_min column.
	swgc = $(col_swgc_min)

	# Add sample weight to total weighted sum.
	W += wt

	# Increment unweighted row counter.
	n += 1

	# Add weighted minus-log-posterior contribution.
	mlpW += wt * mlp

	# Update minimum minus-log-posterior if this sample is lower.
	if (mlp < mlpMin) {
		mlpMin = mlp
	}

	# Update maximum minus-log-posterior if this sample is higher.
	if (mlp > mlpMax) {
		mlpMax = mlp
	}

	# Add weight to Distance violation counter when phi > 10.
	if (phi > 10) {
		phiBad += wt
	}

	# Add weight to SWGC violation counter when swgc < 0.
	if (swgc < 0) {
		swgcBad += wt
	}

	# Evaluate de Sitter condition c1 at selected threshold: dV_V_scf_min >= t.
	c1 = (dv >= selected_t)

	# Evaluate de Sitter condition c2 at selected threshold: ddV_V_at_dV_V_min <= -t.
	c2 = (ddvatmin <= -selected_t)

	# Evaluate de Sitter condition c3 at selected threshold: ddV_V_scf_max <= -t.
	c3 = (ddvmax <= -selected_t)

	# Evaluate de Sitter condition c4 at selected threshold: dV_V_at_ddV_V_max >= t.
	c4 = (dvatmax >= selected_t)

	# Accumulate weighted pass count for c1.
	if (c1) {
		c1w += wt
	}

	# Accumulate weighted pass count for c2.
	if (c2) {
		c2w += wt
	}

	# Accumulate weighted pass count for c3.
	if (c3) {
		c3w += wt
	}

	# Accumulate weighted pass count for c4.
	if (c4) {
		c4w += wt
	}

	# Accumulate weighted pass count for the 4-way OR condition.
	if (c1 || c2 || c3 || c4) {
		orw += wt
	}

	# Accumulate unweighted pass count for the 4-way OR condition.
	if (c1 || c2 || c3 || c4) {
		nOrPass += 1
	}

	# Accumulate weighted pass count for branch A (c1 or c2).
	if (c1 || c2) {
		branchAw += wt
	}

	# Accumulate unweighted pass count for branch A (c1 or c2).
	if (c1 || c2) {
		nBranchAPass += 1
	}

	# Accumulate weighted pass count for branch B (c3 or c4).
	if (c3 || c4) {
		branchBw += wt
	}

	# Accumulate unweighted pass count for branch B (c3 or c4).
	if (c3 || c4) {
		nBranchBPass += 1
	}

	# Accumulate weighted c2 rescue when c1 fails but c2 passes.
	if ((!c1) && c2) {
		rescue2w += wt
	}

	# Accumulate unweighted c2 rescue when c1 fails but c2 passes.
	if ((!c1) && c2) {
		nRescue2 += 1
	}

	# Accumulate weighted c4 rescue when c3 fails but c4 passes.
	if ((!c3) && c4) {
		rescue4w += wt
	}

	# Accumulate unweighted c4 rescue when c3 fails but c4 passes.
	if ((!c3) && c4) {
		nRescue4 += 1
	}

	# Accumulate unweighted Distance violation count.
	if (phi > 10) {
		nPhiBad += 1
	}

	# Accumulate unweighted SWGC violation count.
	if (swgc < 0) {
		nSwgcBad += 1
	}

	# Accumulate weighted pass count for both hard cuts combined.
	if ((phi <= 10) && (swgc >= 0)) {
		wHardPass += wt
	}

	# Accumulate unweighted pass count for both hard cuts combined.
	if ((phi <= 10) && (swgc >= 0)) {
		nHardPass += 1
	}

	# Evaluate YAML-equivalent all-pass condition: DC AND SWGC AND dSC(OR).
	all_pass = ((phi <= 10) && (swgc >= 0) && (c1 || c2 || c3 || c4))

	# Build YAML-aligned added log-likelihood contribution for this sample.
	delta_ll = 0
	if (phi > 10) {
		delta_ll += fail_penalty
	}
	if (swgc < 0) {
		delta_ll += fail_penalty
	}
	if (!(c1 || c2 || c3 || c4)) {
		delta_ll += fail_penalty
	}

	# Convert added log-likelihood to an importance factor for range diagnostics.
	# Clamp to avoid exp() underflow/overflow warnings in AWK.
	if (delta_ll <= -700) {
		imp = 0
	} else if (delta_ll >= 700) {
		imp = 1e300
	} else {
		imp = exp(delta_ll)
	}

	# Track global min/max importance factor.
	if (imp < impMin) {
		impMin = imp
	}
	if (imp > impMax) {
		impMax = imp
	}

	# Accumulate all-pass weighted/unweighted counts.
	if (all_pass) {
		wAllPass += wt
		nAllPass += 1
		w2AllPass += wt * wt

		# Track max retained weight for Cobaya-style importance normalization.
		if (wt > maxAllPassW) {
			maxAllPassW = wt
		}
	}
}

# Print the final summary after processing all rows.
END {
	# Print chain-level header.
	printf "=== CHAIN SUMMARY ===\n"

	# Print detected column indices for traceability across models.
	printf "columns: weight=%d minuslogpost=%d phi_scf_range=%d dV_V_scf_min=%d ddV_V_scf_max=%d ddV_V_at_dV_V_min=%d dV_V_at_ddV_V_max=%d swgc_expr_min=%d\n", col_weight, col_minuslogpost, col_phi_range, col_dv_min, col_ddv_max, col_ddv_at_dv_min, col_dv_at_ddv_max, col_swgc_min

	# Print total rows and total weighted mass.
	printf "rows=%d W=%.12g\n", n, W

	# Print weighted mean, min, and max of minus-log-posterior.
	printf "minuslogpost weighted_mean=%.6f min=%.6f max=%.6f\n", mlpW / W, mlpMin, mlpMax

	# Print section break for hard constraints.
	printf "\n=== SWAMPLAND CONSTRAINTS ===\n"

	# Print Distance Conjecture pass and violation fractions.
	printf "Distance (phi<=10) pass=%.4f%% violate=%.4f%%\n", 100 * (1 - (phiBad / W)), 100 * (phiBad / W)

	# Print SWGC pass and violation fractions.
	printf "SWGC (swgc>=0) pass=%.4f%% violate=%.4f%%\n", 100 * (1 - (swgcBad / W)), 100 * (swgcBad / W)

	# Print combined hard-cut pass fraction.
	printf "DC and SWGC combined pass=%.4f%% violate=%.4f%%\n", 100 * (wHardPass / W), 100 * (1 - (wHardPass / W))

	# Print combined hard-cut unweighted sample count.
	printf "DC and SWGC combined samples=%d/%d\n", nHardPass, n

	# Print section break for de Sitter OR criterion at selected threshold.
	printf "\n=== dE SITTER CONJECTURE (threshold t = %g) ===\n", selected_t

	# Print c1 and c2 weighted pass fractions.
	printf "  c1(dV>=t)=%.4f%% c2(ddV_at_dV<=-t)=%.4f%%\n", 100 * (c1w / W), 100 * (c2w / W)

	# Print c3 and c4 weighted pass fractions.
	printf "  c3(ddVmax<=-t)=%.4f%% c4(dV_at_ddV>=t)=%.4f%%\n", 100 * (c3w / W), 100 * (c4w / W)

	# Print OR pass and violation fractions.
	printf "  OR-pass=%.4f%% violate=%.4f%%\n", 100 * (orw / W), 100 * (1 - (orw / W))

	# Print OR pass as unweighted sample count.
	printf "  OR-pass samples=%d/%d\n", nOrPass, n

	# Print branch A pass and violation fractions.
	printf "  branchA-pass(c1||c2)=%.4f%% violate=%.4f%%\n", 100 * (branchAw / W), 100 * (1 - (branchAw / W))

	# Print branch A pass as unweighted sample count.
	printf "  branchA-pass samples=%d/%d\n", nBranchAPass, n

	# Print branch B pass and violation fractions.
	printf "  branchB-pass(c3||c4)=%.4f%% violate=%.4f%%\n", 100 * (branchBw / W), 100 * (1 - (branchBw / W))

	# Print branch B pass as unweighted sample count.
	printf "  branchB-pass samples=%d/%d\n", nBranchBPass, n

	# Print c2 rescue contribution where c1 is violated.
	printf "  rescue-by-c2(!c1&&c2)=%.4f%% samples=%d/%d\n", 100 * (rescue2w / W), nRescue2, n

	# Print c4 rescue contribution where c3 is violated.
	printf "  rescue-by-c4(!c3&&c4)=%.4f%% samples=%d/%d\n", 100 * (rescue4w / W), nRescue4, n

    # Print combined Swampland (DC+SWGC+dSC) pass and violation fractions.
    printf "\n=== SWAMPLAND CONSTRAINTS COMBINED ===\n"
	printf "All Swampland combined pass=%.4f%% violate=%.4f%%\n", 100 * (wAllPass / W), 100 * (1 - (wAllPass / W))

	# Print combined Swampland unweighted sample count.
	printf "All Swampland combined samples=%d/%d\n", nAllPass, n

	# Print section break for YAML-equivalent post-processing cross-check.
	printf "\n=== YAML-ALIGNED CROSS-CHECK ===\n"

	# Print distinct points surviving all three YAML criteria.
	printf "Final number of distinct sample points: %d\n", nAllPass

	# Print YAML-aligned importance-factor range from internal computations.
	if (impMin <= impMax) {
		printf "Importance weight range: %g -- %g\n", impMin, impMax
	} else {
		printf "Importance weight range: 0 -- 0\n"
	}

	# Print points removed because final effective weight is zero.
	printf "Points deleted due to zero weight: %d\n", (n - nAllPass)

	# Print effective number of single samples if independent: (sum w)/max(w).
	if (maxAllPassW > 0) {
		sumImp = wAllPass / maxAllPassW
		printf "Effective number of single samples if independent (sum w)/max(w): %d\n", int(sumImp)
	} else {
		sumImp = 0
		printf "Effective number of single samples if independent (sum w)/max(w): 0\n"
	}

	# Print effective number of weighted samples if independent: (sum w)^2/sum(w^2).
	if (w2AllPass > 0) {
		printf "Effective number of weighted samples if independent (sum w)^2/sum(w^2): %d\n", int((wAllPass * wAllPass) / w2AllPass)
	} else {
		printf "Effective number of weighted samples if independent (sum w)^2/sum(w^2): 0\n"
	}
}
' ${CHAIN_GLOB}