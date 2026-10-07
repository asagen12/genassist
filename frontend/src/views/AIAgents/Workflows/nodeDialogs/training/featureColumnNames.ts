import { FeatureEngineeringItem, FeatureEngineeringStrategy } from "../../types/nodes";

/**
 * The column names each Feature Engineering feature will create, worked out
 * the same way Train Model does (train_model_node.py: _AUTO_NAME_PREFIXES,
 * _first_free_name, _engineer_features, _apply_column_transform).
 *
 * Only Custom Expression has a New Column Name field. Every other strategy
 * names its columns automatically (log_price, pca_1, poly_year^2,
 * price_bin, ...), numbered when a name is taken (log2_price, pca2_1,
 * price_bin_2). Features saved with a name keep using it.
 */

const AUTO_NAME_PREFIXES: Partial<Record<FeatureEngineeringStrategy, string>> = {
  polynomial: "poly",
  log_transform: "log",
  quantile_transform: "quantile",
  power_transform: "power",
  pca: "pca",
  normalize: "normalized",
  standardize: "standardized",
};

const COLUMN_TRANSFORMS: FeatureEngineeringStrategy[] = [
  "log_transform",
  "quantile_transform",
  "power_transform",
  "pca",
];

/** base, base2, base3, ... (or base, base_2, base_3 with separator "_"). */
function* numbered(base: string, separator = ""): Generator<string> {
  yield base;
  for (let n = 2; n < 1000; n++) yield `${base}${separator}${n}`;
}

function firstFree(
  candidates: Generator<string>,
  makeNames: (candidate: string) => string[],
  taken: Set<string>
): string {
  for (const candidate of candidates) {
    if (!makeNames(candidate).some((name) => taken.has(name))) return candidate;
  }
  throw new Error("No free feature column name");
}

/** itertools.combinations_with_replacement over indices 0..n-1, size k. */
function* combinationsWithReplacement(n: number, k: number, start = 0): Generator<number[]> {
  if (k === 0) {
    yield [];
    return;
  }
  for (let i = start; i < n; i++) {
    for (const rest of combinationsWithReplacement(n, k - 1, i)) yield [i, ...rest];
  }
}

/**
 * scikit-learn PolynomialFeatures(include_bias=False).get_feature_names_out,
 * minus the degree-1 terms (the original columns), e.g. year, month at
 * degree 2 -> ["year^2", "year month", "month^2"].
 */
export function polynomialCombinationNames(columns: string[], degree: number): string[] {
  const names: string[] = [];
  for (let d = 2; d <= degree; d++) {
    for (const combo of combinationsWithReplacement(columns.length, d)) {
      const powers = columns.map((_, i) => combo.filter((c) => c === i).length);
      names.push(
        columns
          .map((col, i) => (powers[i] === 0 ? null : powers[i] === 1 ? col : `${col}^${powers[i]}`))
          .filter((part): part is string => part !== null)
          .join(" ")
      );
    }
  }
  return names;
}

export interface FeatureColumnPlan {
  /** Columns available to this feature (the data's, plus earlier features'). */
  before: string[];
  /** Columns this feature creates. */
  outputs: string[];
  /** Why there are no (or not all) outputs yet, e.g. "choose columns". */
  note?: string;
}

/**
 * Walk the features in order - as training does - and work out each one's
 * output columns, starting from the model's feature columns.
 */
export function planFeatureColumns(
  features: FeatureEngineeringItem[],
  startColumns: string[]
): FeatureColumnPlan[] {
  let current = [...startColumns];
  return features.map((feature) => {
    const before = [...current];
    const taken = new Set(current);
    const name = (feature.newColumnName || "").trim();
    let outputs: string[] = [];
    let note: string | undefined;

    if (feature.strategy === "custom_expression") {
      if (name) outputs = [name];
      else note = "enter a New Column Name";
    } else if (feature.strategy === "bin_numeric") {
      if (!feature.binColumn) note = "choose a column";
      else outputs = [name || firstFree(numbered(`${feature.binColumn}_bin`, "_"), (n) => [n], taken)];
    } else if (feature.strategy === "polynomial") {
      const columns = (feature.polynomialColumns || []).filter((c) => taken.has(c));
      if (!columns.length) note = "choose columns";
      else {
        const combos = polynomialCombinationNames(columns, feature.polynomialDegree || 2);
        const prefix =
          name || firstFree(numbered(AUTO_NAME_PREFIXES.polynomial as string), (p) => combos.map((c) => `${p}_${c}`), taken);
        outputs = combos.map((c) => `${prefix}_${c}`);
      }
    } else if (feature.strategy === "normalize" || feature.strategy === "standardize") {
      const columns = (feature.sourceColumns || []).filter((c) => taken.has(c));
      const auto = !name;
      const prefix =
        name || firstFree(numbered(AUTO_NAME_PREFIXES[feature.strategy] as string), (p) => columns.map((c) => `${p}_${c}`), taken);
      outputs = columns.map((c) => (!auto && columns.length === 1 ? prefix : `${prefix}_${c}`));
    } else if (COLUMN_TRANSFORMS.includes(feature.strategy)) {
      const columns = Array.from(new Set(feature.sourceColumns || [])).filter((c) => taken.has(c));
      const replace = feature.replaceSourceColumns ?? feature.strategy === "pca";
      if (!columns.length) note = "choose columns";
      else {
        const kept = current.filter((c) => !(replace && columns.includes(c)));
        const components = feature.pcaComponents ?? 2;
        const pcaCount = Number.isInteger(components) && components >= 1 ? components : null;
        const auto = !name;
        const namesFor = (prefix: string): string[] => {
          if (feature.strategy === "pca") {
            return Array.from({ length: pcaCount ?? 1 }, (_, i) => `${prefix}_${i + 1}`);
          }
          if (columns.length === 1 && !auto) return [prefix];
          return columns.map((c) => `${prefix}_${c}`);
        };
        const prefix =
          name || firstFree(numbered(AUTO_NAME_PREFIXES[feature.strategy] as string), namesFor, new Set(kept));
        outputs = namesFor(prefix);
        if (feature.strategy === "pca" && pcaCount === null) {
          note = `plus as many components as needed to keep ${Math.round(components * 100)}% of the variance`;
        }
        current = kept;
      }
    }

    current = [...current, ...outputs];
    return { before, outputs, note };
  });
}

/** "Creates log_price, log_income" - shown under each feature. */
export const describeOutputs = (plan: FeatureColumnPlan): string => {
  if (!plan.outputs.length) return `No columns yet - ${plan.note ?? "complete the settings"}.`;
  const shown = plan.outputs.length > 12 ? [...plan.outputs.slice(0, 12), `… (${plan.outputs.length} columns)`] : plan.outputs;
  return `Creates ${shown.join(", ")}${plan.note ? ` ${plan.note}` : ""}.`;
};
