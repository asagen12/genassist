import { describe, it, expect } from "vitest";
import {
  describeOutputs,
  planFeatureColumns,
  polynomialCombinationNames,
} from "@/views/AIAgents/Workflows/nodeDialogs/training/featureColumnNames";
import { FeatureEngineeringItem } from "@/views/AIAgents/Workflows/types/nodes";

// Same cases as backend/tests/unit/test_feature_auto_names.py, so the dialog's
// "Creates ..." preview matches the columns training really creates.
const START = ["year", "month", "price", "income"];
const f = (item: Partial<FeatureEngineeringItem>): FeatureEngineeringItem =>
  ({ id: Math.random().toString(), newColumnName: "", strategy: "custom_expression", ...item }) as FeatureEngineeringItem;
const outputs = (items: Partial<FeatureEngineeringItem>[]) =>
  planFeatureColumns(items.map(f), START).map((p) => p.outputs);

describe("polynomialCombinationNames (matches scikit-learn)", () => {
  it("degree 2 and 3", () => {
    expect(polynomialCombinationNames(["year", "month"], 2)).toEqual(["year^2", "year month", "month^2"]);
    expect(polynomialCombinationNames(["year", "month"], 3)).toEqual([
      "year^2", "year month", "month^2", "year^3", "year^2 month", "year month^2", "month^3",
    ]);
    expect(polynomialCombinationNames(["a", "b", "c"], 2)).toHaveLength(6);
    expect(polynomialCombinationNames(["a", "b", "c", "d"], 3)).toHaveLength(30);
  });
});

describe("automatic feature column names", () => {
  it("each strategy names its columns", () => {
    expect(outputs([{ strategy: "log_transform", sourceColumns: ["price", "income"] }])).toEqual([["log_price", "log_income"]]);
    expect(outputs([{ strategy: "log_transform", sourceColumns: ["price"] }])).toEqual([["log_price"]]);
    expect(outputs([{ strategy: "quantile_transform", sourceColumns: ["price"] }])).toEqual([["quantile_price"]]);
    expect(outputs([{ strategy: "power_transform", sourceColumns: ["price", "income"] }])).toEqual([["power_price", "power_income"]]);
    expect(outputs([{ strategy: "bin_numeric", binColumn: "price", numBins: 3 }])).toEqual([["price_bin"]]);
    expect(outputs([{ strategy: "polynomial", polynomialColumns: ["year", "month"], polynomialDegree: 2 }])).toEqual([
      ["poly_year^2", "poly_year month", "poly_month^2"],
    ]);
    expect(outputs([{ strategy: "pca", sourceColumns: ["price", "income"], pcaComponents: 2 }])).toEqual([["pca_1", "pca_2"]]);
  });

  it("clashes are numbered", () => {
    expect(outputs([
      { strategy: "pca", sourceColumns: ["price", "income"], pcaComponents: 1, replaceSourceColumns: false },
      { strategy: "pca", sourceColumns: ["year", "month"], pcaComponents: 1, replaceSourceColumns: false },
    ])).toEqual([["pca_1"], ["pca2_1"]]);
    expect(outputs([
      { strategy: "log_transform", sourceColumns: ["price"] },
      { strategy: "log_transform", sourceColumns: ["price"] },
    ])).toEqual([["log_price"], ["log2_price"]]);
    expect(outputs([
      { strategy: "bin_numeric", binColumn: "price", numBins: 3 },
      { strategy: "bin_numeric", binColumn: "price", numBins: 5 },
    ])).toEqual([["price_bin"], ["price_bin_2"]]);
  });

  it("saved names are kept; custom expression uses its name", () => {
    expect(outputs([
      { newColumnName: "lp", strategy: "log_transform", sourceColumns: ["price"] },
      { newColumnName: "q", strategy: "quantile_transform", sourceColumns: ["price", "income"] },
      { newColumnName: "f", strategy: "polynomial", polynomialColumns: ["year"], polynomialDegree: 2 },
      { newColumnName: "pb", strategy: "bin_numeric", binColumn: "price", numBins: 3 },
      { newColumnName: "ym", strategy: "custom_expression", expression: "year * month" },
    ])).toEqual([["lp"], ["q_price", "q_income"], ["f_year^2"], ["pb"], ["ym"]]);
  });

  it("a later feature sees earlier outputs, without replaced sources", () => {
    const plan = planFeatureColumns(
      [
        f({ strategy: "pca", sourceColumns: ["price", "income"], pcaComponents: 1 }), // replaces by default
        f({ strategy: "log_transform", sourceColumns: ["pca_1"] }),
      ],
      START
    );
    expect(plan[1].before).toEqual(["year", "month", "pca_1"]);
    expect(plan[1].outputs).toEqual(["log_pca_1"]);
  });

  it("describes what's missing and what is created", () => {
    const [noName, noCols, pcaShare] = planFeatureColumns(
      [
        f({ strategy: "custom_expression", expression: "year * month" }),
        f({ strategy: "log_transform", sourceColumns: [] }),
        f({ strategy: "pca", sourceColumns: ["price", "income"], pcaComponents: 0.95 }),
      ],
      START
    );
    expect(describeOutputs(noName)).toBe("No columns yet - enter a New Column Name.");
    expect(describeOutputs(noCols)).toBe("No columns yet - choose columns.");
    expect(describeOutputs(pcaShare)).toBe(
      "Creates pca_1 plus as many components as needed to keep 95% of the variance."
    );
  });
});
