import { describe, it, expect } from "vitest";
import {
  FEATURE_ENGINEERING_STRATEGY_OPTIONS,
  defaultsForStrategy,
  isColumnTransformStrategy,
  isRetiredFeatureStrategy,
  strategyOptionsFor,
} from "@/views/AIAgents/Workflows/nodeDialogs/training/featureEngineeringStrategies";

describe("feature engineering strategies", () => {
  it("normalize and standardize can't be picked for new features", () => {
    const values = FEATURE_ENGINEERING_STRATEGY_OPTIONS.map((o) => o.value);
    expect(values).toEqual([
      "custom_expression",
      "bin_numeric",
      "polynomial",
      "log_transform",
      "quantile_transform",
      "power_transform",
      "pca",
    ]);
    expect(values).not.toContain("normalize");
    expect(values).not.toContain("standardize");
  });

  it("a feature saved with a retired strategy still shows it, marked retired", () => {
    const options = strategyOptionsFor("standardize");
    expect(options.at(-1)).toEqual({ value: "standardize", label: "Standardize (retired)" });
    expect(isRetiredFeatureStrategy("standardize")).toBe(true);
    expect(isRetiredFeatureStrategy("normalize")).toBe(true);
  });

  it("a current strategy gets only the current options", () => {
    expect(strategyOptionsFor("polynomial")).toEqual(FEATURE_ENGINEERING_STRATEGY_OPTIONS);
    expect(isRetiredFeatureStrategy("custom_expression")).toBe(false);
  });

  it("the four column transforms start from valid defaults", () => {
    expect(defaultsForStrategy("log_transform")).toEqual({ replaceSourceColumns: false });
    expect(defaultsForStrategy("quantile_transform")).toEqual({
      quantileOutputDistribution: "uniform",
      nQuantiles: 1000,
      replaceSourceColumns: false,
    });
    expect(defaultsForStrategy("power_transform")).toEqual({
      powerMethod: "yeo-johnson",
      replaceSourceColumns: false,
    });
    expect(defaultsForStrategy("pca")).toEqual({
      pcaComponents: 2,
      pcaStandardize: true,
      replaceSourceColumns: true,
    });
    expect(defaultsForStrategy("custom_expression")).toEqual({});
    // Shown in the form, so saved too (they used to be shown but not saved).
    expect(defaultsForStrategy("polynomial")).toEqual({ polynomialDegree: 2 });
    expect(defaultsForStrategy("bin_numeric")).toEqual({ numBins: 5 });
    expect(["log_transform", "quantile_transform", "power_transform", "pca"].every((s) =>
      isColumnTransformStrategy(s as never)
    )).toBe(true);
    expect(isColumnTransformStrategy("polynomial")).toBe(false);
  });
});
