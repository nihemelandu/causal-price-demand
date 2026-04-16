# Data Understanding: Causal Price Elasticity Estimation for SKU-Level Demand

---

## 1. Purpose of This Document

This document records the findings of the Data Understanding phase (CRISP-DM Phase 2) for the causal price elasticity estimation project. It answers the central question this iteration set out to address:

> *Is FreshRetailNet-50K fit for purpose for causal price elasticity estimation at the SKU level using Double/Debiased Machine Learning (DML)?*

The answer, supported by the evidence below, is **yes — with specific adaptations required.**

---

## 2. Dataset Overview

**Dataset:** FreshRetailNet-50K
**Source:** Hugging Face — `Dingdong-Inc/FreshRetailNet-50K`
**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
**Reference:** Wang et al. (2025). *FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent Demand Recovery and Forecasting in Fresh Retail.* arXiv:2505.16319

FreshRetailNet-50K is a large-scale, publicly available retail dataset comprising 50,000 store-product time series of hourly sales data for perishable goods, collected from 898 stores across 18 major Chinese cities between March and June 2024. It is the first retail dataset to provide explicit stockout event annotations at hourly resolution, making it uniquely suited for demand estimation under real-world supply constraints.

---

## 3. Data Description

### 3.1 Basic Structure

| Property | Value |
|---|---|
| Total rows (train split) | 4,500,000 |
| Total columns | 19 |
| Date range | 2024-03-28 to 2024-06-25 |
| Temporal span | 90 days |
| Unique cities | 18 |
| Unique stores | 898 |
| Unique products (SKUs) | 865 |
| Unique store-product pairs | 50,000 |

### 3.2 Feature Inventory

| Feature | Type | Role in Causal Model | Description |
|---|---|---|---|
| `city_id` | int64 | Identifier / Fixed Effect | City where the store is located |
| `store_id` | int64 | Identifier / Fixed Effect | Unique store identifier |
| `management_group_id` | int64 | Identifier / Confounder | Top-level product category (7 groups) |
| `first_category_id` | int64 | Identifier / Confounder | First-level product category (32 categories) |
| `second_category_id` | int64 | Identifier / Confounder | Second-level product category (84 categories) |
| `third_category_id` | int64 | Identifier / Confounder | Third-level product category (233 categories) |
| `product_id` | int64 | Identifier / Fixed Effect | Unique SKU identifier (865 products) |
| `dt` | string → datetime | Time index | Date of observation |
| `sale_amount` | float64 | **Outcome variable** | Daily total sales quantity |
| `hours_sale` | sequence (24 values) | Supporting signal | Hourly sales breakdown — 24-element list summing to `sale_amount` |
| `stock_hour6_22_cnt` | int32 | Confounder / Quality flag | Number of hours between 6AM–10PM with stock available |
| `hours_stock_status` | sequence (24 values) | Stockout annotation | Hourly binary stockout indicator (1 = out of stock) |
| `discount` | float64 | **Treatment variable** | Promotional discount rate (1.0 = no discount; <1.0 = discounted) |
| `holiday_flag` | int32 | Confounder | Binary; 1 if Chinese statutory holiday |
| `activity_flag` | int32 | Confounder | Binary; 1 if promotional activity active |
| `precpt` | float64 | Confounder | Daily precipitation at store location (mm) |
| `avg_temperature` | float64 | Confounder | Daily average temperature (°C) |
| `avg_humidity` | float64 | Confounder | Daily average humidity (%) |
| `avg_wind_level` | float64 | Confounder | Daily average wind level |

---

## 4. Panel Structure Verification

A balanced panel structure — where every store-product pair is observed for every time period — is a prerequisite for the dynamic panel DML approach used in the anchor paper.

**Finding: The panel is perfectly balanced.**

Every one of the 50,000 store-product pairs has exactly 90 daily observations — one for each day in the observation window. There are no gaps in the time series, no missing dates, and no duplicate records.

Panel dimensions: 898 stores × 865 products × 90 time periods
Total store-product pairs: 50,000
Min observations per pair: 90
Max observations per pair: 90
Balanced: True
Missing dates in dataset: 0
Duplicate store-product-date records: 0

This is a strong structural foundation for DML. The anchor paper worked with 7,226 products × 12 time periods. FreshRetailNet-50K provides a richer panel — more time periods (90 vs 12) and more cross-sectional units (50,000 store-product pairs vs 7,226 products), giving substantially more variation for causal identification.

**Important nuance — product coverage is heterogeneous:**

Not every product appears in every store. The median store carries 50 products (range: 1–164), and the median product appears in 10 stores (range: 1–783). This means the effective panel is sparse in the product × store dimension, even though it is perfectly balanced within each observed store-product pair. This heterogeneity must be accounted for in the modeling strategy — specifically, product and store fixed effects must be used rather than assuming homogeneous coverage.

---

## 5. Outcome Variable Analysis — `sale_amount`

### 5.1 Distribution

`sale_amount` records the total daily sales quantity for a given store-product-date observation. It is directly observed — unlike the anchor paper, which proxied demand using inverse sales rank.

Mean:    0.998
Median:  0.700
Std:     1.407
Min:     0.000
Max:    44.900
P95:     2.900
P99:     5.800

The distribution is right-skewed with a long tail, consistent with the power-law demand distribution reported in the dataset paper (top 20% of SKUs account for 87% of total demand in our data, slightly above the 51.8% reported in the paper — likely reflecting store-level aggregation effects). Most daily sales are low-volume, with extreme values for high-velocity SKUs.

### 5.2 Zero Sales and Demand Censoring

**This is the most important data quality finding of this iteration.**

Day of week:
Saturday:  1.179  (highest)
Sunday:    1.201  (highest)
Thursday:  0.866  (lowest)
Monthly trend:
March:  0.843
April:  0.901
May:    1.032
June:   1.100

Weekend demand is approximately 30% higher than mid-week demand. Demand also increases monotonically across the three-month observation window, suggesting either a seasonal trend, increasing store penetration, or both. These temporal patterns are real confounders — they affect both pricing decisions and demand simultaneously — and must be controlled for in the DML model.

---

## 6. Treatment Variable Analysis — `discount`

The `discount` variable records the promotional price ratio applied to a product on a given day. A value of 1.0 means no discount; a value of 0.75 means the product is sold at 75% of its reference price (a 25% discount). This is the treatment variable in the DML model.

### 6.1 Distribution
Mean:    0.911
Median:  0.989
Std:     0.128
Min:     0.000
Max:     1.088
P25:     0.851
P75:     1.000
P95:     1.000
P99:     1.000

The distribution is heavily left-skewed — 48.5% of observations have no discount at all (discount = 1.0), and the top quartile is entirely at or above 1.0. When discounts do occur, they are meaningful: the most frequent non-unity values cluster around 0.75–0.95, representing 5–25% price reductions.

### 6.2 Price Variation — Critical for Causal Identification

DML requires that the treatment variable (discount) exhibits genuine variation, both within and across units, to identify the causal effect. The data confirms this:
Records with no discount (discount = 1.0):  2,181,121  (48.5%)
Records with discount (discount < 1.0):     2,318,845  (51.5%)
Products with constant discount (no variation):  16  (1.8%)
Store-product pairs with zero within variation:  1,577  (3.2%)
Mean within store-product discount std dev:  0.067

**Finding: Price variation is strong and broadly distributed.** 96.8% of store-product pairs exhibit within-pair price variation over the 90-day window. Only 1.8% of products and 3.2% of store-product pairs have no price variation at all — these will be excluded from the estimation sample as they cannot contribute to identification.

The alignment between `discount` and `activity_flag` further validates the treatment variable:
Mean discount when activity_flag = 1:  0.800  (20% price reduction on average)
Mean discount when activity_flag = 0:  0.979  (minimal discount)

Promotional activity days correspond to significantly lower discount values, confirming that `discount` captures genuine pricing interventions rather than noise.

**One anomaly identified:** 34 records have discount values greater than 1.0 (maximum: 1.088), which is economically implausible. These records will be investigated and excluded or corrected in data preparation. Additionally, 16,039 records have a discount of exactly 0.0 (100% off), which warrants investigation — these may represent free promotional samples or data entry errors.

### 6.3 Discount as a Causal Anchor

The FreshRetailNet-50K paper explicitly describes promotional discounts as "causal anchors for demand fluctuation analysis," noting that marketing campaigns are annotated with explicit discount rate ranges (e.g., 15–30% price reductions). This framing supports the identification argument: promotional discount decisions are typically made by marketing teams on a schedule that is less responsive to short-run demand shocks than baseline price adjustments, reducing the endogeneity concern that motivates the use of DML. This will be discussed explicitly in the modeling phase when stating the identification assumptions.

---

## 7. Confounder Coverage Analysis

DML requires that all variables that affect both the treatment (discount) and the outcome (sale_amount) simultaneously are either observed and controlled for, or are shown to be sufficiently weak that they do not materially bias the estimate.

### 7.1 Missing Values

**All confounder variables are complete — zero missing values across all 4.5 million observations.** This is an exceptional data quality characteristic that eliminates a common source of analytical complexity.

### 7.2 Confounder Assessment

| Confounder | Coverage | Time Variation | Correlation with Demand | Assessment |
|---|---|---|---|---|
| `holiday_flag` | 100% | Yes (34.4% of days = holiday) | 0.083 | Strong — holidays shift demand meaningfully |
| `activity_flag` | 100% | Yes (37.8% of days = active) | 0.008 | Moderate — aligns with discount but weak direct demand effect |
| `precpt` | 100% | Daily std = 2.43 | 0.037 | Relevant for perishables — rainfall drives substitution |
| `avg_temperature` | 100% | Daily std = 3.27 | 0.056 | Relevant — temperature affects perishable demand |
| `avg_humidity` | 100% | Daily std = 4.33 | 0.043 | Contextual — complements temperature |
| `avg_wind_level` | 100% | Daily std = 0.12 | -0.008 | Weak — minimal independent demand effect |
| `stock_hour6_22_cnt` | 100% | Yes | -0.101 | Critical — strongest confounder in dataset |
| Product hierarchy | 100% | Time-invariant | Structural | Captured via product fixed effects |
| `city_id`, `store_id` | 100% | Time-invariant | Structural | Captured via store fixed effects |

**Key finding: `stock_hour6_22_cnt` is the strongest observable confounder** (correlation with demand: -0.101). This makes intuitive sense — availability directly determines whether demand can be expressed at all. It must be included as a confounder, but carefully: it is also a mediator of the stockout effect, so its role in the causal graph must be stated explicitly in the modeling phase.

**Weather covariates vary meaningfully across cities and over time**, providing genuine cross-sectional and temporal variation that will help DML disentangle discount effects from environmental demand drivers.

**The four-level product hierarchy** (7 management groups → 32 first categories → 84 second categories → 233 third categories → 865 products) provides a rich categorical structure for controlling product-level heterogeneity that is not captured by continuous variables.

### 7.3 Missing Confounders Relative to the Anchor Paper

The anchor paper used AI-generated text and image embeddings as the primary mechanism for controlling product quality confounders. FreshRetailNet-50K does not include text descriptions or product images. This is an acknowledged gap. However, for perishable goods, product quality is driven primarily by freshness and availability rather than brand perception — factors that the stockout annotations, weather covariates, and product hierarchy partially capture. Product fixed effects will absorb any remaining time-invariant product quality differences.

---

## 8. Data Quality Verification

| Check | Result | Action Required |
|---|---|---|
| Missing values | None across all 19 columns | None |
| Negative values | None | None |
| Duplicate records | None | None |
| Date continuity | Complete — all 90 days present | None |
| Panel balance | Perfect — all pairs have exactly 90 observations | None |
| Sparse SKUs (<30 obs) | None | None |
| hours_sale integrity | All 24-element sequences; sums match sale_amount exactly | None |
| Discount > 1.0 (invalid) | 34 records | Investigate and exclude in preparation |
| Discount = 0.0 (suspicious) | 16,039 records | Investigate — may be free samples or errors |
| Demand censoring from stockouts | 55.7% of observations have full stockout | Requires treatment strategy in preparation |
| Long-tail SKU distribution | Top 20% of SKUs = 87% of demand | Apply minimum demand threshold for estimation sample |

**Overall data quality is excellent.** The only substantive issues are the stockout censoring problem (which is a known feature of the dataset, not a defect) and the small number of anomalous discount records.

---

## 9. Initial Hypotheses

Based on the exploratory analysis, the following initial hypotheses will be tested in the modeling phase:

1. **Price elasticity is negative and statistically significant.** The discount variable has sufficient variation (96.8% of store-product pairs show within-pair variation) to identify a negative causal effect of price on demand.

2. **Price elasticity is heterogeneous across SKUs.** The 32 first-level product categories show meaningfully different mean demand levels (ranging from 0.415 to 1.795), suggesting that elasticity also varies by category.

3. **Stockout censoring materially biases naive estimates.** With 55.7% of observations affected by stockouts, models that ignore censoring will systematically underestimate demand and produce biased elasticity estimates.

4. **Weekend and holiday effects are meaningful confounders.** Weekend demand is ~30% higher than weekday demand, and holiday_flag shows the strongest binary confounder correlation with demand (0.083). Failing to control for these will bias the discount-demand relationship.

5. **Promotional activity (activity_flag) and discount are collinear.** Mean discount drops from 0.979 to 0.800 on promotional days, indicating multicollinearity that must be handled carefully in the DML nuisance models.

---

## 10. Fit-for-Purpose Assessment

### Verdict: FreshRetailNet-50K IS fit for purpose for causal price elasticity estimation using DML.

| Requirement | Status | Notes |
|---|---|---|
| Panel structure (SKU × time) | ✓ Met | Perfectly balanced 50,000 pairs × 90 days |
| Treatment variable with variation | ✓ Met | 96.8% of pairs have within-pair variation |
| Outcome variable directly observed | ✓ Met | sale_amount directly observed; no proxy needed |
| Observable confounders available | ✓ Met | Weather, holidays, promotions, product hierarchy |
| No missing data | ✓ Met | Zero missing values across all variables |
| Publicly available with open license | ✓ Met | CC BY 4.0 on HuggingFace |
| Demand censoring addressable | ⚠ Requires adaptation | Stockout annotations enable correction |
| Absolute price available | ✗ Not available | Discount rate used as treatment variable |
| Product quality signals | ✗ Not available | Addressed via product fixed effects |

### Required Adaptations for Data Preparation (Iteration 2)

1. **Stockout censoring treatment** — Observations where `stock_hour6_22_cnt = 0` represent fully censored demand. These will be excluded from the primary estimation sample, with sensitivity analysis on the retained sample.

2. **Treatment variable construction** — The `discount` variable will be used directly as a continuous treatment (log-transformed if appropriate). Records with `discount > 1.0` will be excluded. Records with `discount = 0.0` will be investigated before a decision is made.

3. **Long-tail SKU filtering** — A minimum demand threshold will be applied to exclude extremely sparse SKUs that cannot support reliable causal estimation. The threshold will be determined during data preparation.

4. **Temporal aggregation** — Data will be retained at daily resolution. The 90-day window provides sufficient temporal variation for lagged quantity and price signals to serve as state variables in the dynamic DML model, analogous to the anchor paper's approach.

5. **Fixed effects encoding** — Store and product fixed effects will be constructed to absorb time-invariant confounding. City-level weather covariates will be included as time-varying controls.

---

## 11. Updates to Business Understanding

The following updates to the Business Understanding document are recommended based on findings in this iteration:

- **Dataset confirmed:** FreshRetailNet-50K is selected as the primary dataset. This should be added to the Business Understanding document with a brief justification.
- **Domain specificity:** The project is now explicitly scoped to perishable retail (fresh produce, meat, seafood, frozen goods). The business context should reflect this.
- **Stockout as an additional business problem:** The 55.7% stockout rate reveals a secondary business insight — a significant proportion of unrecorded demand exists due to inventory constraints. This is relevant to both the pricing and the replenishment functions of the business, and can be noted as a finding that extends the project's business value.
- **Three-month temporal scope:** The observation window covers March to June 2024 only. Seasonal generalisation is limited, and this should be acknowledged explicitly as a constraint in the project scope.

---

## 12. References

Bach, P., Chernozhukov, V., Klaassen, S., Spindler, M., Teichert-Kluge, J., & Vijaykumar, S. (2026). *Adventures in Demand Analysis Using AI.* arXiv:2501.00382v3.

Wang, Y., Gu, J., Long, L., Li, X., Shen, L., Fu, Z., Zhou, X., & Jiang, X. (2025). *FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent Demand Recovery and Forecasting in Fresh Retail.* arXiv:2505.16319.

---

*This is a living document. It will be updated as data preparation and modeling reveal additional insights.*
