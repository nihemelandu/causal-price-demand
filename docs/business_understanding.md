# Business Understanding: Causal Price Elasticity Estimation for SKU-Level Demand

---

## 1. Business Context

A mid-sized omnichannel retailer operates across physical stores and a growing e-commerce platform. Like most modern retailers, the business manages thousands of SKUs across multiple product categories, with prices that change frequently in response to promotions, seasonality, competitor activity, and inventory levels.

Commercial and pricing teams make SKU-level pricing decisions daily. These decisions are consequential: price too high and demand drops, leaving revenue on the table; price too low and margin erodes. Getting this balance right requires understanding how demand actually responds to price changes — not just how price and demand are correlated in historical data, but what the true causal effect of a price change on demand is.

The business has an existing demand forecasting model that generates SKU-level sales volume predictions. These forecasts inform replenishment, inventory planning, and promotional planning. However, the forecasting model was built on historical observational data without explicitly accounting for the causal relationship between price and demand. This is a silent but serious limitation: when prices change — which is precisely when accurate forecasts matter most — the model's predictions become unreliable, because it was trained on historical pricing patterns rather than on how demand truly responds to price.

An independent applied data scientist has been engaged to address this gap by developing a proof of concept that estimates the causal effect of price on demand at the SKU level, using observational sales data. The insights generated will inform both pricing decisions and the improvement of the existing demand forecasting model.

---

## 2. The Problem

### 2.1 The Core Business Problem

The business cannot reliably answer the question: *"If we change the price of a SKU, what will happen to demand?"*

This is not a data availability problem. The business has rich historical sales and pricing data. It is an analytical problem: the relationship between price and demand in observational data is confounded. Factors such as product quality, promotional activity, seasonality, and competitor pricing affect both the price set and the demand observed simultaneously. A naive analysis of this data will produce a biased estimate of the price-demand relationship — and acting on a biased estimate leads to systematically wrong pricing decisions.

### 2.2 Why Standard Approaches Fall Short

A standard regression of sales on price — the most common analytical approach — does not isolate the causal effect of price on demand. It captures correlation, not causation. In practice, this produces elasticity estimates that are either implausibly small (suggesting price barely affects demand) or directionally misleading. Both outcomes are dangerous when used to inform pricing strategy.

The existing demand forecasting model compounds this problem. Because it was built on observational data without causal correction, it has implicitly learned the old pricing policy rather than the true demand response to price. When pricing decisions deviate from historical patterns — which is the whole point of pricing optimization — the forecast model's predictions degrade in exactly the scenarios where they are needed most. This is known as the off-policy prediction problem.

### 2.3 The Opportunity

Correcting for confounding in observational sales data using causal inference methods produces price elasticity estimates that are more accurate, more credible, and more actionable. These estimates:

- Tell the pricing team how demand will genuinely respond to a price change, at the SKU level
- Reveal which SKUs are price-sensitive and which are not, enabling differentiated pricing strategies
- Improve the reliability of the demand forecasting model under changing price conditions
- Provide a principled, defensible basis for pricing recommendations that can be communicated to leadership

---

## 3. Strategic and Technical Goals

### 3.1 Strategic Goal

To provide the pricing and commercial teams with reliable, SKU-level estimates of how demand responds to price changes — derived from observational sales data using rigorous causal inference methods — so that pricing decisions are grounded in evidence rather than assumption, and the existing demand forecasting model is made more trustworthy under off-policy pricing conditions.

### 3.2 Technical Goal

To apply Double/Debiased Machine Learning (DML) to estimate the causal effect of price on SKU-level demand from observational sales data, explicitly controlling for confounders, quantifying heterogeneity in price elasticity across SKUs, and assessing the sensitivity of results to unobserved confounding — producing a reproducible, well-documented Python workflow that serves as a proof of concept for broader adoption.

---

## 4. Methodological Approach

This project adopts a causal inference framework rather than a purely predictive one. The distinction is critical: predictive models learn associations in historical data; causal models isolate the effect of a specific variable — in this case, price — on an outcome, holding all else equal.

The chosen method is **Double/Debiased Machine Learning (DML)**, introduced by Chernozhukov et al. (2018) and applied to demand analysis in the anchor reference paper *Adventures in Demand Analysis Using AI* (Bach, Chernozhukov et al., 2026). DML is particularly well suited to this problem because it:

- Handles high-dimensional confounders using flexible machine learning methods
- Produces valid statistical inference on the causal price effect despite using machine learning in intermediate steps
- Corrects for the endogeneity of price in observational data
- Supports estimation of both average and heterogeneous price elasticity across SKUs

The anchor paper demonstrates that naive regression yields implausibly small and biased elasticity estimates, and that DML with appropriate confounding control produces more credible and economically meaningful results. This project replicates and adapts that methodology in an omnichannel retail context.

---

## 5. Stakeholders

| Stakeholder | Role | What They Need From This Project |
|---|---|---|
| **Pricing & Revenue Team** | Primary consumer of outputs | Reliable SKU-level elasticity estimates to inform pricing decisions |
| **Category Managers** | Commercial decision makers | Understanding of which SKUs and categories are price-sensitive |
| **Data Science Lead** | Internal champion and quality gatekeeper | Methodological rigor, reproducibility, and clear technical documentation |
| **Finance Director** | Business impact evaluator | Confidence that pricing recommendations will improve revenue and margin |
| **General Manager** | Executive sponsor | A clear bottom-line narrative: what is the business impact and is it worth pursuing at scale? |

---

## 6. What Success Looks Like

Success is defined at two levels: business outcomes and technical quality. Both must be satisfied for the project to be considered successful.

### 6.1 Business KPIs

These measure whether the project delivers real business value:

- **Revenue uplift from informed pricing** — do pricing decisions guided by causal elasticity estimates generate more revenue than the status quo?
- **Gross margin improvement** — are we pricing to protect margin, not just drive volume?
- **Forecast reliability under price changes** — does incorporating causal price effects improve forecast accuracy when prices change, compared to the existing model?
- **Pricing decision adoption rate** — are the pricing and commercial teams actually using the elasticity estimates? Adoption is the ultimate signal of trust and utility.
- **Revenue at risk quantified** — can we demonstrate how much revenue is being left on the table by the current approach, creating a compelling case for broader adoption?

### 6.2 Technical Metrics

These measure whether the model is doing what we claim it does:

- **Statistical significance and plausibility of elasticity estimates** — are estimates negative (as economic theory requires), statistically significant, and within an economically reasonable range?
- **Bias reduction from causal correction** — how much does the DML estimate differ from the naive regression estimate? A meaningful difference validates the need for the causal approach.
- **Nuisance model performance (R²)** — how well do the intermediate machine learning models predict price and quantity after controlling for confounders? Poor nuisance model performance undermines the causal estimate.
- **Residual orthogonality** — are the partialled-out residuals of price and quantity genuinely uncorrelated with confounders, as DML requires?
- **Heterogeneity in elasticity across SKUs** — is there statistically significant variation in price sensitivity across SKUs, or is a single average elasticity sufficient?
- **Sensitivity to unobserved confounding** — how strong would unobserved confounding need to be to overturn the findings? This follows the sensitivity analysis framework of Chernozhukov et al. (2021).

---

## 7. Project Scope

### In Scope
- Estimating causal price elasticity at the SKU level using observational sales data
- Controlling for observable confounders using DML
- Estimating both homogeneous (average) and heterogeneous (SKU-varying) elasticity
- Sensitivity analysis for unobserved confounding
- A reproducible Python workflow hosted on GitHub

### Out of Scope (for this proof of concept)
- Real-time or dynamic pricing optimization
- Competitor pricing data integration
- Causal estimation using randomized price experiments
- Full production deployment

---

## 8. Assumptions and Constraints

- Observational sales data contains sufficient price variation across SKUs and time periods to support causal identification
- Key confounders — such as promotions, seasonality, and product characteristics — are observable in the data and can be controlled for
- The causal effect of price on demand is assumed to be stable over the period of analysis (no structural breaks)
- This is a proof of concept; findings will require validation before informing live pricing decisions

---

## 9. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Insufficient price variation in data | Medium | High | Assess price variation in data understanding phase; consider alternative identification strategies |
| Key confounders unobserved | Medium | High | Conduct sensitivity analysis to bound the impact of unobserved confounding |
| Elasticity estimates not actionable at SKU level | Low | Medium | Begin with category-level estimation; disaggregate to SKU as data supports |
| Stakeholder distrust of causal methods | Medium | Medium | Communicate findings clearly; show comparison between naive and causal estimates |

---

## 10. Reference

Bach, P., Chernozhukov, V., Klaassen, S., Spindler, M., Teichert-Kluge, J., & Vijaykumar, S. (2026). *Adventures in Demand Analysis Using AI.* arXiv:2501.00382v3. Also available as CeMMAP Working Paper CWP01/25.

---

*This is a living document. It will be updated as project understanding deepens through subsequent iterations.*
