# The Business Problem

An online fashion retailer sells thousands of articles, each with a limited lifecycle of roughly **100 weeks**. At the end of the season, unsold stock is a direct loss — it either gets written off or sold at deep clearance discounts that destroy margin. Running out of stock too early is equally costly — lost sales and disappointed customers.

The retailer controls one lever to manage this: **price**. By adjusting discounts week by week, the pricing team tries to match the rate of sales to the rate of stock depletion, clearing inventory as close to season end as possible.

To set prices optimally, the pricing team needs to answer one question every week for every article:

> **If I set the discount to X% next week, how many units will I sell?**

This requires a demand forecasting model that can answer that question accurately — not just for the discount level the retailer has been using, but for **any discount level** the pricing team is considering, including levels that have rarely or never been applied to that article before.

# Why This Is Hard

The historical sales data the model trains on is deeply misleading. Discounts were **not applied randomly** — they were applied precisely when demand was already weak. As a result, the data shows:

- High discounts coinciding with low sales
- Low discounts coinciding with high sales

A naive model learns this corrupted pattern and concludes that discounting barely helps — or even hurts — demand.

When the pricing team asks:

> **What happens if I apply a 40% discount next week?**

the naive model gives a systematically wrong answer, and the pricing decision built on that answer destroys value.

# The Business Objective

Build a demand forecasting model that correctly estimates **how demand responds to price interventions**, including price levels not previously seen in the training data, so that the pricing team can make optimal discounting decisions that:

- Maximize revenue
- Clear inventory by season end
- Avoid unnecessary markdowns
- Reduce stockouts and lost sales

## Translate the Business Problem into a Precise Analytical Question

This step is **covered**.

We established the analytical question precisely:

> Given observable covariates **z** (seasonality, stock levels, demand trends, and article features) and a desired pricing intervention **do(discount)**, estimate:
>
> **E[demand | do(discount), z]**
>
> the expected demand under a pricing intervention, including interventions that were not observed in the historical training data.

The key challenge is not simply forecasting future demand, but estimating the **causal effect of price changes on demand** so that the model can generalize to discount levels that have rarely or never been applied before.

## Define What Success Looks Like

This step is **partially covered**.

From the paper, model performance is evaluated using the following metrics:

### Forecast Accuracy Metrics

* **MAE (Mean Absolute Error)** — measures the average magnitude of forecast errors.
* **MSE (Mean Squared Error)** — penalizes larger forecasting errors more heavily than MAE.

### Business-Oriented Metric

* **Demand Error** — a downstream pricing metric that weights forecasting errors by the recommended retail price, reflecting the business impact of incorrect pricing decisions.

### Causal Estimation Metrics

* **MAE Effect** — measures how accurately the model recovers the true price elasticity relative to synthetic ground truth.
* **MSE Effect** — measures squared error in elasticity estimation relative to synthetic ground truth.

Together, these metrics evaluate both:

1. **Predictive performance** (how accurately demand is forecasted), and
2. **Causal performance** (how accurately the model estimates the effect of price interventions on demand).

