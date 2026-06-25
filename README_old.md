# causal-price-demand
Estimated the causal effect of price and discounts on SKU-level demand using FreshRetailNet‑50K. Applied propensity score and modern causal inference methods to correct for endogeneity, quantify true price elasticity, evaluate SKU/store-level patterns, and generate actionable insights with reproducible Python workflows and visualizations.

# Causal Price Elasticity Estimation for SKU-Level Demand

## 📌 Overview
This project estimates **causal price elasticity at the SKU level** using observational retail data.  
Traditional demand models often produce **biased estimates** because pricing decisions are endogenous (i.e., correlated with demand shocks).  

To address this, we apply **Double/Debiased Machine Learning (DML)** to isolate the true causal effect of price on demand and generate **actionable pricing insights**.

---

## 🎯 Business Objective
Improve pricing decisions to:
- Maximize revenue  
- Optimize discount strategies  
- Reduce waste for perishable goods  

---

## 💼 Business Process

### 1. Problem
Retailers rely on historical data to estimate demand. However:
- Prices are **not randomly assigned**
- They respond to demand, promotions, and inventory levels  

This creates **endogeneity**, leading to:
- Incorrect elasticity estimates  
- Poor pricing decisions  
- Revenue loss  

---

### 2. Key Business Questions
- How sensitive is demand to price changes?
- Which SKUs should be discounted?
- Where are we overpricing or underpricing?

---

### 3. Decision Lever
- SKU-level pricing  
- Discount timing and depth  

---

### 4. Success Metrics (KPIs)
- Revenue uplift (simulated)  
- Margin improvement  
- Inventory turnover  
- Reduction in pricing inefficiencies  

---

### 5. Business Impact
- More accurate elasticity estimates  
- Improved pricing strategies  
- Better handling of perishable inventory  
- Data-driven revenue optimization  

---

## ⚙️ Technical Process

### 1. Dataset
**FreshRetailNet-50K — Perishable Retail Sales & Price Data**

Contains:
- SKU-level sales (demand)
- Prices (treatment variable)
- Time, promotions, and contextual features  

---

### 2. Core Challenge: Endogeneity
Price is correlated with demand shocks:
- High demand → higher prices  
- Promotions → lower prices + higher demand  

Naive models mistake correlation for causation.

---

### 3. Baseline Model (Naive Approach)
- Ordinary Least Squares (OLS)
- Assumes price is exogenous ❌  

Used to demonstrate:
- Bias in standard demand estimation  

---

### 4. Causal Methodology

**Reference:**  
Victor Chernozhukov et al. (2018)  
*Double/Debiased Machine Learning for Treatment and Structural Parameters*

#### Approach:
1. Model price using controls (ML model)
2. Model demand using controls (ML model)
3. Residualize both (remove confounding)
4. Estimate causal effect using residuals

#### Techniques:
- Double Machine Learning (DML)  
- Cross-fitting  
- Regularization / ML-based nuisance models  

---

### 5. Elasticity Estimation
Convert causal effect into:

Elasticity = (dQ/dP) * (P / Q)

Computed at:
- SKU level  
- Category level  

---

### 6. Validation
- Compare OLS vs DML estimates  
- Check:
  - Sign (typically negative)  
  - Magnitude (economically plausible)  
- Stability across time and SKUs  

---

### 7. Pricing Simulation
- Apply estimated elasticities  
- Simulate new pricing strategies  
- Measure potential revenue impact  

---

## ⚠️ Challenges & Solutions

### Challenge 1: Endogeneity
- Price influenced by demand  

**Solution:**  
- Double Machine Learning (DML)

---

### Challenge 2: High-Dimensional Controls
- Many SKUs, time effects, features  

**Solution:**  
- Machine learning models for nuisance estimation  

---

### Challenge 3: Missing Confounders
- Limited visibility into all drivers (e.g., competitor pricing)

**Solution:**  
- Proxy variables:
  - Time fixed effects  
  - SKU/store controls  

---

### Challenge 4: Interpretability
- ML models are complex  

**Solution:**  
- Translate results into elasticity (business-friendly metric)

---

## 🔗 Connecting Business & Technical Work

| Business Need | Technical Solution |
|------|------|
| Improve pricing decisions | Estimate elasticity |
| Remove bias in demand | Handle endogeneity |
| Extract reliable insights | Use causal inference |
| Increase revenue | Simulate pricing strategies |

---

## 🧠 Key Takeaways
- Correlation ≠ causation in pricing  
- Endogeneity is a critical issue in demand estimation  
- DML enables causal inference in high-dimensional settings  
- Elasticity is the bridge between modeling and business decisions  

---

## 🚀 Future Improvements
- Incorporate competitor pricing data  
- Model dynamic pricing strategies  
- Use causal forests for heterogeneous treatment effects  
- Deploy real-time pricing recommendations  

---

## 🛠️ Tech Stack
- Python  
- pandas / numpy  
- scikit-learn  
- statsmodels  
- econml / doubleml (optional)  

---

## 📖 References
- Chernozhukov, V. et al. (2018)  
  *Double/Debiased Machine Learning for Treatment and Structural Parameters*

---

## 📬 Summary (Interview-Ready)
This project addresses bias in traditional demand models by applying causal inference techniques to estimate **true price elasticity**.  
The results enable **data-driven pricing decisions** that can improve revenue and operational efficiency in retail settings.
