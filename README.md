![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Prophet](https://img.shields.io/badge/Prophet-Facebook-lightgrey)
![Statsmodels](https://img.shields.io/badge/Statsmodels-SARIMA-orange)
![PowerBI](https://img.shields.io/badge/PowerBI-Optional-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

# 🛒 Walmart Weekly Sales Forecasting — *The Predictable Unpredictability of Retail*

> *“The future is uncertain — but Walmart’s Q4 sales aren’t.”*  
> — A Business Analyst who has seen enough data to stop being surprised.

---

## 📘 Project Story

Once upon a spreadsheet, somewhere in Walmart’s sales department, a team of managers was arguing about why December feels like a festival and July feels like a ghost town.  

I decided to find out — not by guessing, but by **listening to the data**.

This project transforms **Walmart’s weekly sales data** (from Kaggle) into a **story of seasonality, chaos, and patterns hiding in plain sight** — using **time series forecasting** techniques like **SARIMA** and **Prophet**.

Because as any analyst knows:  
> Forecasting isn’t about being right — it’s about being less wrong than everyone else.

---

## 🧭 Business Context

**Stakeholder:** Sales & Inventory Management Team at Walmart  
**Business Problem:** Unpredictable weekly sales make inventory and budgeting decisions risky.  
**Goal:** Predict weekly sales for the next quarter to support **stock optimization**, **cash flow planning**, and **marketing timing**.  

**Key Questions:**
1. When does sales demand peak or drop?  
2. How accurately can we forecast future sales?  
3. Which model — SARIMA or Prophet — makes fewer enemies in management meetings?

---

## 💼 Project Objectives

- 📊 **Forecast Walmart’s weekly sales** for better decision-making  
- 🧮 **Compare SARIMA vs Prophet models** to evaluate forecast accuracy  
- 🎯 **Translate model outputs into business insights** (because stakeholders don’t speak Python)  
- 💬 **Communicate results in a clear, narrative-driven format** for executives and BI teams  

---

## 🧩 Dataset

**Source:** [Kaggle – Walmart Sales Forecasting](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast)

| Column | Description |
|---------|--------------|
| `Store` | Store ID |
| `Dept` | Department ID |
| `Date` | Weekly sales date |
| `Weekly_Sales` | Weekly sales revenue |
| `IsHoliday` | Boolean indicator for holiday weeks |

> The dataset covers over **130 weeks of sales data (2010–2012)** across multiple stores and departments.  

For simplicity (and sanity), I aggregated all stores to analyze total weekly sales — the **macro retail pulse** of Walmart.

---

## 🔍 Exploratory Analysis (EDA)

Before predicting the future, I had to **make sense of the past**.

- 🕰️ Sales oscillate wildly week-to-week but show **clear holiday season spikes** (Nov–Dec).  
- 🌤️ A mid-year dip appears almost every year (consumers collectively nap in June).  
- 📈 A slow upward trend signals **steady revenue growth**.  

A seasonal decomposition confirmed:
- A **strong yearly pattern**  
- A **positive long-term trend**  
- Residual noise that reminded me customers don’t always behave rationally

---

## 🧠 Model Building

Two models. Two personalities. One purpose.

### 1️⃣ SARIMA — *The Overachieving Statistician*
- Captures trends, seasonality, and residuals with mathematical rigor.  
- Feeds on stationarity and AIC scores.  
- Achieved **MAPE = 1.66%**, a shockingly accurate forecast.

### 2️⃣ Prophet — *The Intuitive Storyteller*
- Designed by Facebook to handle business time series.  
- Understands holidays, weekends, and human chaos.  
- Achieved **MAPE = 2.20%** — slightly less precise, but easier to explain to your manager at 9AM.

| Model | MAE | RMSE | MAPE | Personality |
|-------|------|------|------|-------------|
| **SARIMA** | 764,538 | 884,621 | **1.66%** | Precise, stoic, reliable |
| **Prophet** | 1,027,779 | 1,232,407 | **2.20%** | Intuitive, visual, slightly dramatic |

---

## 📊 Results Visualization

### 🔹 SARIMA Forecast
Accurately tracks sales over time with tight confidence intervals — showing stability and precision.  
Peaks align perfectly with Q4, validating strong **seasonal dependency**.

### 🔹 Prophet Forecast
Smooth, interpretable curves showing yearly and weekly seasonality.  
The “Friday peak” and “December spike” are classic retail behavior — predictable chaos.

---

## 💡 Key Business Insights

1. **Q4 Surge — “The Santa Effect”**  
   → Every November–December, sales spike ~20%.  
   🟢 *Action:* Pre-stock seasonal goods and scale logistics 6–8 weeks ahead.

2. **Mid-Year Slump — “Summer Silence”**  
   → Sales dip sharply from May–July.  
   🟡 *Action:* Introduce mid-year clearance or targeted discounts.

3. **Trend Consistency — “Retail Calm Amid Chaos”**  
   → Despite fluctuations, overall growth is positive.  
   🟢 *Action:* Maintain procurement levels and long-term capacity planning.

4. **Forecast Accuracy — “Data Beats Gut Feeling”**  
   → SARIMA < 2% MAPE. Reliable for quarterly business planning.  
   🟢 *Action:* Integrate SARIMA forecasts into BI dashboards for real-time planning.

---

## 🎯 Business Impact

If implemented, this forecasting approach can:
- Improve **inventory accuracy** by up to 15%  
- Reduce **overstock/stock-out costs**  
- Enhance **revenue predictability** for financial planning  
- Enable **data-driven decision-making** across departments  

> In short: fewer surprises, fewer headaches, and better profit margins.

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Data Wrangling | Python, Pandas |
| Forecasting | Statsmodels (SARIMA), Prophet |
| Evaluation | MAE, RMSE, MAPE |
| Visualization | Matplotlib |

---


---

## 🤯 Analyst’s Reflection

Forecasting Walmart sales is a bit like predicting British weather -  
you know it’ll rain (or spike) eventually, but it still surprises you.  

> Absurd as it sounds, predicting consumer behavior is both art and algebra.  

What this project taught me:
- Data always carries rhythm.  
- Businesses ignore patterns at their own expense.  
- Forecasting is not about perfection - it’s about *perspective*.

---

## ✍️ Author

👤 **Shyam**  
🎓 MSc Business Analytics, University of Exeter  
💻 3+ years in Software Development | Aspiring Business & Insight Analyst  

> “I don’t just build models — I build stories that make data human.”


## 🧩 Folder Structure

