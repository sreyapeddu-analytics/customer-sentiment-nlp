# Dashboard — Customer Sentiment KPIs

This folder documents the KPI layer a BI dashboard (Power BI / Tableau) for
this analysis would expose. Charts in the repo's `images/` folder can be
dropped into a dashboard tool as the visual layer.

## KPI definitions

| KPI | Definition | Grain | Refresh |
|-----|-----------|-------|---------|
| **Avg rating** | `AVG(rating)` | product category, month | on data refresh |
| **% positive** | reviews with rating ≥ 4 ÷ all reviews | product category, month | on data refresh |
| **% negative** | reviews with rating ≤ 2 ÷ all reviews | product category, month | on data refresh |
| **Review volume** | `COUNT(review_id)` | product category, month | on data refresh |
| **Avg review length** | mean word count of cleaned review text | rating band | on data refresh |
| **Top complaint themes** | highest-weight negative n-grams from the logistic regression model (see notebook) | overall / category | on model retrain |
| **Top praise themes** | highest-weight positive n-grams from the logistic regression model | overall / category | on model retrain |

## Suggested dashboard pages

1. **Overview** — avg rating trend, % positive/% negative by category, review volume by month.
2. **Voice of customer** — top praise vs complaint n-grams (bar charts), word-level drivers per category.
3. **Model performance** — confusion matrix + per-class precision/recall/F1 from the latest model run.

## Notes

- All source data is synthetic (see `data/README.md`); KPIs are computed, not collected.
- Sentiment labels are derived from star ratings, not from a human-labeled gold set —
  treat model metrics as pipeline-demonstration numbers, not production claims.
