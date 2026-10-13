-- analysis_queries.sql
-- SQLite-compatible exploratory queries over data/reviews.csv
-- (import the CSV into a table named `reviews` first).
--
-- Example import:
--   sqlite3 reviews.db
--   .mode csv
--   .import --skip 1 data/reviews.csv reviews

-- 1. Average rating and review count by product category
SELECT
    product_category,
    COUNT(*)                        AS review_count,
    ROUND(AVG(rating), 2)            AS avg_rating,
    ROUND(AVG(rating), 2)            AS avg_rating_dup_check
FROM reviews
GROUP BY product_category
ORDER BY avg_rating DESC;

-- 2. Sentiment share by product category
--    (sentiment derived from rating: 1-2 negative, 3 neutral, 4-5 positive)
SELECT
    product_category,
    COUNT(*) AS total_reviews,
    ROUND(100.0 * SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_positive,
    ROUND(100.0 * SUM(CASE WHEN rating = 3  THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_neutral,
    ROUND(100.0 * SUM(CASE WHEN rating <= 2 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_negative
FROM reviews
GROUP BY product_category
ORDER BY pct_negative DESC;

-- 3. Review count and average rating by star rating (distribution check)
SELECT
    rating,
    COUNT(*)             AS review_count,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM reviews), 1) AS pct_of_reviews
FROM reviews
GROUP BY rating
ORDER BY rating;

-- 4. Lowest-rated categories: average rating below 3.5 (watchlist)
SELECT
    product_category,
    ROUND(AVG(rating), 2) AS avg_rating,
    COUNT(*)              AS review_count
FROM reviews
GROUP BY product_category
HAVING AVG(rating) < 3.5
ORDER BY avg_rating;

-- 5. Negative-review volume by category and month (trend input for a dashboard)
SELECT
    product_category,
    SUBSTR(review_date, 1, 7) AS review_month,
    COUNT(*)                  AS negative_reviews
FROM reviews
WHERE rating <= 2
GROUP BY product_category, review_month
ORDER BY review_month, negative_reviews DESC;
