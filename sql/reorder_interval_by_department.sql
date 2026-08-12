-- Average / median days between orders, by department -- e.g. does
-- produce get reordered faster than pantry staples?
-- Uses MEDIAN in addition to AVG since order gaps are right-skewed
-- (a few very long gaps pull the mean up).
SELECT
    d.department_name,
    ROUND(AVG(o.days_since_prior), 2)    AS avg_days_between_orders,
    ROUND(MEDIAN(o.days_since_prior), 2) AS median_days_between_orders
FROM fact_orders o
JOIN fact_order_items foi ON o.order_id = foi.order_id
JOIN dim_products p       ON foi.product_id = p.product_id
JOIN dim_departments d    ON p.department_id = d.department_id
WHERE o.days_since_prior IS NOT NULL
GROUP BY d.department_name
ORDER BY avg_days_between_orders ASC;
