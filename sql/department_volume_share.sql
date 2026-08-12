-- Each department's share of total item volume across the whole dataset.
SELECT
    d.department_name,
    COUNT(foi.order_id) AS item_volume,
    ROUND(COUNT(foi.order_id) * 100.0 / (SELECT COUNT(*) FROM fact_order_items), 2) AS volume_share_pct
FROM fact_order_items foi
JOIN dim_products p    ON foi.product_id = p.product_id
JOIN dim_departments d ON p.department_id = d.department_id
GROUP BY d.department_name
ORDER BY item_volume DESC;
