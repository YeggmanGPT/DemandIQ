-- Which departments get reordered the most (loyalty/staple signal)?
-- reorder_rate_pct = share of line items in that department that were
-- a repeat purchase (is_reordered = 1) rather than a first-time buy.
SELECT
    d.department_name,
    COUNT(foi.product_id)                                           AS total_item_orders,
    SUM(foi.is_reordered)                                            AS total_reorders,
    ROUND(SUM(foi.is_reordered) * 100.0 / COUNT(foi.product_id), 2)  AS reorder_rate_pct
FROM fact_order_items foi
JOIN dim_products p     ON foi.product_id = p.product_id
JOIN dim_departments d  ON p.department_id = d.department_id
GROUP BY d.department_name
ORDER BY reorder_rate_pct DESC;
