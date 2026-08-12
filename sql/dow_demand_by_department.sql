-- Department demand pivoted across day of week (0=Sunday ... 6=Saturday)
-- -- a quick way to spot which departments have a weekday vs weekend skew.
SELECT
    d.department_name,
    SUM(CASE WHEN o.order_dow = 0 THEN 1 ELSE 0 END) AS sun,
    SUM(CASE WHEN o.order_dow = 1 THEN 1 ELSE 0 END) AS mon,
    SUM(CASE WHEN o.order_dow = 2 THEN 1 ELSE 0 END) AS tue,
    SUM(CASE WHEN o.order_dow = 3 THEN 1 ELSE 0 END) AS wed,
    SUM(CASE WHEN o.order_dow = 4 THEN 1 ELSE 0 END) AS thu,
    SUM(CASE WHEN o.order_dow = 5 THEN 1 ELSE 0 END) AS fri,
    SUM(CASE WHEN o.order_dow = 6 THEN 1 ELSE 0 END) AS sat,
    COUNT(foi.product_id)                             AS total_demand
FROM fact_order_items foi
JOIN fact_orders o      ON foi.order_id = o.order_id
JOIN dim_products p     ON foi.product_id = p.product_id
JOIN dim_departments d  ON p.department_id = d.department_id
GROUP BY d.department_name
ORDER BY total_demand DESC;
