-- Department demand by hour of day -- feeds the "when does this
-- department peak" question, e.g. for the SQL Explorer dashboard page.
SELECT
    d.department_name,
    o.order_hour,
    COUNT(foi.product_id) AS hourly_orders
FROM fact_order_items foi
JOIN fact_orders o     ON foi.order_id = o.order_id
JOIN dim_products p    ON foi.product_id = p.product_id
JOIN dim_departments d ON p.department_id = d.department_id
GROUP BY d.department_name, o.order_hour
ORDER BY d.department_name, o.order_hour;
