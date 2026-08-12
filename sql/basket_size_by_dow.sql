-- Average items per order, by day of week. Instacart encodes order_dow
-- as 0-6 but doesn't document which end is which -- treat "day_name"
-- below as a labeled guess (0=Sunday), and say so if asked in interviews.
SELECT
    o.order_dow,
    CASE o.order_dow
        WHEN 0 THEN 'Sunday'
        WHEN 1 THEN 'Monday'
        WHEN 2 THEN 'Tuesday'
        WHEN 3 THEN 'Wednesday'
        WHEN 4 THEN 'Thursday'
        WHEN 5 THEN 'Friday'
        WHEN 6 THEN 'Saturday'
    END AS day_name,
    COUNT(DISTINCT o.order_id)                                            AS total_orders,
    COUNT(foi.product_id)                                                 AS total_items,
    ROUND(COUNT(foi.product_id) * 1.0 / COUNT(DISTINCT o.order_id), 2)    AS avg_basket_size
FROM fact_orders o
JOIN fact_order_items foi ON o.order_id = foi.order_id
GROUP BY o.order_dow
ORDER BY o.order_dow ASC;
