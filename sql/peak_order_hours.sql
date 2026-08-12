-- What time of day do people order? Useful for staffing / delivery-window
-- planning questions.
SELECT
    order_hour,
    COUNT(order_id)                                                AS total_orders,
    ROUND(COUNT(order_id) * 100.0 / SUM(COUNT(order_id)) OVER (), 2) AS pct_of_total
FROM fact_orders
GROUP BY order_hour
ORDER BY order_hour ASC;
