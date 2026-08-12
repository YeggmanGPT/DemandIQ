-- Top 3 best-selling products within each department.
-- DENSE_RANK (rather than ROW_NUMBER) so genuine ties share a rank
-- instead of one winning arbitrarily.
WITH ranked_products AS (
    SELECT
        d.department_name,
        p.product_name,
        COUNT(foi.order_id) AS order_count,
        DENSE_RANK() OVER (
            PARTITION BY d.department_id ORDER BY COUNT(foi.order_id) DESC
        ) AS rank
    FROM fact_order_items foi
    JOIN dim_products p    ON foi.product_id = p.product_id
    JOIN dim_departments d ON p.department_id = d.department_id
    GROUP BY d.department_id, d.department_name, p.product_name
)
SELECT department_name, product_name, order_count
FROM ranked_products
WHERE rank <= 3
ORDER BY department_name, rank;
