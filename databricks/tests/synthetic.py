"""Small Instacart-shaped CSVs for tests (same files, columns and quirks).

Quirks reproduced on purpose because the pipeline has to handle them:
  * days_since_prior_order is empty for a user's first order and written
    as "8.0" (float text) otherwise
  * 'test' orders exist in orders.csv but have no rows in either
    order_products file
  * order_dow follows the gaps (dow_n = dow_{n-1} + gap mod 7)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

DEPARTMENTS = [
    "frozen", "other", "bakery", "produce", "alcohol", "international", "beverages", "pets",
    "dry goods pasta", "bulk", "personal care", "meat seafood", "pantry", "breakfast",
    "canned goods", "dairy eggs", "household", "babies", "snacks", "deli", "missing",
]


def generate(out_dir: str, n_users: int = 1200, n_products: int = 600, seed: int = 7) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    departments = pd.DataFrame({"department_id": range(1, 22), "department": DEPARTMENTS})
    aisles = pd.DataFrame({"aisle_id": range(1, 135), "aisle": [f"aisle {i}" for i in range(1, 135)]})
    aisle_dept = {a: (a - 1) % 21 + 1 for a in aisles["aisle_id"]}
    product_aisle = rng.integers(1, 135, size=n_products)
    products = pd.DataFrame({
        "product_id": range(1, n_products + 1),
        # commas and embedded quotes force CSV quoting/escaping, as in the real products.csv
        "product_name": [f'Product {i}, size {i % 5}' + (' "Family Pack"' if i % 7 == 0 else '')
                         for i in range(1, n_products + 1)],
        "aisle_id": product_aisle,
        "department_id": [aisle_dept[a] for a in product_aisle],
    })
    popularity = 1.0 / np.arange(1, n_products + 1) ** 0.8
    popularity /= popularity.sum()

    orders, prior_lines, train_lines = [], [], []
    order_id = 1
    for user_id in range(1, n_users + 1):
        n_orders = int(rng.integers(4, 30))
        last_split = "train" if rng.random() < 0.7 else "test"
        dow, bought = int(rng.integers(0, 7)), set()
        for number in range(1, n_orders + 1):
            gap = None if number == 1 else int(rng.integers(3, 31))
            if gap is not None:
                dow = (dow + gap) % 7
            split = last_split if number == n_orders else "prior"
            orders.append({
                "order_id": order_id, "user_id": user_id, "eval_set": split, "order_number": number,
                "order_dow": dow, "order_hour_of_day": int(rng.integers(6, 23)),
                "days_since_prior_order": "" if gap is None else f"{gap}.0",
            })
            if split != "test":
                basket = rng.choice(n_products, size=int(rng.integers(1, 12)), replace=False, p=popularity) + 1
                target = prior_lines if split == "prior" else train_lines
                for position, product_id in enumerate(basket, start=1):
                    target.append({"order_id": order_id, "product_id": int(product_id),
                                   "add_to_cart_order": position, "reordered": int(product_id in bought)})
                    bought.add(int(product_id))
            order_id += 1

    frames = {
        "departments.csv": departments,
        "aisles.csv": aisles,
        "products.csv": products,
        "orders.csv": pd.DataFrame(orders),
        "order_products__prior.csv": pd.DataFrame(prior_lines),
        "order_products__train.csv": pd.DataFrame(train_lines),
    }
    for name, frame in frames.items():
        frame.to_csv(os.path.join(out_dir, name), index=False)
    return {name: len(frame) for name, frame in frames.items()}


def append_rows(out_dir: str, filename: str, rows: list[dict]) -> None:
    """Append raw text rows (values written as given) to one of the CSVs."""
    path = os.path.join(out_dir, filename)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    pd.DataFrame(rows, columns=header).to_csv(path, mode="a", header=False, index=False)
