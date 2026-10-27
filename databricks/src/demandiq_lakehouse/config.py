"""Names, paths and constants shared by every layer of the lakehouse.

Everything that differs between environments (local tests, the dev
target, the prod target) lives in ``LakehouseConfig`` and arrives as job
parameters, so the transformation code itself never hard-codes a
catalog, schema or path.
"""

from __future__ import annotations

from dataclasses import dataclass

LAYERS = ("bronze", "silver", "gold")

# Bronze table name -> (CSV file in the landing volume, expected header).
# The header is checked on read (enforceSchema=False), so a renamed or
# wrong file fails loudly at ingestion instead of loading shifted columns.
RAW_SOURCES: dict[str, tuple[str, list[str]]] = {
    "orders_raw": (
        "orders.csv",
        ["order_id", "user_id", "eval_set", "order_number", "order_dow",
         "order_hour_of_day", "days_since_prior_order"],
    ),
    "order_products_prior_raw": (
        "order_products__prior.csv",
        ["order_id", "product_id", "add_to_cart_order", "reordered"],
    ),
    "order_products_train_raw": (
        "order_products__train.csv",
        ["order_id", "product_id", "add_to_cart_order", "reordered"],
    ),
    "products_raw": (
        "products.csv",
        ["product_id", "product_name", "aisle_id", "department_id"],
    ),
    "aisles_raw": ("aisles.csv", ["aisle_id", "aisle"]),
    "departments_raw": ("departments.csv", ["department_id", "department"]),
}

# Date reconstruction (same idea as features.py in the DuckDB version):
# Instacart only releases relative timing, so each user gets a
# deterministic pseudo-random anchor day and their real gap-days are
# walked forward from it. See README "Estimated order dates".
SIM_START = "2024-01-01"
ANCHOR_WINDOW_DAYS = 90
ANCHOR_SEED = 42

# Anomaly thresholds, same as anomaly.py / BUILD_LOG.md.
MEDIUM_THRESHOLD_PCT = 15.0
HIGH_THRESHOLD_PCT = 25.0

DAY_NAMES = {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday",
             4: "Thursday", 5: "Friday", 6: "Saturday"}


def _quote(identifier: str) -> str:
    return "`" + identifier.replace("`", "``") + "`"


@dataclass(frozen=True)
class LakehouseConfig:
    catalog: str = "workspace"
    schema_prefix: str = "demandiq_"
    landing_path: str = "/Volumes/workspace/demandiq_landing/raw/instacart"
    table_format: str = "delta"
    run_id: str = "manual"

    def schema(self, layer: str) -> str:
        if layer not in LAYERS:
            raise ValueError(f"unknown layer {layer!r}, expected one of {LAYERS}")
        return f"{self.schema_prefix}{layer}"

    def schema_fqn(self, layer: str) -> str:
        return f"{_quote(self.catalog)}.{_quote(self.schema(layer))}"

    def table(self, layer: str, name: str) -> str:
        return f"{self.schema_fqn(layer)}.{_quote(name)}"

    def bronze(self, name: str) -> str:
        return self.table("bronze", name)

    def silver(self, name: str) -> str:
        return self.table("silver", name)

    def gold(self, name: str) -> str:
        return self.table("gold", name)

    @property
    def is_delta(self) -> bool:
        return self.table_format.lower() == "delta"
