"""The dashboard's SQL must run against the gold tables the job produces,
and every widget must point at a dataset column that exists."""

from __future__ import annotations

import json
import os

import pytest

DASHBOARD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "dashboards", "demandiq_lakehouse.lvdash.json")


@pytest.fixture(scope="module")
def dashboard():
    with open(DASHBOARD) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def dataset_columns(spark, built, dashboard):
    cfg = built["cfg"]
    spark.sql(f"USE {cfg.schema_fqn('gold')}")  # the bundle sets dataset_schema the same way
    columns = {}
    try:
        for ds in dashboard["datasets"]:
            df = spark.sql("".join(ds["queryLines"]))
            if ds["name"] != "quarantine":  # empty on clean data, which is the good case
                assert df.count() > 0, f"dataset {ds['name']} returned no rows"
            columns[ds["name"]] = set(df.columns)
    finally:
        spark.sql("USE spark_catalog.default")
    return columns


def _widgets(dashboard):
    for page in dashboard["pages"]:
        for item in page["layout"]:
            yield page["name"], item["widget"], item["position"]


def test_every_dataset_query_runs(dataset_columns, dashboard):
    assert set(dataset_columns) == {d["name"] for d in dashboard["datasets"]}


def test_widgets_reference_existing_columns(dashboard, dataset_columns):
    for page, widget, _ in _widgets(dashboard):
        for query in widget.get("queries", []):
            ds = query["query"]["datasetName"]
            assert ds in dataset_columns, f"{page}/{widget['name']}: unknown dataset {ds}"
            for field in query["query"]["fields"]:
                if "associative_filter_predicate_group" in field["expression"]:
                    continue
                column = field["expression"].strip("`")
                assert column in dataset_columns[ds], f"{page}/{widget['name']}: {column} not in {ds}"
        encodings = widget.get("spec", {}).get("encodings", {})
        query_fields = {f["name"] for q in widget.get("queries", []) for f in q["query"]["fields"]}
        for enc in (encodings.get("columns", []) + encodings.get("fields", [])
                    + [v for k, v in encodings.items() if isinstance(v, dict)]):
            if "fieldName" in enc:
                assert enc["fieldName"] in query_fields, f"{page}/{widget['name']}: {enc['fieldName']}"


def test_layout_fits_the_grid_without_overlap(dashboard):
    names = [w["name"] for _, w, _ in _widgets(dashboard)]
    assert len(names) == len(set(names)), "widget names must be unique"
    for page in dashboard["pages"]:
        occupied = set()
        for item in page["layout"]:
            p = item["position"]
            assert 0 <= p["x"] and p["x"] + p["width"] <= 6, f"{item['widget']['name']} is off the grid"
            cells = {(x, y) for x in range(p["x"], p["x"] + p["width"])
                     for y in range(p["y"], p["y"] + p["height"])}
            assert not occupied & cells, f"{item['widget']['name']} overlaps another widget"
            occupied |= cells
