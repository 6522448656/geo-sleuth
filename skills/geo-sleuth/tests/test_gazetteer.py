#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Offline gazetteer regressions: bilingual OSM names. Run: uv run skills/geo-sleuth/tests/test_gazetteer.py"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import gazetteer  # noqa: E402

BOUNDS = {"minlat": 1.0, "minlon": 2.0, "maxlat": 3.0, "maxlon": 4.0}


def rel(osm_id: int, level: int, **tags) -> dict:
    tags = {k.replace("_", ":").replace(":Hans", "-Hans"): v for k, v in tags.items()}
    return {"type": "relation", "id": osm_id, "bounds": BOUNDS, "tags": {"admin_level": str(level), **tags}}


# Tags as OSM has them (2026-10): the ethnic autonomous regions and Hong Kong carry a second script in `name`
INNER_MONGOLIA = rel(161349, 4, name="内蒙古自治区 ᠦᠪᠦᠷ ᠮᠣᠩᠭᠤᠯ ᠤᠨ ᠥᠪᠡᠷᠲᠡᠭᠡᠨ ᠵᠠᠰᠠᠬᠣ ᠣᠷᠣᠨ", name_zh="内蒙古自治区", name_en="Inner Mongolia")
TIBET = rel(153292, 4, name="西藏自治区 བོད་རང་སྐྱོང་ལྗོངས།", name_zh="西藏自治区", name_en="Xizang")
HONG_KONG_4 = rel(20044132, 4, name="香港 Hong Kong", name_zh="香港", name_en="Hong Kong")
HONG_KONG_3 = rel(913110, 3, name="香港 Hong Kong", name_zh="香港", name_en="Hong Kong")
MACAU = rel(1867188, 3, name="澳門 Macau", name_zh="澳門", name_zh_Hans="澳门", name_en="Macau")
TAIWAN = rel(449220, 2, name="臺灣", name_zh_Hans="台湾", name_en="Taiwan")
SICHUAN = rel(913068, 4, name="四川省", name_zh="四川省", name_en="Sichuan")
RIO = rel(57963, 4, **{"name": "Rio de Janeiro", "name:pt": "Rio de Janeiro", "name:en": "Rio de Janeiro"})


class BilingualNames(unittest.TestCase):
    def test_rows_use_the_plain_name_and_keep_every_variant(self):
        rows = {r["osm_id"]: r for r in gazetteer._rel_rows({"elements": [INNER_MONGOLIA, TIBET, HONG_KONG_4, MACAU, SICHUAN, RIO]})}
        self.assertEqual(rows[161349]["name"], "内蒙古自治区")
        self.assertEqual(rows[153292]["name"], "西藏自治区")
        self.assertEqual(rows[20044132]["name"], "香港")
        self.assertEqual(rows[1867188]["name"], "澳門")
        self.assertIn("澳门", rows[1867188]["names"])
        self.assertEqual(rows[913068]["name"], "四川省")
        self.assertEqual(rows[57963]["name"], "Rio de Janeiro")     # a name with spaces that isn't bilingual stays whole

    def test_duplicate_elements_from_a_union_query_are_listed_once(self):
        self.assertEqual(len(gazetteer._rel_rows({"elements": [TIBET, TIBET]})), 1)

    def test_name_lookup_matches_every_name_key_and_escapes_quotes(self):
        sel = gazetteer._by_name('A "B"', '["admin_level"="4"]', "(area.p)")
        for key in gazetteer.NAME_KEYS:
            self.assertIn(f'rel(area.p)["{key}"="A \\"B\\""]["boundary"="administrative"]["admin_level"="4"];', sel)


if __name__ == "__main__":
    unittest.main()
