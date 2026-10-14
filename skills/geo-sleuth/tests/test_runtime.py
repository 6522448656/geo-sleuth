#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow", "pillow-heif", "numpy", "pyyaml"]
# ///
"""Offline runtime regressions. Run: uv run skills/geo-sleuth/tests/test_runtime.py"""
from __future__ import annotations

import asyncio
import contextlib
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import http.server
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from PIL import Image

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _net
import _browser
import _device
import _image
import match
import baidu_pano
import doctor
import evidence
import gsv
import osm
import poi
import revimg


class NetworkTests(unittest.TestCase):
    def test_route_precedence(self):
        with patch.dict(os.environ, {"GEO_PROXY": "http://localhost:8123"}, clear=True):
            self.assertEqual(_net.resolve_proxy(), "http://localhost:8123")
            self.assertEqual(_net.resolve_proxy("http://localhost:8234"), "http://localhost:8234")
            self.assertIsNone(_net.resolve_proxy("direct"))
            self.assertIsNone(_net.resolve_proxy(""))
        with patch.dict(os.environ, {"HTTPS_PROXY": "http://localhost:9999"}, clear=True):
            self.assertIsNone(_net.resolve_proxy())

    @unittest.skipUnless(shutil.which("curl"), "curl required")
    def test_curl_reaches_direct_and_explicit_proxy_despite_ambient_settings(self):
        paths = []
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                paths.append(self.path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            def log_message(self, *args):
                pass
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:1", "http_proxy": "http://127.0.0.1:1",
                                         "ALL_PROXY": "http://127.0.0.1:1", "NO_PROXY": "*", "GEO_PROXY": "http://127.0.0.1:1"}):
                self.assertEqual(_net.fetch_bytes(base + "/direct", "direct"), b"ok")
                self.assertEqual(_net.fetch_bytes("http://example.invalid/proxied", base), b"ok")
            self.assertEqual(paths, ["/direct", "http://example.invalid/proxied"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_model_route_overrides_ambient_variables(self):
        with patch.dict(os.environ, {"GEO_PROXY": "http://chosen:8080", "https_proxy": "http://old:8080", "NO_PROXY": "*"}, clear=True):
            _net.model_proxy_env()
            self.assertEqual(os.environ["HTTPS_PROXY"], "http://chosen:8080")
            if os.name != "nt":   # Windows environment variable names are case-insensitive
                self.assertNotIn("https_proxy", os.environ)
            self.assertNotIn("NO_PROXY", os.environ)
            _net.model_proxy_env("direct")
            self.assertNotIn("HTTPS_PROXY", os.environ)
            self.assertEqual(os.environ["NO_PROXY"], "*")

    def test_baidu_near_forwards_proxy(self):
        with patch.object(baidu_pano, "fetch_bytes", return_value=b'{"content":null}') as fetch:
            self.assertIsNone(baidu_pano.near(35, 110, proxy="http://chosen:8080"))
            self.assertEqual(fetch.call_args.args[1], "http://chosen:8080")

    def test_all_poi_sources_forward_proxy(self):
        with patch.object(poi, "_curl", return_value='{}') as fetch:
            poi.search_so("school", None, 2, "direct")
            self.assertEqual(fetch.call_args.kwargs["proxy"], "direct")
            poi.search_sug("school", "http://chosen:8080")
            self.assertEqual(fetch.call_args.kwargs["proxy"], "http://chosen:8080")
        with patch.object(poi, "_curl", return_value='[]') as fetch:
            poi.search_osm("school", None, 2, "http://chosen:8080", "us")
            self.assertEqual(fetch.call_args.kwargs["proxy"], "http://chosen:8080")

    def test_intake_forwards_route_to_both_engines(self):
        import intake
        from PIL import Image
        commands = []
        def run(cmd, **kwargs):
            commands.append(cmd)
            return 0, "", ""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            photo = root / "photo.jpg"
            Image.new("RGB", (100, 100), "white").save(photo)
            # Preparation writes files used by intake; run the actual small local helpers.
            original = intake._run
            def local_or_search(cmd, **kwargs):
                if any(str(part).endswith("revimg.py") for part in cmd):
                    return run(cmd, **kwargs)
                return original([sys.executable, *cmd[2:]], **kwargs)
            with patch.object(intake, "_run", side_effect=local_or_search), patch.object(sys, "argv", ["intake.py", str(photo), "--out-dir", str(root / "out"), "--no-ocr", "--max-variants", "1", "--proxy", "direct"]):
                intake.main()
            self.assertEqual(len(commands), 2)
            for cmd in commands:
                self.assertEqual(cmd[cmd.index("--proxy") + 1], "direct")
            self.assertEqual({cmd[cmd.index("--engines") + 1] for cmd in commands}, {"baidu", "yandex"})


class BrowserTests(unittest.IsolatedAsyncioTestCase):
    async def test_chromium_fallback_uses_same_route(self):
        browser = object()
        launch = AsyncMock(side_effect=[RuntimeError("Chrome absent"), browser])
        p = SimpleNamespace(chromium=SimpleNamespace(launch=launch))
        actual, label = await _browser.launch_browser(p, "socks5h://localhost:8123")
        self.assertIs(actual, browser)
        self.assertEqual(label, "Playwright Chromium")
        for call in launch.call_args_list:
            self.assertEqual(call.kwargs["proxy"]["server"], "socks5://localhost:8123")
        self.assertEqual(launch.call_args_list[0].kwargs["channel"], "chrome")
        self.assertNotIn("channel", launch.call_args_list[1].kwargs)

    async def test_explicit_direct_overrides_saved_browser_proxy(self):
        launch = AsyncMock(return_value=object())
        p = SimpleNamespace(chromium=SimpleNamespace(launch=launch))
        with patch.dict(os.environ, {"GEO_PROXY": "http://old:8080"}):
            await _browser.launch_browser(p, "direct")
        self.assertNotIn("proxy", launch.call_args.kwargs)
        self.assertIn("--no-proxy-server", launch.call_args.kwargs["args"])

    async def test_both_browsers_missing_gives_install_fix(self):
        p = SimpleNamespace(chromium=SimpleNamespace(launch=AsyncMock(side_effect=RuntimeError("missing"))))
        with self.assertRaisesRegex(RuntimeError, "uvx playwright install chromium"):
            await _browser.launch_browser(p, "direct")

    async def test_reverse_search_all_engines_get_selected_route(self):
        browser = SimpleNamespace(new_context=AsyncMock(), close=AsyncMock())
        manager = AsyncMock()
        manager.__aenter__.return_value = object()
        # Avoid requiring Playwright in this offline test's environment.
        fake = SimpleNamespace(async_playwright=lambda: manager)
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, {"playwright.async_api": fake}), patch.object(revimg, "launch_browser", AsyncMock(return_value=(browser, "test"))) as launch, patch.object(revimg, "_text", AsyncMock(return_value={"links": []})), patch.object(revimg, "_baidu", AsyncMock(return_value={"links": []})) as baidu, patch.object(revimg, "_yandex", AsyncMock(return_value={"links": []})):
            await revimg.run([Path("photo.jpg")], ["baidu", "yandex"], Path(folder), "direct", ["school"], ["bing"])
            self.assertEqual(launch.await_count, 3)
            self.assertTrue(all(call.args[1] == "direct" for call in launch.await_args_list))
            self.assertEqual(baidu.await_args.args[-1], "direct")


class DoctorTests(unittest.TestCase):
    def test_probe_distinguishes_transport_failure_and_service_block(self):
        for returncode, stdout, status in [(0, "200", "PASS"), (0, "429", "WARN"), (7, "000", "FAIL")]:
            with self.subTest(status=status), patch.object(doctor.subprocess, "run", return_value=SimpleNamespace(returncode=returncode, stdout=stdout)):
                row = doctor.probe(("service", "https://example.invalid"), "direct")
                self.assertEqual(row["status"], status)

    def test_local_checks_do_not_probe_network_or_print_proxy_credentials(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(doctor, "browser_check", AsyncMock(return_value=doctor.check("browser", "PASS", "ready"))), patch.object(doctor, "probe") as probe, patch.dict(os.environ, {"GEO_PROXY": "http://user:secret@localhost:8080"}):
            report = doctor.diagnose(False, None)
            probe.assert_not_called()
            self.assertTrue(report["ok"])
            self.assertNotIn("secret", json.dumps(report))
            self.assertEqual(report["connection"], "configured proxy")


QH = '[out:json][timeout:180];(area["name"="青海省"];area["name:zh"="青海省"];area["name:zh-Hans"="青海省"];)->.searchArea;nwr["power"="line"](area.searchArea);out geom tags;'
LINE = {"type": "way", "id": 1, "geometry": [{"lat": 37.7, "lon": 95.3}, {"lat": 37.8, "lon": 95.4}], "tags": {"power": "line"}}


def count(n):
    return {"elements": [{"type": "count", "tags": {"total": str(n)}}]}


class OverpassMirrorTests(unittest.TestCase):
    """One mirror answered a whole-province --area query with a clean empty result while others had thousands of lines; it got cached."""

    def run_with(self, replies, ql=QH):
        calls = []

        def post(ep, q, proxy, timeout):
            calls.append(ep)
            return replies(ep, q)

        with tempfile.TemporaryDirectory() as d, patch.object(osm, "_post", side_effect=post), patch.object(osm.time, "sleep"):
            data = osm.run(ql, None, Path(d))
            cached = list(Path(d).glob("*.json"))
        return data, calls, cached

    def test_empty_from_one_mirror_is_checked_on_another(self):
        data, calls, cached = self.run_with(lambda ep, q: ({"elements": []} if ep == osm.ENDPOINTS[0] else {"elements": [LINE]}, ""))
        self.assertEqual(len(data["elements"]), 1)
        self.assertEqual(calls, osm.ENDPOINTS[:2])
        self.assertEqual(len(cached), 1)

    def test_empty_confirmed_by_two_mirrors_is_cached(self):
        data, calls, cached = self.run_with(lambda ep, q: ({"elements": []}, ""))
        self.assertEqual(data["elements"], [])
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(cached), 1)

    def test_unconfirmed_empty_is_returned_but_not_cached(self):
        data, calls, cached = self.run_with(lambda ep, q: ({"elements": []}, "") if ep == osm.ENDPOINTS[0] else (None, "503"))
        self.assertEqual(data["elements"], [])
        self.assertEqual(cached, [])
        self.assertEqual(len(calls), len(osm.ENDPOINTS))           # one round, no retry sleeps

    def test_zero_area_count_needs_a_second_mirror(self):
        ql = '[out:json][timeout:120];(area["name"="青海省"];)->.a;.a out count;'
        data, _, _ = self.run_with(lambda ep, q: (count(0) if ep == osm.ENDPOINTS[0] else count(1), ""), ql)
        self.assertEqual(data["elements"][0]["tags"]["total"], "1")

    def test_bbox_empty_is_trusted(self):
        ql = '[out:json][timeout:180];nwr["power"="line"](37.6,95.2,37.8,95.4);out geom tags;'
        _, calls, cached = self.run_with(lambda ep, q: ({"elements": []}, ""), ql)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(cached), 1)

    def test_failure_names_every_mirror(self):
        replies = {osm.ENDPOINTS[0]: (None, "HTTP 406: Not Acceptable"), osm.ENDPOINTS[1]: (None, "HTTP 500: Internal Server Error"),
                   osm.ENDPOINTS[2]: (None, "HTTP 504: server too busy")}
        with self.assertRaises(SystemExit) as cm:
            self.run_with(lambda ep, q: replies[ep])
        for code in ("406", "500", "504"):
            self.assertIn(code, str(cm.exception))

    def test_error_remark_not_cached(self):
        def replies(ep, q):
            if ep == osm.ENDPOINTS[0]:
                return {"elements": [], "remark": "runtime error: Query timed out"}, ""
            return {"elements": [LINE]}, ""
        data, _, cached = self.run_with(replies)
        self.assertEqual(len(data["elements"]), 1)
        self.assertEqual(len(cached), 1)


class StreetViewTests(unittest.TestCase):
    def test_lookup_replies(self):
        ok = json.dumps([[0], [[1], [2, "Z4kP7jsHKunVOqfsg_khyw"], None, [None, None, [["168 W 60th Ave"], ["Vancouver"]]], None,
                               [[None, [[None, None, 49.21598, -123.10927], None, [90.4]], None, [[]], None, None, None, None, []]],
                               [None] * 7 + [[2024, 5]]]])
        cases = {
            ok: "Z4kP7jsHKunVOqfsg_khyw",
            '[[5,"generic","Search returned no images."]]': None,
        }
        for reply, want in cases.items():
            with patch.object(gsv, "_curl", return_value=reply.encode()):
                res = gsv.near(49.2161, -123.1093, 50, None)
                self.assertEqual(res and res["id"], want)
        ok_res = None
        with patch.object(gsv, "_curl", return_value=ok.encode()):
            ok_res = gsv.near(49.2161, -123.1093, 50, None)
        self.assertEqual(ok_res["date"], "2024-05")
        for bad in ('[[5,"generic","GeoPhotoService.SingleImageSearch is decommissioned and turned down."]]',
                    '[3,"Invalid JSON payload received."]', "<html>blocked</html>", ""):
            with patch.object(gsv, "_curl", return_value=bad.encode()), self.assertRaises(gsv.ServiceError):
                gsv.near(49.2161, -123.1093, 50, None)



def _coverage(panos: list[tuple[str, float, float]]) -> str:
    """A coverage tile reply in the layout photometa/ac/v1 returns; no panoramas gives the empty-tile layout."""
    if not panos:
        return ")]}'\n" + json.dumps([None, None, [["ChA0YTkw", [None, None, [None, None, 40.0, -40.0]]]]])
    rows = [[[[2, pid], None, [[None, None, la, lo], [19.2, None, 63.4], [45.9, 0.27, 1.5]]], [27, 113]] for pid, la, lo in panos]
    return ")]}'\n" + json.dumps([None, [None, rows]])


class CoverageTests(unittest.TestCase):
    PID = "77-Z-M-G0QlSNisjusa5DA"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def world(self, panos):
        """_curl stand-in serving each z17 tile only the panoramas inside it; records requested tiles."""
        asked = []

        def curl(url, proxy, out=None, data=None):
            x, y = map(int, re.search(r"!1i(\d+)!2i(\d+)", url).groups())
            asked.append((x, y))
            inside = [(pid, la, lo) for pid, la, lo in panos
                      if tuple(int(v // 256) for v in gsv.geo.ll2px(17, la, lo)) == (x, y)]
            return _coverage(inside).encode()
        return curl, asked

    def test_empty_tile_vs_failed_lookup(self):
        with patch.object(gsv, "_curl", return_value=_coverage([]).encode()):
            self.assertEqual(gsv.tile_panos(1, 1, None, self.cache), [])
        with patch.object(gsv, "_curl", return_value=_coverage([(self.PID, 51.44, 5.47)]).encode()):
            self.assertEqual(gsv.tile_panos(2, 2, None, self.cache), [(self.PID, 51.44, 5.47)])
        for bad in ("<html>429 Too Many Requests</html>", "", ")]}'\n<html>429 Too Many Requests</html>",
                    ")]}'\n[3,\"Invalid request\"]", ")]}'\n{\"error\": 1}"):
            with patch.object(gsv, "_curl", return_value=bad.encode()), self.assertRaises(gsv.ServiceError):
                gsv.tile_panos(3, 3, None, self.cache)

    def test_failed_reply_is_not_cached(self):
        bad = ")]}'\n<html>429 Too Many Requests</html>"
        with patch.object(gsv, "_curl", return_value=bad.encode()), self.assertRaises(gsv.ServiceError):
            gsv.tile_panos(5, 5, None, self.cache)
        self.assertEqual([f.name for f in self.cache.iterdir()], [])
        good = _coverage([(self.PID, 51.44, 5.47)]).encode()
        with patch.object(gsv, "_curl", return_value=good) as m:
            self.assertEqual(len(gsv.tile_panos(5, 5, None, self.cache)), 1)
            self.assertEqual(m.call_count, 1)  # retried, not served from cache
            self.assertEqual(len(gsv.tile_panos(5, 5, None, self.cache)), 1)
            self.assertEqual(m.call_count, 1)  # a valid reply is cached
        # an error reply left in the cache by an older version is fetched again
        (self.cache / "cov17_6_6.txt").write_text(bad, encoding="utf-8")
        with patch.object(gsv, "_curl", return_value=good) as m:
            self.assertEqual(len(gsv.tile_panos(6, 6, None, self.cache)), 1)
            self.assertEqual(m.call_count, 1)

    def test_disappearing_coverage_cache_is_refetched(self):
        cached = self.cache / "cov17_7_7.txt"
        cached.write_text(")]}'\n<html>429 Too Many Requests</html>", encoding="utf-8")
        read_text = Path.read_text

        def read(path, *args, **kwargs):
            if path == cached:
                path.unlink()  # another worker removed a stale entry before this read
            return read_text(path, *args, **kwargs)

        good = _coverage([(self.PID, 51.44, 5.47)]).encode()
        with patch.object(Path, "read_text", read), patch.object(gsv, "_curl", return_value=good) as fetch:
            self.assertEqual(gsv.tile_panos(7, 7, None, self.cache), [(self.PID, 51.44, 5.47)])
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(gsv._coverage_rows(cached.read_text(encoding="utf-8")), gsv._coverage_rows(good.decode()))

    def test_shared_coverage_tile_uses_one_fetch(self):
        for stale in (False, True):
            with self.subTest(stale=stale):
                cached = self.cache / "cov17_8_8.txt"
                if stale:
                    cached.write_text(")]}'\n<html>429 Too Many Requests</html>", encoding="utf-8")
                fetching, release, started = threading.Event(), threading.Event(), threading.Event()
                good = _coverage([(self.PID, 51.44, 5.47)]).encode()

                def curl(*args, **kwargs):
                    fetching.set()
                    if not release.wait(5):
                        raise RuntimeError("test did not release coverage request")
                    return good

                def second():
                    started.set()
                    return gsv.tile_panos(8, 8, None, self.cache)

                with patch.object(gsv, "_curl", side_effect=curl) as fetch, ThreadPoolExecutor(2) as ex:
                    first = ex.submit(gsv.tile_panos, 8, 8, None, self.cache)
                    try:
                        self.assertTrue(fetching.wait(5))
                        other = ex.submit(second)
                        self.assertTrue(started.wait(5))
                        with self.assertRaises(TimeoutError):
                            other.result(timeout=0.1)
                    finally:
                        release.set()
                    expected = [(self.PID, 51.44, 5.47)]
                    self.assertEqual(first.result(timeout=5), expected)
                    self.assertEqual(other.result(timeout=5), expected)
                    self.assertEqual(fetch.call_count, 1)
                cached.unlink()

    def test_targets_reports_failed_lookups(self):
        pts = self.cache / "pts.json"
        pts.write_text(json.dumps({"church": [51.44, 5.47]}), encoding="utf-8")
        out = self.cache / "t.index.json"
        argv = ["gsv.py", "targets", "--points", str(pts), "--out", str(out), "--cache", str(self.cache / "c")]
        with patch.object(gsv, "_curl", return_value=b"<html>429 Too Many Requests</html>"), \
                patch.object(sys, "argv", argv), patch("sys.stdout", new=io.StringIO()) as stdout, \
                patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit) as e:
            gsv.main()
        self.assertIn("lookup failed", str(e.exception.code))
        self.assertIn("0 targets without a panorama", stdout.getvalue())
        self.assertIn("1 lookups failed", stdout.getvalue())

    def _edge_target(self, lat):
        """A target 1 px inside the eastern edge of its z17 tile."""
        x, y = gsv.geo.ll2px(17, lat, 5.0)
        tx = int(x // 256)
        tlat, tlon = gsv.geo.px2ll(17, (tx + 1) * 256 - 1, y)
        return tx, tlat, tlon

    def _east(self, lat, lon, metres):
        return lon + math.degrees(metres / (gsv.geo._R * math.cos(math.radians(lat))))

    def test_radius_wider_than_one_tile(self):
        tx, lat, lon = self._edge_target(51.5)
        plon = self._east(lat, lon, 250)
        self.assertEqual(int(gsv.geo.ll2px(17, lat, plon)[0] // 256), tx + 2)  # outside the old 3x3 block
        curl, asked = self.world([(self.PID, lat, plon)])
        with patch.object(gsv, "_curl", side_effect=curl):
            got = gsv.panos_around(lat, lon, 12, 300, 2, 15, None, self.cache)
        self.assertEqual([g[0] for g in got], [self.PID])
        self.assertAlmostEqual(got[0][3], 250, delta=1)
        self.assertIn(tx + 2, {a[0] for a in asked})
        with patch.object(gsv, "_curl", side_effect=curl):  # the exact distance filter still applies
            self.assertEqual(gsv.panos_around(lat, lon, 12, 200, 2, 15, None, self.cache), [])

    def test_high_latitude_default_radius(self):
        tx, lat, lon = self._edge_target(78.2)
        plon = self._east(lat, lon, 69)
        self.assertEqual(int(gsv.geo.ll2px(17, lat, plon)[0] // 256), tx + 2)
        curl, _ = self.world([(self.PID, lat, plon)])
        with patch.object(gsv, "_curl", side_effect=curl):
            got = gsv.panos_around(lat, lon, 12, 70, 2, 15, None, self.cache)
        self.assertEqual([g[0] for g in got], [self.PID])

    def test_tile_range_stays_small(self):
        self.assertLessEqual(len(gsv.tiles_around(51.5, 5.0, 70)), 4)  # a 70 m radius spans at most 2x2 tiles of ~190 m
        tiles = gsv.tiles_around(0.0, 179.9999, 300)  # wraps across the antimeridian
        self.assertTrue(all(0 <= x < 2 ** 17 for x, _ in tiles))
        self.assertIn(0, {x for x, _ in tiles})

class FontTests(unittest.TestCase):
    def _tried_paths(self, env):
        tried = []

        def fake_truetype(path, size):
            tried.append(path)
            raise OSError("not found")

        with patch.dict(os.environ, env, clear=True),                 patch.object(baidu_pano.ImageFont, "truetype", side_effect=fake_truetype),                 patch.object(baidu_pano.ImageFont, "load_default", return_value="default"):
            self.assertEqual(baidu_pano._font(16), "default")
        return tried

    def test_windows_cjk_fonts_are_tried(self):
        tried = self._tried_paths({})
        self.assertIn("C:/Windows/Fonts/msyh.ttc", tried)
        self.assertIn("C:/Windows/Fonts/msyhbd.ttc", tried)
        self.assertIn("C:/Windows/Fonts/simsun.ttc", tried)

    def test_geo_font_is_tried_first(self):
        tried = self._tried_paths({"GEO_FONT": "D:/fonts/custom.ttf"})
        self.assertEqual(tried[0], "D:/fonts/custom.ttf")

    def test_first_available_font_is_used(self):
        def fake_truetype(path, size):
            if path != "C:/Windows/Fonts/simhei.ttf":
                raise OSError("not found")
            return path

        with patch.dict(os.environ, {}, clear=True),                 patch.object(baidu_pano.ImageFont, "truetype", side_effect=fake_truetype):
            self.assertEqual(baidu_pano._font(16), "C:/Windows/Fonts/simhei.ttf")


class SkillMetadataTests(unittest.TestCase):
    def test_frontmatter_meets_agent_skills_spec(self):
        root = Path(__file__).resolve().parents[1]
        m = re.match(r"---\n(.*?)\n---\n", (root / "SKILL.md").read_text(encoding="utf-8"), re.S)
        self.assertIsNotNone(m, "frontmatter must open and close with --- lines")
        meta = yaml.safe_load(m.group(1))                  # strict parser: a ': ' inside the description breaks it
        self.assertIsInstance(meta, dict)
        self.assertEqual(meta["name"], root.name)
        self.assertIsInstance(meta["description"], str)
        self.assertTrue(meta["description"].strip())
        self.assertLessEqual(len(meta["description"]), 1024)   # https://agentskills.io/specification


class EvidenceSpecTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        (self.base / "area.jpg").write_bytes(b"jpg")
        (self.base / "area.json").write_text("{}", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_map_is_named_instead_of_key_error(self):
        with self.assertRaises(SystemExit) as e:
            evidence.build({"camera": [1, 2]}, self.base, self.base / "out.jpg")
        self.assertIn("missing 'map'", str(e.exception.code))

    def test_map_without_georeference(self):
        (self.base / "area.json").unlink()
        self.assertIn("no georeference", evidence.check_spec({"map": "area.jpg"}, self.base)[0])

    def test_item_keys_and_panel_images(self):
        errs = evidence.check_spec({"map": "area.jpg", "labels": [{"at": [1, 2]}], "lines": [{"from": [1, 2], "bearing": 3}],
                                    "panels": [{"image": "sv1.jpg"}]}, self.base)
        self.assertEqual(len(errs), 3)
        self.assertIn("labels[0] is missing 'text'", errs[0])
        self.assertIn("lines[0] is missing 'length_m'", errs[1])
        self.assertIn("panels[0] image not found", errs[2])

    def test_valid_spec(self):
        self.assertEqual(evidence.check_spec({"map": "area.jpg", "labels": [{"at": [1, 2], "text": "x"}]}, self.base), [])

    def test_invalid_path_and_item_types_are_named(self):
        cases = [
            ([], "JSON object"),
            *[({"map": value}, "'map'") for value in (None, 7, "")],
            *[({"map": "area.jpg", key: value}, repr(key))
              for key, value in (("labels", None), ("labels", {}), ("lines", "bad"), ("panels", None))],
            *[({"map": "area.jpg", key: [value]}, f"{key}[0]")
              for key, value in (("labels", None), ("labels", "at text"), ("lines", 3))],
            ({"map": "area.jpg", "panels": [{"image": None}]}, "panels[0]"),
        ]
        for spec, expected in cases:
            with self.subTest(spec=spec):
                self.assertTrue(any(expected in error for error in evidence.check_spec(spec, self.base)))

    def test_missing_files_are_reported_together(self):
        errors = evidence.check_spec({"map": "missing.jpg", "panels": [{"image": "missing-panel.jpg"}]}, self.base)
        self.assertEqual(len(errors), 3)
        self.assertIn("map not found", errors[0])
        self.assertIn("no georeference", errors[1])
        self.assertIn("panels[0] image not found", errors[2])

    def test_invalid_spec_cli_exits_without_traceback(self):
        spec_path = self.base / "bad.json"
        spec_path.write_text(json.dumps({"map": None, "labels": None, "panels": [{"image": None}]}), encoding="utf-8")
        out = self.base / "out.jpg"
        result = subprocess.run([sys.executable, str(Path(evidence.__file__)), str(spec_path), "--out", str(out)],
                                capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 1)
        for expected in ("'map'", "'labels'", "panels[0]", "evidence.py --help"):
            self.assertIn(expected, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(out.exists())


class DeviceTests(unittest.TestCase):
    @staticmethod
    def _torch(cuda=False, mps=False):
        return SimpleNamespace(__version__="2.0.0+test",
                               cuda=SimpleNamespace(is_available=lambda: cuda),
                               backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)))

    def _pick(self, value=None, **have):
        env = {} if value is None else {"GEO_DEVICE": value}
        with patch.dict(os.environ, env, clear=True):
            return _device.torch_device(self._torch(**have))

    def _exit_message(self, value, **have):
        with self.assertRaises(SystemExit) as caught:
            self._pick(value, **have)
        message = str(caught.exception.code)
        self.assertEqual(len(message.splitlines()), 1)
        self.assertIn("torch 2.0.0+test", message)
        return message

    def test_auto_selection_order(self):
        self.assertEqual(self._pick(cuda=True, mps=True), "cuda")
        self.assertEqual(self._pick(cuda=True), "cuda")
        self.assertEqual(self._pick(mps=True), "mps")
        self.assertEqual(self._pick(), "cpu")
        self.assertEqual(self._pick("", cuda=True), "cuda")

    def test_explicit_override(self):
        self.assertEqual(self._pick("cpu", cuda=True, mps=True), "cpu")
        self.assertEqual(self._pick("cuda", cuda=True, mps=True), "cuda")
        self.assertEqual(self._pick("mps", cuda=True, mps=True), "mps")
        self.assertEqual(self._pick(" CUDA ", cuda=True), "cuda")

    def test_invalid_value_exits_with_one_line(self):
        for value in ("gpu", "cuda:0", "auto"):
            message = self._exit_message(value, cuda=True)
            self.assertIn(repr(value), message)
            self.assertIn("cpu, cuda, mps", message)

    def test_unavailable_device_exits_instead_of_falling_back(self):
        message = self._exit_message("cuda", mps=True)
        self.assertIn("GEO_DEVICE=cuda", message)
        self.assertIn("cuda=False, mps=True", message)
        message = self._exit_message("mps", cuda=True)
        self.assertIn("GEO_DEVICE=mps", message)
        self.assertIn("cuda=True, mps=False", message)

    def test_both_model_scripts_use_the_shared_helper(self):
        scripts = Path(_device.__file__).parent
        for name in ("match.py", "sat_scan.py"):
            text = (scripts / name).read_text(encoding="utf-8")
            self.assertIn("self.dev = torch_device(torch)", text)
            self.assertNotIn("mps.is_available", text)


HEIC_HEAD = b"\x00\x00\x00\x1cftypheic" + b"\x00" * 64


class ImageLoadingTests(unittest.TestCase):
    """scripts/_image.py: explicit formats, content sniffing, RAW rejection, failure categories, folder discovery."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def _save(self, name, fmt="JPEG", color=(200, 30, 30), size=(32, 24)):
        Image.new("RGB", size, color).save(self.d / name, format=fmt)
        return self.d / name

    def _category(self, path):
        with self.assertRaises(_image.ImageReadError) as caught:
            _image.open_image(path)
        self.assertEqual(len(caught.exception.reason.splitlines()), 1)
        return caught.exception.category, caught.exception.reason

    def test_sniff_reads_content_not_suffix(self):
        heads = {
            "JPEG": b"\xff\xd8\xff\xe0", "PNG": b"\x89PNG\r\n\x1a\n", "WEBP": b"RIFF\x00\x00\x00\x00WEBPVP8 ",
            "GIF": b"GIF89a", "HEIF": HEIC_HEAD[:16], "AVIF": b"\x00\x00\x00\x1cftypavif", "TIFF": b"II*\x00\x08\x00\x00\x00",
            "TEXT": b'{"status":400}',
        }
        for kind, head in heads.items():
            self.assertEqual(_image.sniff(head, ".jpg"), kind)
        self.assertEqual(_image.sniff(b"II*\x00\x10\x00\x00\x00CR\x02\x00", ".jpg"), "RAW")      # Canon CR2
        self.assertEqual(_image.sniff(b"\x00\x00\x00\x18ftypcrx ", ".cr3"), "RAW")               # Canon CR3
        self.assertEqual(_image.sniff(b"MM\x00*\x00\x00\x00\x08", ".NEF"), "RAW")               # TIFF-based RAW by suffix
        self.assertEqual(_image.sniff(b"FUJIFILMCCD-RAW 0201", ".raf"), "RAW")
        self.assertIsNone(_image.sniff(b"\x00\x01\x02\x03", ".jpg"))

    def test_supported_formats_open_whatever_the_suffix(self):
        for name, fmt in (("a.jpg", "JPEG"), ("b.png", "PNG"), ("c.webp", "WEBP"), ("d.jpg", "PNG"), ("e", "JPEG"), ("f.gif", "GIF")):
            with _image.open_image(self._save(name, fmt)) as im:
                self.assertEqual((im.format, im.size), (fmt, (32, 24)))

    @unittest.skipUnless(_image.HEIF, "pillow-heif not installed")
    def test_heic_decodes_and_keeps_metadata(self):
        ex = Image.Exif()
        ex[0x8769] = {0xA405: 26}                       # FocalLengthIn35mmFilm
        Image.new("RGB", (40, 30), (0, 0, 255)).save(self.d / "phone.heic", format="HEIF", exif=ex)
        im = _image.open_image(self.d / "phone.heic")
        self.assertEqual((im.format, im.size), ("HEIF", (40, 30)))
        self.assertGreater(im.getpixel((0, 0))[2], 200)
        self.assertEqual(im.getexif().get_ifd(0x8769).get(0xA405), 26)

    def test_failure_categories(self):
        (self.d / "empty.jpg").write_bytes(b"")
        (self.d / "error.jpg").write_bytes(b'{"status":400,"msg":"bad"}')
        (self.d / "noise.jpg").write_bytes(b"\x00\x01\x02\x03" * 20)
        (self.d / "IMG_1.CR2").write_bytes(b"II*\x00\x10\x00\x00\x00CR\x02\x00" + b"\x00" * 64)
        (self.d / "IMG_2.dng").write_bytes(b"II*\x00\x08\x00\x00\x00" + b"\x00" * 64)
        full = self._save("full.jpg").read_bytes()
        (self.d / "cut.jpg").write_bytes(full[: len(full) // 2])
        Image.new("RGB", (8, 8)).save(self.d / "frame.ppm", format="PPM")
        expect = {"empty.jpg": "empty", "error.jpg": "not-image", "noise.jpg": "not-image", "IMG_1.CR2": "raw",
                  "IMG_2.dng": "raw", "cut.jpg": "decode-error", "frame.ppm": "unsupported"}
        for name, category in expect.items():
            got, reason = self._category(self.d / name)
            self.assertEqual(got, category, f"{name}: {reason}")
        self.assertIn("export it as JPEG or PNG", self._category(self.d / "IMG_1.CR2")[1])

    def test_tiff_based_raw_is_rejected_by_content(self):
        """Structure of an iPhone ProRAW DNG: the first image is a full-size 8-bit preview marked reduced-resolution,
        the real 10-bit linear raw sits in a SubIFD. Pillow would read the preview, whatever the suffix."""
        from PIL import TiffImagePlugin
        for name, tags in (("dng_named.jpg", {254: 1, 50706: b"\x01\x06\x00\x00"}), ("preview_only", {254: 1, 330: 100})):
            info = TiffImagePlugin.ImageFileDirectory_v2()
            for k, v in tags.items():
                info[k] = v
            if 50706 in tags:
                info.tagtype[50706] = 1
            Image.new("RGB", (40, 30), (90, 120, 150)).save(self.d / name, format="TIFF", tiffinfo=info)
            category, reason = self._category(self.d / name)
            self.assertEqual(category, "raw", name)
            self.assertIn("embedded preview", reason)
        self._save("plain.tif", "TIFF")
        with _image.open_image(self.d / "plain.tif") as im:
            self.assertEqual(im.format, "TIFF")

    def test_heif_failure_keeps_its_cause(self):
        (self.d / "x.jpg").write_bytes(HEIC_HEAD)
        with patch.object(_image, "HEIF", False):
            self.assertEqual(self._category(self.d / "x.jpg")[0], "missing-decoder")
        if _image.HEIF:
            category, reason = self._category(self.d / "x.jpg")
            self.assertEqual(category, "decode-error")
            self.assertIn("HEIF that could not be decoded (", reason)     # the decoder's own message, not a guess

    def test_find_images_keeps_old_order_then_adds_new_formats(self):
        for name, fmt in (("b.jpg", "JPEG"), ("a.jpeg", "JPEG"), ("c.png", "PNG"), ("d.webp", "WEBP"),
                          ("e.HEIC", "PNG"), ("f.heif", "PNG"), ("g.bin", "PNG"), ("h", "JPEG")):
            self._save(name, fmt)
        (self.d / "notes.json").write_text("{}", encoding="utf-8")
        (self.d / "readme.txt").write_text("hello", encoding="utf-8")
        (self.d / "IMG_3.CR2").write_bytes(b"II*\x00\x10\x00\x00\x00CR\x02\x00")
        names = [f.name for f in _image.find_images(self.d)]
        old = sorted(["b.jpg", "a.jpeg"]) + ["c.png"]
        self.assertEqual(names[:3], old)
        self.assertEqual(sorted(names[3:]), sorted(["d.webp", "e.HEIC", "f.heif", "IMG_3.CR2", "g.bin", "h"]))
        self.assertNotIn("notes.json", names)
        self.assertNotIn("readme.txt", names)

    def test_open_or_exit_is_one_line(self):
        (self.d / "IMG_4.NEF").write_bytes(b"MM\x00*\x00\x00\x00\x08" + b"\x00" * 64)
        with self.assertRaises(SystemExit) as caught:
            _image.open_or_exit(self.d / "IMG_4.NEF", "photo")
        message = str(caught.exception.code)
        self.assertEqual(len(message.splitlines()), 1)
        self.assertIn("cannot read photo", message)
        self.assertIn("export it as JPEG or PNG", message)

    def test_photo_scripts_exit_cleanly_on_raw_and_read_heic(self):
        script = Path(_image.__file__).with_name("imgprep.py")
        raw = self.d / "IMG_5.ARW"
        raw.write_bytes(b"II*\x00\x08\x00\x00\x00" + b"\x00" * 64)
        r = subprocess.run([sys.executable, str(script), "zoom", str(raw), "--box", "0,0,4,4", "--out", str(self.d / "z.png")],
                           capture_output=True, text=True, encoding="utf-8")
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn("Traceback", r.stderr)
        self.assertIn("export it as JPEG or PNG", r.stderr)
        if _image.HEIF:
            Image.new("RGB", (40, 30), (0, 200, 0)).save(self.d / "p.heic", format="HEIF")
            r = subprocess.run([sys.executable, str(script), "zoom", str(self.d / "p.heic"), "--box", "0,0,20,20", "--out", str(self.d / "z.png")],
                               capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue((self.d / "z.png").exists())


class MatchImageLoadingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def _save(self, name, fmt="JPEG", color=(200, 30, 30)):
        Image.new("RGB", (32, 24), color).save(self.d / name, format=fmt)

    def _load(self, files):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            ok, ims = match._open_images(files)
        return ok, ims, err.getvalue()

    def test_unreadable_files_are_skipped_with_reasons(self):
        self._save("a.jpg")
        self._save("b.jpg", color=(30, 200, 30))
        (self.d / "heic.jpg").write_bytes(HEIC_HEAD)
        (self.d / "error.jpg").write_bytes('{"status":400,"msg":"bad"}'.encode())
        (self.d / "empty.jpg").write_bytes(b"")
        full = (self.d / "a.jpg").read_bytes()
        (self.d / "cut.jpg").write_bytes(full[: len(full) // 2])
        ok, ims, err = self._load(sorted(self.d.glob("*.jpg")))
        self.assertEqual([f.name for f in ok], ["a.jpg", "b.jpg"])
        self.assertEqual([im.size for im in ims], [(32, 24), (32, 24)])
        self.assertEqual(len(err.strip().splitlines()), 1)
        self.assertIn("skipped 4 of 6 files", err)
        self.assertIn("text, not an image", err)
        self.assertIn("1 empty file", err)

    def test_missing_decoder_is_reported_as_such(self):
        self._save("a.jpg")
        (self.d / "h.heic").write_bytes(HEIC_HEAD)
        with patch.object(_image, "HEIF", False):
            ok, _, err = self._load([self.d / "a.jpg", self.d / "h.heic"])
        self.assertEqual([f.name for f in ok], ["a.jpg"])
        self.assertIn("needs pillow-heif", err)

    def test_all_unreadable_exits_with_one_line(self):
        (self.d / "x.jpg").write_bytes(b"<html>blocked</html>")
        (self.d / "y.CR2").write_bytes(b"II*\x00\x10\x00\x00\x00CR\x02\x00")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            match._open_images([self.d / "x.jpg", self.d / "y.CR2"])
        self.assertEqual(len(str(caught.exception.code).splitlines()), 1)
        self.assertIn("none of the 2 candidate files", str(caught.exception.code))

    def _candidates(self):
        args = SimpleNamespace(images=str(self.d), items=None, panos=None, cache=str(self.d / ".cache"))
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            items, ims = match._load_candidates(args)
        return items, ims, err.getvalue()

    def test_mixed_folder_keeps_ids_aligned(self):
        self._save("first.jpg", color=(255, 0, 0))
        (self.d / "middle.jpg").write_bytes(HEIC_HEAD)
        self._save("last.png", "PNG", color=(0, 0, 255))
        self._save("UPPER.WEBP", "WEBP", color=(0, 255, 0))
        self._save("renamed.bin", "PNG", color=(255, 255, 0))
        items, ims, _ = self._candidates()
        ids = [it["id"] for it in items]
        self.assertEqual(ids[:2], ["first", "last"])
        self.assertEqual(sorted(ids[2:]), ["UPPER", "renamed"])
        colors = {it["id"]: im.convert("RGB").getpixel((0, 0)) for it, im in zip(items, ims)}
        self.assertGreater(colors["first"][0], 240)
        self.assertGreater(colors["last"][2], 240)
        self.assertGreater(colors["UPPER"][1], 240)

    @unittest.skipUnless(_image.HEIF, "pillow-heif not installed")
    def test_heic_only_folder(self):
        for i, c in enumerate(((255, 0, 0), (0, 0, 255))):
            Image.new("RGB", (40, 30), c).save(self.d / f"p{i}.HEIC", format="HEIF")
        Image.new("RGB", (40, 30), (0, 200, 0)).save(self.d / "phone.jpg", format="HEIF")
        items, ims, err = self._candidates()
        self.assertEqual([it["id"] for it in items], ["phone", "p0", "p1"])
        self.assertEqual([im.size for im in ims], [(40, 30)] * 3)
        self.assertEqual(err, "")


if __name__ == "__main__":
    unittest.main()
