"""模块用途：检验观测表的核算、对齐与文件保护；人工样本不代表实验结果。"""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_erco_observations.py"
FUELS = ["biomass", "coal", "hydro", "natural_gas", "nuclear", "other", "petroleum", "solar", "wind"]
G = "net_generation_mwh"
E = "co2_mass_kg_for_electricity"
F = "generated_co2_rate_kg_per_mwh_for_electricity"
C = "consumed_co2_rate_kg_per_mwh_for_electricity"
T = "UTC Time at End of Hour"


def fixtures(negative=False, idle_emissions=False):
    grid = pd.date_range("2023-01-01 06:00", periods=3, freq="h", tz="UTC").astype("datetime64[ns, UTC]")
    rows, carbon, eia = [], [], []
    for timestamp in grid:
        total_g, total_e = 0.0, 0.0
        for fuel in FUELS:
            generation = {"coal": 50.0, "natural_gas": 30.0, "wind": 20.0}.get(fuel, 0.0)
            emissions = {"coal": 45000.0, "natural_gas": 12000.0}.get(fuel, 0.0)
            if negative and fuel == "other":
                generation = -1.0
            if idle_emissions and fuel == "biomass":
                emissions = 10.0
            rows.append({"fuel_category": fuel, "datetime_utc": str(timestamp), G: generation, E: emissions, F: emissions / generation if generation > 0 else 0.0})
            total_g += generation
            total_e += emissions
        rows.append({"fuel_category": "total", "datetime_utc": str(timestamp), G: total_g, E: total_e, F: round(total_e / total_g, 2)})
        carbon.append({"datetime_utc": str(timestamp), C: 900.0, C + "_adjusted": 890.0})
        eia.append({"Balancing Authority": "ERCO", T: (timestamp + pd.Timedelta(hours=1)).strftime("%m/%d/%Y %I:%M:%S %p"), "Demand (MW)": 80.0, "Net Generation (MW)": 105.0, "Net Generation (MW) from Wind": 21.0, "Net Generation (MW) from Solar": 0.0, "Net Generation (MW) from Coal": 51.0, "Net Generation (MW) from Natural Gas": 33.0})
    return pd.DataFrame(rows), pd.DataFrame(carbon), pd.DataFrame(eia), grid


class ObservationTests(unittest.TestCase):
    def module(self):
        self.assertTrue(SCRIPT.is_file(), "build_erco_observations.py 尚未实现")
        spec = importlib.util.spec_from_file_location("erco_builder", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def build(self, *args, **kwargs):
        return self.module().build_observations(*args, **kwargs)

    def test_formula_alignment_and_boundaries(self):
        power, carbon, eia, grid = fixtures()
        original = [x.copy(deep=True) for x in (power, carbon, eia)]
        data, report = self.build(power, carbon, eia, grid)
        self.assertEqual(data.shape, (3, 40))
        self.assertTrue(data.index.equals(grid))
        self.assertEqual(data.iloc[0]["factor_generated_kg_per_mwh"], 570.0)
        self.assertEqual(data.iloc[0]["generation_total_mwh"], 100.0)
        self.assertEqual(data.iloc[0]["demand_mw"], 80.0)
        self.assertEqual(data.iloc[0]["reference_eia_generation_total_mw"], 105.0)
        self.assertEqual(data.iloc[0]["reference_consumed_factor_kg_per_mwh"], 900.0)
        self.assertTrue(data["factor_solar_kg_per_mwh"].isna().all())
        self.assertEqual(report["hourly_rows"], 3)
        for before, after in zip(original, (power, carbon, eia)):
            pd.testing.assert_frame_equal(before, after)

    def test_negative_generation_retained(self):
        data, report = self.build(*fixtures(negative=True))
        self.assertTrue(data["generation_other_mwh"].eq(-1).all())
        self.assertTrue(data["flag_negative_generation"].eq(1).all())
        self.assertTrue(data["factor_other_kg_per_mwh"].isna().all())
        self.assertAlmostEqual(data.iloc[0]["factor_generated_kg_per_mwh"], 57000 / 99)
        self.assertEqual(report["negative_generation_records"], 3)

    def test_zero_generation_emissions_still_in_total(self):
        data, _ = self.build(*fixtures(idle_emissions=True))
        self.assertAlmostEqual(data.iloc[0]["factor_generated_kg_per_mwh"], 570.1)
        self.assertTrue(data["factor_biomass_kg_per_mwh"].isna().all())

    def test_non_erco_records_ignored_before_date_parse(self):
        power, carbon, eia, grid = fixtures()
        other = eia.iloc[[0]].copy()
        other["Balancing Authority"] = "OTHER"
        other[T] = "invalid outside selected region"
        data, _ = self.build(power, carbon, pd.concat([eia, other]), grid)
        self.assertEqual(len(data), 3)

    def test_invalid_inputs_rejected(self):
        def duplicate_power(p, c, e):
            return pd.concat([p, p.iloc[[0]]]), c, e

        mutations = {
            "duplicate power": duplicate_power,
            "missing fuel record": lambda p, c, e: (p.iloc[1:], c, e),
            "missing total": lambda p, c, e: (p[p.fuel_category != "total"], c, e),
            "duplicate carbon": lambda p, c, e: (p, pd.concat([c, c.iloc[[0]]]), e),
            "duplicate EIA": lambda p, c, e: (p, c, pd.concat([e, e.iloc[[0]]])),
            "missing hour": lambda p, c, e: (p, c.iloc[1:], e),
            "missing column": lambda p, c, e: (p.drop(columns=[E]), c, e),
            "nonfinite mass": lambda p, c, e: (p.assign(**{E: np.inf}), c, e),
            "negative mass": lambda p, c, e: (p.assign(**{E: -1.0}), c, e),
            "nonfinite demand": lambda p, c, e: (p, c, e.assign(**{"Demand (MW)": np.nan})),
            "zero total generation": lambda p, c, e: (p.assign(**{G: 0.0}), c, e),
            "unknown fuel": lambda p, c, e: (p.replace({"fuel_category": {"wind": "unexpected"}}), c, e),
            "time offset": lambda p, c, e: (p, c, e.assign(**{T: "01/01/2023 6:30:00 AM"})),
            "null date": lambda p, c, e: (p, c.assign(datetime_utc=None), e),
            "no ERCO": lambda p, c, e: (p, c, e.assign(**{"Balancing Authority": "OTHER"})),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                p, c, e, grid = fixtures()
                module = self.module()
                with self.assertRaises(ValueError):
                    module.build_observations(*mutate(p, c, e), grid)

    def test_rounding_tolerance_and_large_total_mismatch(self):
        power, carbon, eia, grid = fixtures()
        totals = power.fuel_category.eq("total")
        power.loc[totals, G] += 0.03
        power.loc[totals, E] += 0.02
        power.loc[totals, F] = (power.loc[totals, E] / power.loc[totals, G]).round(2)
        _, report = self.build(power, carbon, eia, grid)
        self.assertAlmostEqual(report["max_component_sum_difference_from_total"]["net_generation_mwh"], 0.03)
        power.loc[totals, G] += 1
        with self.assertRaises(ValueError):
            self.build(power, carbon, eia, grid)

    def test_official_rate_mismatch_rejected(self):
        power, carbon, eia, grid = fixtures()
        power.loc[power.fuel_category.eq("total"), F] += 1
        with self.assertRaises(ValueError):
            self.build(power, carbon, eia, grid)

    def test_published_factor_accounts_for_rounded_mass_and_generation(self):
        power, carbon, eia, grid = fixtures()
        power[[G, E, F]] = 0.0
        selected = power.fuel_category.isin(["coal", "total"])
        # 真实发布记录的数字：三列独立舍入，不能只考虑因子自身的 0.005。
        power.loc[selected, G] = 44013.31
        power.loc[selected, E] = 13403154.26
        power.loc[selected, F] = 304.52
        data, _ = self.build(power, carbon, eia, grid)
        self.assertAlmostEqual(data.iloc[0]["factor_generated_kg_per_mwh"], 13403154.26 / 44013.31)
        power.loc[power.fuel_category.eq("total"), F] -= 0.01
        with self.assertRaises(ValueError):
            self.build(power, carbon, eia, grid)

    def test_missing_entire_hour_in_all_sources_rejected(self):
        power, carbon, eia, grid = fixtures()
        with self.assertRaises(ValueError):
            self.build(power.iloc[10:], carbon.iloc[1:], eia.iloc[1:], grid)

    def test_default_grid_is_full_erco_2023(self):
        power, carbon, eia, _ = fixtures()
        with self.assertRaises(ValueError):
            self.build(power, carbon, eia)

    def test_write_roundtrip_and_no_overwrite(self):
        module = self.module()
        data, report = module.build_observations(*fixtures())
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "observations"
            path = module.write_observations(data, report, output)
            self.assertEqual(path, output / "observations.csv")
            original = path.read_bytes()
            saved = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["output_sha256"], module.sha256_file(path))
            self.assertEqual(saved["status"], "OBSERVATIONS_BUILT_NOT_FORECAST_VALIDATED")
            with self.assertRaises(FileExistsError):
                module.write_observations(data, report, output)
            self.assertEqual(path.read_bytes(), original)

    def test_hash_and_missing_or_changed_source(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(FileNotFoundError):
                module.verify_sources(root)
            for relative in module.SOURCE_SHA256:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"abc")
            self.assertEqual(module.sha256_file(path), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                module.verify_sources(root)

    def test_cli_help_and_missing_inputs_no_output(self):
        self.module()
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run([sys.executable, str(SCRIPT), "--help"], cwd=temp, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([sys.executable, str(SCRIPT), "--project-root", temp], cwd=temp, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((Path(temp) / "data" / "processed").exists())

    def test_output_directory_cannot_target_raw_data(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(ValueError):
                module.run(root, root / "data" / "raw" / "new")
            self.assertFalse((root / "data" / "raw" / "new").exists())


if __name__ == "__main__":
    unittest.main()
