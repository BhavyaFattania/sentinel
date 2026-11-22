from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch

from build_sar_npy import load_s1
from export_yieldsat_fields import same_pixel_indices
from fetch_sar_rtc import search
from split_fields import AFF_FILES, load_split, require_fresh, write_manifest
from train_aff import FieldData, MirandaFeatureFusion, compute_norm
from train_fusion import FusionNet


class PilotTests(unittest.TestCase):
    def test_split_rejects_overlap_and_mismatch(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            rows = []
            for split in ("train", "val", "test"):
                field = root / split / "Argentina" / split
                field.mkdir(parents=True)
                for file in AFF_FILES:
                    (field / file).touch()
                rows.append(dict(country="Argentina", split=split, field=split))
            write_manifest(root / "split_manifest.csv", rows)
            self.assertEqual([len(x) for x in load_split(root, AFF_FILES)], [1, 1, 1])
            with self.assertRaisesRegex(ValueError, "not empty"):
                require_fresh(root)
            with self.assertRaisesRegex(ValueError, "missing required arrays"):
                load_split(root, AFF_FILES + ("sar.npy",))
            write_manifest(root / "split_manifest.csv", rows[:2])
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_split(root, AFF_FILES)
            (root / "test" / "Argentina" / "test").rename(root / "test" / "Argentina" / "train")
            with self.assertRaisesRegex(ValueError, "more than once"):
                load_split(root, AFF_FILES)

    def test_terrain_channel_constant_during_training_stays_finite(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            paths = []
            for name, terrain in (("train", 0), ("val", 2)):
                target = root / name
                target.mkdir()
                arrays = dict(s2=np.zeros((24, 12, 2, 2)), weather=np.zeros((24, 4)),
                              dem=np.full((5, 2, 2), terrain), soil=np.zeros((24, 2, 2)),
                              coord=np.zeros((3, 2, 2)), mask=np.ones((2, 2), bool))
                for key, value in arrays.items():
                    np.save(target / f"{key}.npy", value)
                np.save(target / "yield.npy", np.array([[1, 2], [3, 4]], np.float32))
                paths.append(target)
            stats, y_stats = compute_norm(paths[:1], False, "sar.npy")
            data = FieldData(paths[1], stats, y_stats, False, "sar.npy", -1)
            np.testing.assert_array_equal(data.dem, 2)

    def test_s2_mode_is_independent_of_radar_and_shares_initial_weights(self):
        net = FusionNet("s2", ch=2, T=2).eval()
        x = torch.ones(2, 12, 4, 4)
        sar = torch.ones(2, 2, 4, 4)
        valid = torch.ones(2, 1, 4, 4)
        s2_valid = valid.clone()
        s2_valid[0] = 0
        with torch.no_grad():
            a, _ = net.fuse(x, sar, s2_valid, valid)
            b, _ = net.fuse(x, -sar, s2_valid, torch.zeros_like(valid))
        torch.testing.assert_close(a, b)
        self.assertEqual(float(a[0].abs().sum()), 0)
        torch.manual_seed(17)
        optical = MirandaFeatureFusion()
        torch.manual_seed(17)
        fused = MirandaFeatureFusion(use_sar=True)
        for name, value in optical.state_dict().items():
            torch.testing.assert_close(value, fused.state_dict()[name])

    def test_label_grid_allows_subpixel_rounding_but_rejects_shift(self):
        optical = SimpleNamespace(shape=(45, 94), height=45, width=94, crs="EPSG:32633",
                                  transform=from_origin(409504.30, 5929829.01, 10, 10))
        labels = SimpleNamespace(shape=optical.shape, crs=optical.crs,
                                 transform=from_origin(409504.18, 5929829.18, 9.9894, 9.9894))
        self.assertTrue(same_pixel_indices(optical, labels))
        labels.transform = from_origin(409504.30, 5929839.01, 10, 10)
        self.assertFalse(same_pixel_indices(optical, labels))

    def test_catalog_search_paginates_and_rejects_out_of_season_items(self):
        def item(date):
            return {"properties": {"datetime": date + "T12:00:00Z"}}

        pages = [
            {"features": [item("2016-08-01"), item("2016-09-19")],
             "links": [{"rel": "next", "href": "https://example.org/next"}]},
            {"features": [item("2017-08-24"), item("2017-08-25")], "links": []},
        ]
        with patch("fetch_sar_rtc.get_json", side_effect=pages) as get:
            results = list(search((13.72, 53.50, 13.74, 53.51),
                                  np.datetime64("2016-09-19"), np.datetime64("2017-08-24")))
        self.assertEqual(results, [item("2016-09-19"), item("2017-08-24")])
        self.assertIn("/stac/v1/search?", get.call_args_list[0].args[0])
        self.assertEqual(get.call_args_list[1].args[0], "https://example.org/next")

    def test_radar_window_uses_target_grid(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "source.tif"
            transform = from_origin(500000, 9000000, 10, 10)
            source = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
            with rasterio.open(path, "w", driver="GTiff", crs="EPSG:32720", transform=transform,
                               width=4, height=4, count=2, dtype="float32") as f:
                f.write(source)
            out = load_s1(path, dict(crs="EPSG:32720", transform=from_origin(500010, 8999990, 10, 10),
                                     width=2, height=2))
            np.testing.assert_array_equal(out, source[:, 1:3, 1:3])

    def test_runner_propagates_trainer_failure(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "folds" / "Argentina" / "fold_0").mkdir(parents=True)
            runner = Path(__file__).resolve().parents[1] / "run_folds.py"
            result = subprocess.run([sys.executable, "-B", str(runner), "--folds", str(root / "folds"),
                                     "--runs", str(root / "runs"), "--modes", "s2"],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Empty train split", result.stderr)
            self.assertFalse((root / "runs" / "all_folds.csv").exists())


if __name__ == "__main__":
    unittest.main()
