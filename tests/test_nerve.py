"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Unit Tests: tests/test_nerve.py

Validates the Mandibular Inferior Alveolar Nerve Tracing Engine:
- Parametric spline & tube filter generation.
- Arc length calculation.
- 2D orthogonal slice intersection calculations.
- Centerline Euclidean distance query.
- Serialization to JSON/CSV.
"""

import json
import unittest
import math
from dental.nerve_tracer import NerveTracer, NerveChannel, NerveTrack


class TestNerveTracer(unittest.TestCase):
    """Unit tests for the Mandibular Nerve Tracer."""

    def setUp(self):
        self.tracer = NerveTracer()
        self.track = self.tracer.get_track(NerveChannel.LEFT)

    def test_add_and_undo_points(self):
        self.assertEqual(len(self.track.points), 0)

        # Add 3 points along a simulated mandibular canal path
        self.tracer.add_point(-14.0, -18.0, -20.0, channel=NerveChannel.LEFT)
        self.tracer.add_point(-16.0, -10.0, -15.0, channel=NerveChannel.LEFT)
        self.tracer.add_point(-18.0, 5.0, -8.0, channel=NerveChannel.LEFT)

        self.assertEqual(len(self.track.points), 3)
        self.assertGreater(self.track.get_total_length_mm(), 20.0)

        # Undo
        removed = self.tracer.undo_last_point(channel=NerveChannel.LEFT)
        self.assertEqual(removed, (-18.0, 5.0, -8.0))
        self.assertEqual(len(self.track.points), 2)

    def test_2d_slice_intersections(self):
        # Add points spanning from Z=-20 to Z=0
        self.tracer.add_point(-14.0, -18.0, -20.0, channel=NerveChannel.LEFT)
        self.tracer.add_point(-14.0, -18.0, 0.0, channel=NerveChannel.LEFT)

        # Intersect with Axial slice at Z=-10.0
        intersections = self.track.calculate_2d_slice_intersections(
            "axial", (0.0, 0.0, -10.0)
        )

        self.assertGreaterEqual(len(intersections), 1)
        ix, iy, iz, r = intersections[0]
        self.assertAlmostEqual(ix, -14.0, delta=0.5)
        self.assertAlmostEqual(iy, -18.0, delta=0.5)
        self.assertAlmostEqual(iz, -10.0, delta=0.5)
        self.assertAlmostEqual(r, 1.0)

    def test_distance_query(self):
        self.tracer.add_point(0.0, 0.0, 0.0, channel=NerveChannel.LEFT)
        self.tracer.add_point(0.0, 0.0, 10.0, channel=NerveChannel.LEFT)

        # Point (3.0, 4.0, 5.0) has distance 5.0 from the centerline along Z-axis
        query_pt = (3.0, 4.0, 5.0)
        dist, ch = self.tracer.get_distance_to_nearest_nerve(query_pt)

        self.assertAlmostEqual(dist, 5.0, delta=0.2)
        self.assertEqual(ch, "left")

    def test_export_json_csv(self):
        self.tracer.add_point(-14.0, -18.0, -20.0, channel=NerveChannel.LEFT)
        self.tracer.add_point(14.0, -18.0, -20.0, channel=NerveChannel.RIGHT)

        # JSON
        json_str = self.tracer.export_to_json()
        data = json.loads(json_str)
        self.assertIn("channels", data)
        self.assertEqual(len(data["channels"]["left"]["points"]), 1)
        self.assertEqual(len(data["channels"]["right"]["points"]), 1)

        # CSV
        csv_str = self.tracer.export_to_csv()
        self.assertIn("channel,index,x_mm,y_mm,z_mm", csv_str)
        self.assertIn("left,0,-14.000,-18.000,-20.000", csv_str)
        self.assertIn("right,0,14.000,-18.000,-20.000", csv_str)

    def test_clear_nerve(self):
        self.tracer.add_point(-14.0, -18.0, -20.0, channel=NerveChannel.LEFT)
        self.assertEqual(len(self.track.points), 1)

        self.tracer.clear_nerve(channel=NerveChannel.LEFT)
        self.assertEqual(len(self.track.points), 0)
        self.assertEqual(self.track.get_total_length_mm(), 0.0)


if __name__ == "__main__":
    unittest.main()
