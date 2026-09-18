import os
import tempfile
import unittest

import cv2
import numpy as np

from vis import normalized_cxcywh_to_xyxy, save_detection_visualization
from engine import get_visualization_output_dir
from main_aitod import get_args_parser, resolve_resume_path


class DetectionVisualizationTests(unittest.TestCase):
    def test_explicit_resume_path_takes_precedence_over_output_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            open(os.path.join(directory, 'checkpoint.pth'), 'wb').close()

            self.assertEqual(
                resolve_resume_path('chosen.pth', directory),
                'chosen.pth',
            )

    def test_resume_path_defaults_to_output_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = os.path.join(directory, 'checkpoint.pth')
            open(checkpoint_path, 'wb').close()

            self.assertEqual(resolve_resume_path('', directory), checkpoint_path)

    def test_visualization_output_dir_prefers_custom_value(self):
        args = type('Args', (), {
            'output_dir': 'outputs',
            'vis_output_dir': 'custom_visualizations',
        })()

        self.assertEqual(
            get_visualization_output_dir(args),
            'custom_visualizations',
        )

    def test_visualization_output_dir_defaults_under_output_dir(self):
        args = type('Args', (), {
            'output_dir': 'outputs',
            'vis_output_dir': '',
        })()

        self.assertEqual(
            get_visualization_output_dir(args),
            os.path.join('outputs', 'visualizations'),
        )

    def test_visualization_parser_defaults(self):
        args = get_args_parser().parse_args([])

        self.assertFalse(args.visualize)
        self.assertEqual(args.vis_score_thresh, 0.3)
        self.assertEqual(args.vis_output_dir, '')

    def test_visualization_parser_overrides(self):
        args = get_args_parser().parse_args([
            '--visualize', '--vis_score_thresh', '0.55', '--vis_output_dir', 'rendered',
        ])

        self.assertTrue(args.visualize)
        self.assertEqual(args.vis_score_thresh, 0.55)
        self.assertEqual(args.vis_output_dir, 'rendered')

    def test_normalized_cxcywh_to_xyxy_converts_boxes_and_handles_empty(self):
        boxes = np.array([[0.5, 0.5, 0.4, 0.2]], dtype=np.float32)

        converted = normalized_cxcywh_to_xyxy(boxes, width=100, height=50)

        np.testing.assert_allclose(converted, [[30.0, 20.0, 70.0, 30.0]])
        self.assertEqual(
            normalized_cxcywh_to_xyxy(np.empty((0, 4)), 100, 50).shape,
            (0, 4),
        )

    def test_save_detection_visualization_writes_gt_and_thresholded_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "source.png")
            save_dir = os.path.join(directory, "visualizations")
            image = np.full((40, 60, 3), 255, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(image_path, image))

            saved = save_detection_visualization(
                image_path=image_path,
                image_id=7,
                prediction={
                    "boxes": np.array([[2, 2, 12, 12], [30, 2, 40, 12]], dtype=np.float32),
                    "scores": np.array([0.3, 0.29], dtype=np.float32),
                    "labels": np.array([0, 1]),
                },
                gt_boxes=np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
                gt_labels=np.array([1]),
                save_dir=save_dir,
                score_thresh=0.3,
            )

            gt_path = os.path.join(save_dir, "gt", "7_source.png")
            pred_path = os.path.join(save_dir, "pred", "7_source.png")
            self.assertTrue(saved)
            self.assertTrue(os.path.isfile(gt_path))
            self.assertTrue(os.path.isfile(pred_path))

            gt_image = cv2.imread(gt_path)
            self.assertFalse(np.array_equal(gt_image[16, 24], [255, 255, 255]))

            pred_image = cv2.imread(pred_path)
            self.assertFalse(np.array_equal(pred_image[2, 2], [255, 255, 255]))
            self.assertTrue(np.array_equal(pred_image[2, 30], [255, 255, 255]))

    def test_save_detection_visualization_returns_false_for_missing_image(self):
        with tempfile.TemporaryDirectory() as directory:
            saved = save_detection_visualization(
                image_path=os.path.join(directory, "missing.png"),
                image_id="missing",
                prediction={"boxes": [], "scores": [], "labels": []},
                gt_boxes=[],
                gt_labels=[],
                save_dir=os.path.join(directory, "visualizations"),
            )

        self.assertFalse(saved)


if __name__ == "__main__":
    unittest.main()
