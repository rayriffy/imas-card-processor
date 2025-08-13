import argparse
import os
from typing import List, Tuple, Optional

import cv2
import numpy as np


def ensure_dir_exists(directory_path: str) -> None:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)


def read_image_color(image_path: str) -> np.ndarray:
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")
    return image_bgr


def rotate_image_keep_size(image_bgr: np.ndarray, angle_degrees: float) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    rotated = cv2.warpAffine(
        image_bgr,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def estimate_skew_angle(binary_image: np.ndarray) -> float:
    # Use standard Hough transform on edges to estimate dominant angle (near 0 or 90)
    edges = cv2.Canny(binary_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, threshold=200)
    if lines is None:
        return 0.0

    angles = []
    for line in lines[:300]:
        rho, theta = line[0]
        angle_deg = np.degrees(theta)
        # Map angle to range [-90, 90)
        mapped = ((angle_deg + 90.0) % 180.0) - 90.0
        # Focus near horizontal (0 deg) and vertical (±90 deg)
        if -15 <= mapped <= 15:
            angles.append(mapped)
        elif 75 <= abs(mapped) <= 105:
            # Convert near-vertical to an equivalent small skew around 0
            if mapped > 0:
                angles.append(mapped - 90.0)
            else:
                angles.append(mapped + 90.0)

    if not angles:
        return 0.0

    # Use median to be robust
    return float(np.median(angles))


def adaptive_binary(gray_image: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=35,
        C=10,
    )


def extract_line_masks(binary_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    height, width = binary_image.shape[:2]

    # Kernel sizes proportional to image size
    horiz_kernel_len = max(10, width // 30)
    vert_kernel_len = max(10, height // 30)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_kernel_len, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_kernel_len))

    # Extract horizontal lines
    horizontal_mask = cv2.erode(binary_image, horizontal_kernel, iterations=1)
    horizontal_mask = cv2.dilate(horizontal_mask, horizontal_kernel, iterations=1)

    # Extract vertical lines
    vertical_mask = cv2.erode(binary_image, vertical_kernel, iterations=1)
    vertical_mask = cv2.dilate(vertical_mask, vertical_kernel, iterations=1)

    return horizontal_mask, vertical_mask


def group_true_runs(boolean_array: np.ndarray) -> List[Tuple[int, int]]:
    # Returns list of (start_index_inclusive, end_index_inclusive)
    runs: List[Tuple[int, int]] = []
    in_run = False
    run_start = 0
    for idx, is_true in enumerate(boolean_array):
        if is_true and not in_run:
            in_run = True
            run_start = idx
        elif not is_true and in_run:
            runs.append((run_start, idx - 1))
            in_run = False
    if in_run:
        runs.append((run_start, len(boolean_array) - 1))
    return runs


def detect_line_positions_from_mask(
    line_mask: np.ndarray, axis: int, length_threshold_ratio: float
) -> List[int]:
    # axis=0 for vertical lines (sum over rows), axis=1 for horizontal lines (sum over cols)
    # length_threshold_ratio: fraction of the opposite dimension that must be white to consider a line
    height, width = line_mask.shape[:2]
    if axis == 0:
        # Vertical: for each column, count white pixels
        counts = np.sum(line_mask > 0, axis=0)
        threshold = int(length_threshold_ratio * height)
        mask = counts >= threshold
        runs = group_true_runs(mask)
        centers = [int((start + end) / 2) for start, end in runs]
        return centers
    else:
        # Horizontal: for each row, count white pixels
        counts = np.sum(line_mask > 0, axis=1)
        threshold = int(length_threshold_ratio * width)
        mask = counts >= threshold
        runs = group_true_runs(mask)
        centers = [int((start + end) / 2) for start, end in runs]
        return centers


def add_image_borders_as_lines(
    positions: List[int], max_value: int, include_borders: bool
) -> List[int]:
    deduped = sorted(set(positions))
    if include_borders:
        deduped = sorted(set([0] + deduped + [max_value - 1]))
    return deduped


def compute_uniform_positions(max_value: int, cells: int) -> List[int]:
    if cells <= 0:
        return [0, max(0, max_value - 1)]
    # Uniform split including borders [0, ..., max-1]
    step = (max_value - 1) / float(cells)
    return [int(round(i * step)) for i in range(cells + 1)]


def kmeans_1d_positions(values: List[int], k: int, max_iters: int = 50) -> List[int]:
    # Simple 1D k-means. Returns sorted cluster centers as ints.
    if k <= 0:
        return []
    if len(values) == 0:
        return []
    values_np = np.array(sorted(values), dtype=np.float32)
    # Initialize centers uniformly across the value range
    min_v, max_v = float(values_np[0]), float(values_np[-1])
    if min_v == max_v:
        return [int(round(min_v))] * k
    centers = np.linspace(min_v, max_v, k).astype(np.float32)
    for _ in range(max_iters):
        # Assign
        distances = np.abs(values_np[:, None] - centers[None, :])
        assignments = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for ci in range(k):
            cluster_vals = values_np[assignments == ci]
            if cluster_vals.size > 0:
                new_centers[ci] = float(np.median(cluster_vals))
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    centers_sorted = sorted(int(round(c)) for c in centers.tolist())
    return centers_sorted


def adjust_positions_to_expected(
    positions: List[int], expected_cells: int, max_value: int
) -> List[int]:
    # Ensure borders
    positions_with_borders = sorted(set([0] + [int(p) for p in positions] + [max_value - 1]))

    if expected_cells is None or expected_cells <= 0:
        return positions_with_borders

    expected_edges = expected_cells + 1
    if len(positions_with_borders) == expected_edges:
        return positions_with_borders
    if len(positions_with_borders) > expected_edges:
        return kmeans_1d_positions(positions_with_borders, expected_edges)

    # Too few edges: synthesize uniform
    return compute_uniform_positions(max_value, expected_cells)


def crop_cells_from_grid(
    image_bgr: np.ndarray,
    horizontal_positions: List[int],
    vertical_positions: List[int],
    inner_margin_px: int,
    output_dir: str,
    zero_pad: bool,
    tighten_to_card: bool,
    tighten_margin_px: int,
    debug_overlay_path: Optional[str] = None,
) -> int:
    saved = 0
    num_rows = max(0, len(horizontal_positions) - 1)
    num_cols = max(0, len(vertical_positions) - 1)
    pad_fmt = 2 if zero_pad else 0

    debug_overlay = None
    if debug_overlay_path:
        debug_overlay = image_bgr.copy()

    for row_index in range(num_rows):
        y1 = horizontal_positions[row_index]
        y2 = horizontal_positions[row_index + 1]
        for col_index in range(num_cols):
            x1 = vertical_positions[col_index]
            x2 = vertical_positions[col_index + 1]

            y_start = max(y1 + inner_margin_px, 0)
            y_end = max(min(y2 - inner_margin_px, image_bgr.shape[0]), y_start + 1)
            x_start = max(x1 + inner_margin_px, 0)
            x_end = max(min(x2 - inner_margin_px, image_bgr.shape[1]), x_start + 1)

            if debug_overlay is not None:
                cv2.rectangle(
                    debug_overlay,
                    (int(x_start), int(y_start)),
                    (int(x_end - 1), int(y_end - 1)),
                    (0, 0, 255),
                    2,
                )

            card = image_bgr[y_start:y_end, x_start:x_end]
            if tighten_to_card:
                tightened = tighten_crop_to_card_edges(card, tighten_margin_px)
                if tightened is not None and tightened.size != 0:
                    card = tightened
            if card.size == 0:
                continue

            row_label = f"{row_index + 1:0{pad_fmt}d}" if zero_pad else f"{row_index + 1}"
            col_label = f"{col_index + 1:0{pad_fmt}d}" if zero_pad else f"{col_index + 1}"
            filename = f"{row_label}-{col_label}.png"
            cv2.imwrite(os.path.join(output_dir, filename), card)
            saved += 1

    if debug_overlay is not None:
        cv2.imwrite(debug_overlay_path, debug_overlay)

    return saved


def tighten_crop_to_card_edges(cell_bgr: np.ndarray, margin_px: int = 2) -> np.ndarray:
    height, width = cell_bgr.shape[:2]
    if height < 10 or width < 10:
        return cell_bgr

    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    candidates: List[np.ndarray] = []

    # Edges
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    candidates.append(edges)

    # Otsu (both polarities)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thr_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    candidates.append(thr)
    candidates.append(thr_inv)

    best_bbox = None
    best_score = -1.0
    cell_area = float(height * width)

    for cand in candidates:
        contours, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < 0.2 * cell_area or area > 0.995 * cell_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 10:
                continue
            rect_area = float(w * h)
            rectangularity = area / max(rect_area, 1.0)
            aspect_ratio = max(w, h) / float(min(w, h))
            if aspect_ratio > 3.5:
                continue
            score = area * rectangularity
            if score > best_score:
                best_score = score
                best_bbox = (x, y, w, h)

    if best_bbox is None:
        return cell_bgr

    x, y, w, h = best_bbox
    x0 = max(x + margin_px, 0)
    y0 = max(y + margin_px, 0)
    x1 = min(x + w - margin_px, width)
    y1 = min(y + h - margin_px, height)
    if x1 <= x0 or y1 <= y0:
        return cell_bgr
    return cell_bgr[y0:y1, x0:x1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect card grid and crop each card.")
    parser.add_argument("--input", required=True, help="Path to the scanned image.")
    parser.add_argument(
        "--output",
        required=True,
        help="Directory to save cropped cards (will be created if missing).",
    )
    parser.add_argument(
        "--length-threshold",
        type=float,
        default=0.5,
        help="Fraction of dimension a line must span to be considered (default: 0.5).",
    )
    parser.add_argument(
        "--inner-margin-px",
        type=int,
        default=5,
        help="Pixels shaved off inside each grid cell to avoid line artifacts (default: 5).",
    )
    parser.add_argument(
        "--include-image-borders",
        action="store_true",
        help="Treat image borders as grid lines if outer lines are missing.",
    )
    parser.add_argument(
        "--zero-pad",
        action="store_true",
        help="Zero-pad row/column numbers in filenames (e.g., 01-02.png).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate debug images into the output directory.",
    )
    parser.add_argument(
        "--tighten-to-card",
        action="store_true",
        help="Within each grid cell, tighten the crop to the dominant rectangular contour (card edges).",
    )
    parser.add_argument(
        "--tighten-margin-px",
        type=int,
        default=2,
        help="Margin to shave inside the detected card rectangle when tightening (default: 2).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Expected number of rows of cards (optional).",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Expected number of columns of cards (optional).",
    )
    parser.add_argument(
        "--force-equal-grid",
        action="store_true",
        help="Ignore detection and split the image into an equal grid using --rows and --cols.",
    )

    args = parser.parse_args()

    ensure_dir_exists(args.output)

    image_bgr = read_image_color(args.input)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = image_gray.shape[:2]

    if args.force_equal_grid and args.rows and args.cols:
        # Skip detection entirely
        binary = adaptive_binary(image_gray)
        if args.debug:
            horizontal_mask, vertical_mask = extract_line_masks(binary)
            cv2.imwrite(os.path.join(args.output, "_debug_01_binary.png"), binary)
            cv2.imwrite(os.path.join(args.output, "_debug_02_horizontal_mask.png"), horizontal_mask)
            cv2.imwrite(os.path.join(args.output, "_debug_03_vertical_mask.png"), vertical_mask)

        horizontal_positions = compute_uniform_positions(height, int(args.rows))
        vertical_positions = compute_uniform_positions(width, int(args.cols))
        print(f"Using equal grid: rows={args.rows} cols={args.cols} -> edges: H={len(horizontal_positions)} V={len(vertical_positions)}")
    else:
        binary = adaptive_binary(image_gray)

        # Optional deskew using Hough-based angle estimate on binary
        skew_angle_deg = estimate_skew_angle(binary)
        if abs(skew_angle_deg) > 0.5:
            image_bgr = rotate_image_keep_size(image_bgr, skew_angle_deg)
            image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            binary = adaptive_binary(image_gray)

        horizontal_mask, vertical_mask = extract_line_masks(binary)

        if args.debug:
            cv2.imwrite(os.path.join(args.output, "_debug_01_binary.png"), binary)
            cv2.imwrite(os.path.join(args.output, "_debug_02_horizontal_mask.png"), horizontal_mask)
            cv2.imwrite(os.path.join(args.output, "_debug_03_vertical_mask.png"), vertical_mask)

        horizontal_positions_raw = detect_line_positions_from_mask(
            horizontal_mask, axis=1, length_threshold_ratio=float(args.length_threshold)
        )
        vertical_positions_raw = detect_line_positions_from_mask(
            vertical_mask, axis=0, length_threshold_ratio=float(args.length_threshold)
        )

        print(
            f"Detected lines: H={len(horizontal_positions_raw)} V={len(vertical_positions_raw)}"
        )

        if args.include_image_borders:
            horizontal_positions_raw = add_image_borders_as_lines(horizontal_positions_raw, height, True)
            vertical_positions_raw = add_image_borders_as_lines(vertical_positions_raw, width, True)

        # Fit to expected counts if provided
        if args.rows is not None:
            horizontal_positions = adjust_positions_to_expected(
                horizontal_positions_raw, int(args.rows), height
            )
        else:
            horizontal_positions = sorted(set(horizontal_positions_raw))

        if args.cols is not None:
            vertical_positions = adjust_positions_to_expected(
                vertical_positions_raw, int(args.cols), width
            )
        else:
            vertical_positions = sorted(set(vertical_positions_raw))

        print(
            f"Final edges: H={len(horizontal_positions)} (expected {args.rows and int(args.rows)+1 or 'auto'}) "
            f"V={len(vertical_positions)} (expected {args.cols and int(args.cols)+1 or 'auto'})"
        )

    if len(horizontal_positions) < 2 or len(vertical_positions) < 2:
        raise RuntimeError(
            "Not enough grid edges to crop. Provide --rows and --cols, or use --force-equal-grid, or enable --include-image-borders."
        )

    saved = crop_cells_from_grid(
        image_bgr=image_bgr,
        horizontal_positions=sorted(horizontal_positions),
        vertical_positions=sorted(vertical_positions),
        inner_margin_px=int(args.inner_margin_px),
        output_dir=args.output,
        zero_pad=bool(args.zero_pad),
        tighten_to_card=bool(args.tighten_to_card),
        tighten_margin_px=int(args.tighten_margin_px),
        debug_overlay_path=(
            os.path.join(args.output, "_debug_04_crop_boxes.png") if args.debug else None
        ),
    )

    print(f"Saved {saved} cropped cards to: {args.output}")


if __name__ == "__main__":
    main()


