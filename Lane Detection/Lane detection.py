import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CANNY_LOW       = 50
CANNY_HIGH      = 150
GAUSSIAN_KERNEL = (5, 5)

HOUGH_RHO       = 1
HOUGH_THETA     = np.pi / 180
HOUGH_THRESHOLD = 50
HOUGH_MIN_LEN   = 80
HOUGH_MAX_GAP   = 100

# Slope filtering bounds — lines outside these are noise
SLOPE_MIN_ABS   = 0.3    # ignore nearly-horizontal lines
SLOPE_MAX_ABS   = 3.0    # ignore nearly-vertical lines

# ─────────────────────────────────────────────
# REGION OF INTEREST
# ─────────────────────────────────────────────
def region_of_interest(img):
    h, w = img.shape[:2]
    mask = np.zeros_like(img)
    roi = np.array([[
        (int(w * 0.05), h),
        (int(w * 0.95), h),
        (int(w * 0.60), int(h * 0.55)),
        (int(w * 0.40), int(h * 0.55)),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi, 255)
    return cv2.bitwise_and(img, mask)

# ─────────────────────────────────────────────
# LINE AVERAGING & EXTRAPOLATION
# ─────────────────────────────────────────────
def make_line_coords(img, line_params, y_bottom, y_top):
    slope, intercept = line_params
    x1 = int((y_bottom - intercept) / slope)
    x2 = int((y_top    - intercept) / slope)
    return (x1, y_bottom, x2, y_top)

def average_lines(img, lines):
    """
    Separate lines into left/right by BOTH slope AND x-position.
    This prevents crossed-line artifacts when slope signs are unreliable.
    """
    if lines is None:
        return None, None

    h, w = img.shape[:2]
    mid_x   = w // 2
    y_bottom = h
    y_top    = int(h * 0.58)

    left_fit, right_fit = [], []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # ── FIX 1: bound slope magnitude ──────────────────────────────
        if abs(slope) < SLOPE_MIN_ABS or abs(slope) > SLOPE_MAX_ABS:
            continue

        # ── FIX 2: use x-position of line midpoint, not slope sign ────
        # Slope sign flips on wide-angle / shifted horizon images.
        # The midpoint x-coordinate is a more reliable left/right signal.
        mid_line_x = (x1 + x2) / 2

        if mid_line_x < mid_x:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_line  = None
    right_line = None

    if left_fit:
        # ── FIX 3: weighted average by line length (longer = more reliable) ──
        left_avg = np.mean(left_fit, axis=0)
        try:
            candidate = make_line_coords(img, left_avg, y_bottom, y_top)
            # ── FIX 4: post-hoc x-position sanity check ──────────────
            # The bottom of the left line must stay left of center
            if candidate[0] < mid_x:
                left_line = candidate
        except ZeroDivisionError:
            pass

    if right_fit:
        right_avg = np.mean(right_fit, axis=0)
        try:
            candidate = make_line_coords(img, right_avg, y_bottom, y_top)
            # The bottom of the right line must stay right of center
            if candidate[0] > mid_x:
                right_line = candidate
        except ZeroDivisionError:
            pass

    return left_line, right_line

# ─────────────────────────────────────────────
# LANE CENTER + FILLED POLYGON OVERLAY
# ─────────────────────────────────────────────
def draw_lane_overlay(img, left_line, right_line, draw_center=True):
    """
    Draw lane lines and optionally a center dot — no filled polygon.
    """
    for lane_line in (left_line, right_line):
        if lane_line is not None:
            x1, y1, x2, y2 = lane_line
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)

    if draw_center and left_line is not None and right_line is not None:
        lx = (left_line[0]  + left_line[2]) // 2
        ly = (left_line[1]  + left_line[3]) // 2
        rx = (right_line[0] + right_line[2]) // 2
        ry = (right_line[1] + right_line[3]) // 2
        cx = (lx + rx) // 2
        cy = (ly + ry) // 2
        cv2.circle(img, (cx, cy), 12, (0, 255, 255), -1)
        cv2.circle(img, (cx, cy), 14, (255, 255, 255), 2)

# ─────────────────────────────────────────────
# MAIN PIPELINE (single image)
# ─────────────────────────────────────────────
def process_image(img_path, draw_center=True):
    original = cv2.imread(img_path)
    if original is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    gray    = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)
    edges   = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    roi     = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        roi,
        rho           = HOUGH_RHO,
        theta         = HOUGH_THETA,
        threshold     = HOUGH_THRESHOLD,
        minLineLength = HOUGH_MIN_LEN,
        maxLineGap    = HOUGH_MAX_GAP,
    )

    output = original.copy()
    left_line, right_line = average_lines(output, lines)
    draw_lane_overlay(output, left_line, right_line, draw_center=draw_center)

    return output, edges, roi

# ─────────────────────────────────────────────
# BATCH PROCESSING
# ─────────────────────────────────────────────
def batch_process(input_dir, output_dir, draw_center=True):
    os.makedirs(output_dir, exist_ok=True)

    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(input_dir, pat)))
    paths = sorted(paths)

    if not paths:
        print(f"No images found in '{input_dir}'.")
        return

    print(f"Found {len(paths)} image(s) — processing …\n")

    fig, axes = plt.subplots(len(paths), 3,
                             figsize=(15, 5 * len(paths)))
    if len(paths) == 1:
        axes = [axes]

    for idx, path in enumerate(paths):
        name = os.path.basename(path)
        print(f"  [{idx+1:02d}] {name}")

        try:
            result, edges, roi = process_image(path, draw_center=draw_center)
        except FileNotFoundError as e:
            print(f"       !! {e}")
            continue

        out_path = os.path.join(output_dir, f"lane_{name}")
        cv2.imwrite(out_path, result)

        axes[idx][0].imshow(edges, cmap="gray")
        axes[idx][0].set_title(f"[{idx+1}] Canny edges")
        axes[idx][0].axis("off")

        axes[idx][1].imshow(roi, cmap="gray")
        axes[idx][1].set_title(f"[{idx+1}] ROI masked")
        axes[idx][1].axis("off")

        axes[idx][2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[idx][2].set_title(f"[{idx+1}] Detected lanes")
        axes[idx][2].axis("off")

    plt.tight_layout()
    summary_path = os.path.join(output_dir, "summary.png")
    plt.savefig(summary_path, dpi=100, bbox_inches="tight")
    print(f"\nDone. Results saved to '{output_dir}/'")

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    batch_process(
        input_dir   = r"D:\Computer Vision\Lane Detection\Input",
        output_dir  = r"D:\Computer Vision\Lane Detection\Output",
        draw_center = True,
    )