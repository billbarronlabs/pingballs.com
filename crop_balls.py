"""
Golf Ball Image Cropper
-----------------------
Hybrid detection: gradient-based Hough + color-distance from towel.
Picks the best circle from both methods, then refines with radial
edge scanning.

Usage:
    python crop_balls.py
"""

import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import cv2


INPUT_DIR = Path("ballimages")
OUTPUT_DIR = Path("ballimages/cropped")
THUMB_SIZE = 80
LARGE_SIZE = 400
WORK_SIZE = 800


def score_circle(cx, cy, r, grad_norm, sw, sh):
    """Score a circle candidate based on centrality, bounds, size, edge strength."""
    # Centrality — ball should be near image center
    dx = (cx - sw / 2) / (sw / 2)
    dy = (cy - sh / 2) / (sh / 2)
    center_score = max(0, 1.0 - math.sqrt(dx * dx + dy * dy))

    # Bounds — circle should fit within image; soft penalty if it extends
    margin = r * 0.05
    overshoot = 0
    if cx - r < -margin:
        overshoot += abs(cx - r + margin)
    if cy - r < -margin:
        overshoot += abs(cy - r + margin)
    if cx + r > sw + margin:
        overshoot += (cx + r) - (sw + margin)
    if cy + r > sh + margin:
        overshoot += (cy + r) - (sh + margin)
    # Soft penalty: each pixel of overshoot reduces score gradually
    fits = max(0.1, 1.0 - overshoot / (r * 0.5 + 1))

    # Size preference — ball radius as fraction of shorter dimension
    # Accept wide range: 0.20 to 0.50 of min dimension
    size_frac = r / min(sh, sw)
    if 0.20 < size_frac < 0.50:
        size_score = 1.0
    elif size_frac >= 0.50:
        # Gentle falloff for large balls (tight crops)
        size_score = max(0.1, 1.0 - (size_frac - 0.50) * 3)
    else:
        size_score = max(0, 0.5 - abs(size_frac - 0.35))

    # Edge strength along perimeter of the gradient image
    num_samples = 72
    edge_vals = []
    in_bounds = 0
    for i in range(num_samples):
        angle = 2 * math.pi * i / num_samples
        px = int(cx + math.cos(angle) * r)
        py = int(cy + math.sin(angle) * r)
        if 0 <= px < sw and 0 <= py < sh:
            edge_vals.append(float(grad_norm[py, px]))
            in_bounds += 1
    edge_score = (np.mean(edge_vals) / 255.0) if edge_vals else 0
    # Penalize if too many perimeter samples fall outside bounds
    coverage = in_bounds / num_samples
    if coverage < 0.6:
        edge_score *= coverage

    return center_score * fits * size_score * (0.3 + edge_score)


def refine_circle(grad_norm, cx, cy, r, sw, sh):
    """
    Refine circle by scanning radially on gradient image.
    Find strongest gradient peak near expected radius at each angle,
    then fit a circle to those edge points.

    Also detects asymmetric edge coverage (strong edges on one side,
    weak on the other) and nudges the center toward the weak side,
    since a lopsided detection usually means the center is biased
    away from the blending edge.
    """
    num_angles = 72
    edge_points = []
    angle_strengths = []  # (angle, strength, found_in_bounds)

    r_min_scan = max(10, int(r * 0.60))
    r_max_scan = min(int(r * 1.50), max(sw, sh) - 1)

    for i in range(num_angles):
        angle = 2 * math.pi * i / num_angles
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        best_val = 0
        best_dist = r

        for d in range(r_min_scan, r_max_scan + 1):
            px = int(cx + cos_a * d)
            py = int(cy + sin_a * d)
            if 0 <= px < sw and 0 <= py < sh:
                val = float(grad_norm[py, px])
                if val > best_val:
                    best_val = val
                    best_dist = d

        angle_strengths.append((angle, best_val))

        if best_val > 20:
            ex = cx + cos_a * best_dist
            ey = cy + sin_a * best_dist
            edge_points.append((ex, ey, best_dist))

    if len(edge_points) < 20:
        return cx, cy, r

    # Two-pass circle fit with outlier removal
    def fit_circle(pts):
        """Algebraic least-squares circle fit."""
        x = pts[:, 0]
        y = pts[:, 1]
        A = np.column_stack([x, y, np.ones(len(x))])
        b_vec = -(x**2 + y**2)
        res, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
        D, E, F = res
        fcx = -D / 2.0
        fcy = -E / 2.0
        val = fcx**2 + fcy**2 - F
        if val < 0:
            return None
        fr = math.sqrt(val)
        return fcx, fcy, fr

    points = np.array([(p[0], p[1]) for p in edge_points])

    # Pass 1: fit all points
    result1 = fit_circle(points)
    if result1 is None:
        return cx, cy, r
    f1cx, f1cy, f1r = result1

    # Pass 2: remove points far from the fitted circle, refit
    residuals = np.sqrt((points[:, 0] - f1cx)**2 + (points[:, 1] - f1cy)**2) - f1r
    abs_res = np.abs(residuals)
    threshold = np.median(abs_res) * 2.5
    inliers = points[abs_res <= threshold]

    if len(inliers) >= 15:
        result2 = fit_circle(inliers)
        if result2 is not None:
            f1cx, f1cy, f1r = result2

    fit_cx, fit_cy, fit_r = f1cx, f1cy, f1r

    # Sanity checks — don't let refinement move too far
    shift = math.sqrt((fit_cx - cx)**2 + (fit_cy - cy)**2)
    if shift > r * 0.30 or fit_r < r * 0.70 or fit_r > r * 1.30:
        dists = sorted([p[2] for p in edge_points])
        median_r = dists[len(dists) // 2]
        # Don't let median_r vary too much either
        if median_r < r * 0.80 or median_r > r * 1.20:
            median_r = r
        fit_cx, fit_cy, fit_r = cx, cy, median_r

    # --- Asymmetry correction ---
    # Measure average edge strength in each quadrant / half to detect
    # when one side of the ball has weak edges (blending with background).
    # If there's a significant asymmetry, nudge center toward the weak side
    # because the detection was biased toward the strong side.
    strengths = np.array(angle_strengths)  # (angle, strength)

    # Compute weighted centroid of strong-edge angles
    # Strong edges pull the centroid toward that side; the true center
    # is offset in the opposite direction.
    strong_threshold = np.median(strengths[:, 1]) * 0.5
    wx_sum, wy_sum, w_sum = 0.0, 0.0, 0.0
    for angle, strength in angle_strengths:
        if strength > strong_threshold:
            w = strength
            wx_sum += math.cos(angle) * w
            wy_sum += math.sin(angle) * w
            w_sum += w

    if w_sum > 0:
        # Bias direction = where strong edges cluster (unit vector)
        bias_x = wx_sum / w_sum
        bias_y = wy_sum / w_sum
        bias_mag = math.sqrt(bias_x**2 + bias_y**2)

        # Only apply correction if bias is significant (>0.20 = edges
        # not evenly distributed). Max nudge = 10% of radius.
        if bias_mag > 0.20:
            nudge = min(bias_mag * 0.20, 0.10) * fit_r
            # Nudge OPPOSITE to bias (toward weak side)
            fit_cx -= bias_x / bias_mag * nudge
            fit_cy -= bias_y / bias_mag * nudge

    return fit_cx, fit_cy, fit_r


def hough_detect(grad_norm, sh, sw):
    """Find best circle using Hough on gradient image."""
    min_r = int(min(sh, sw) * 0.15)
    max_r = int(min(sh, sw) * 0.55)

    candidates = []
    for p1 in [100, 80, 60]:
        for p2 in [40, 35, 30, 25, 20]:
            circles = cv2.HoughCircles(
                grad_norm, cv2.HOUGH_GRADIENT, dp=1.2,
                minDist=min(sh, sw) // 4,
                param1=p1, param2=p2,
                minRadius=min_r, maxRadius=max_r
            )
            if circles is not None:
                for c in circles[0]:
                    candidates.append((float(c[0]), float(c[1]), float(c[2])))

    if not candidates:
        return None, 0

    best = None
    best_score = -1
    for cx, cy, r in candidates:
        score = score_circle(cx, cy, r, grad_norm, sw, sh)
        if score > best_score:
            best_score = score
            best = (cx, cy, r)

    return best, best_score


def get_background_color(smooth, sh, sw):
    """
    Sample background color from image edges, filtering out inconsistent
    corners (e.g. UI chrome in screenshots). Uses median of edge strips
    rather than just corners for robustness.
    """
    m = 20  # margin pixels
    strips = []

    # Top and bottom edge strips (full width)
    strips.append(smooth[0:m, :, :].reshape(-1, 3))
    strips.append(smooth[sh - m:sh, :, :].reshape(-1, 3))
    # Left and right edge strips (full height)
    strips.append(smooth[:, 0:m, :].reshape(-1, 3))
    strips.append(smooth[:, sw - m:sw, :].reshape(-1, 3))

    # Get median color of each strip
    strip_colors = [np.median(s, axis=0) for s in strips]

    # Check consistency — filter out strips that are very different
    # (handles screenshots where top/bottom have UI chrome)
    all_colors = np.array(strip_colors)
    median_color = np.median(all_colors, axis=0)
    dists = [np.linalg.norm(c - median_color) for c in strip_colors]

    # Keep strips within 50 color distance of median
    good_strips = [s for s, d in zip(strips, dists) if d < 50]

    if len(good_strips) >= 2:
        combined = np.concatenate(good_strips, axis=0)
    else:
        # Fall back to all strips if filtering is too aggressive
        combined = np.concatenate(strips, axis=0)

    return np.median(combined, axis=0)


def color_distance_detect(smooth, grad_norm, sh, sw):
    """Find ball by color distance from background.

    Uses mask asymmetry correction: when the detected mask is lopsided
    (one side of ball blends with background), the minEnclosingCircle
    center gets biased toward the strong-contrast side. We detect this
    by comparing the mask centroid to the enclosing circle center and
    nudge back toward the true geometric center.
    """
    bg_color = get_background_color(smooth, sh, sw)

    smooth_f = smooth.astype(np.float32)
    diff = smooth_f - bg_color.reshape(1, 1, 3)
    color_dist = np.sqrt(np.sum(diff**2, axis=2))
    color_dist_norm = (color_dist / (color_dist.max() + 1e-6) * 255).astype(np.uint8)

    _, mask = cv2.threshold(
        color_dist_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0

    largest = max(contours, key=cv2.contourArea)
    (mcx, mcy), mradius = cv2.minEnclosingCircle(largest)

    # --- Mask asymmetry correction ---
    # Compute mask centroid (center of mass of the white pixels).
    # For a perfect circle, centroid == enclosing circle center.
    # When the mask is lopsided (one side cut off by color blending),
    # the centroid is biased toward the detected side. The true ball
    # center is on the OPPOSITE side of the centroid from where it's
    # biased, i.e. the enclosing circle center is closer to truth.
    # We extrapolate: true_center = enclosing_center + (enclosing_center - centroid) * factor
    M = cv2.moments(largest)
    if M["m00"] > 0:
        cent_x = M["m10"] / M["m00"]
        cent_y = M["m01"] / M["m00"]

        # How far centroid is from enclosing center, relative to radius
        drift_x = mcx - cent_x
        drift_y = mcy - cent_y
        drift = math.sqrt(drift_x**2 + drift_y**2)
        drift_frac = drift / (mradius + 1e-6)

        # If significant drift (>3% of radius), correct by pushing center
        # FURTHER from centroid (same direction as enclosing-to-centroid offset)
        if drift_frac > 0.03:
            correction = min(drift_frac * 1.5, 0.15) * mradius
            if drift > 0:
                corr_cx = mcx + (drift_x / drift) * correction
                corr_cy = mcy + (drift_y / drift) * correction
            else:
                corr_cx, corr_cy = mcx, mcy
        else:
            corr_cx, corr_cy = mcx, mcy
    else:
        corr_cx, corr_cy = mcx, mcy

    # Try multiple center/radius combinations and pick best score
    best_cx, best_cy, best_r = mcx, mcy, mradius
    best_score = score_circle(mcx, mcy, mradius, grad_norm, sw, sh)

    # Option 1b: corrected center + enclosing radius
    s = score_circle(corr_cx, corr_cy, mradius, grad_norm, sw, sh)
    if s > best_score:
        best_cx, best_cy, best_r = corr_cx, corr_cy, mradius
        best_score = s

    if len(largest) >= 5:
        ellipse = cv2.fitEllipse(largest)
        ecx, ecy = ellipse[0]
        ew, eh = ellipse[1]
        aspect = max(ew, eh) / (min(ew, eh) + 1e-6)
        e_radius = (ew + eh) / 4.0

        if aspect < 1.5:
            # Option 2: ellipse center + minEnclosing radius
            s = score_circle(ecx, ecy, mradius, grad_norm, sw, sh)
            if s > best_score:
                best_cx, best_cy, best_r = ecx, ecy, mradius
                best_score = s

            # Option 3: ellipse center + ellipse radius
            s = score_circle(ecx, ecy, e_radius, grad_norm, sw, sh)
            if s > best_score:
                best_cx, best_cy, best_r = ecx, ecy, e_radius
                best_score = s

            # Option 4: averaged center (splits the difference when
            # contour is lopsided) + minEnclosing radius
            avg_cx = (mcx + ecx) / 2
            avg_cy = (mcy + ecy) / 2
            s = score_circle(avg_cx, avg_cy, mradius, grad_norm, sw, sh)
            if s > best_score:
                best_cx, best_cy, best_r = avg_cx, avg_cy, mradius
                best_score = s

            # Option 5: corrected center + ellipse center averaged
            avg2_cx = (corr_cx + ecx) / 2
            avg2_cy = (corr_cy + ecy) / 2
            s = score_circle(avg2_cx, avg2_cy, mradius, grad_norm, sw, sh)
            if s > best_score:
                best_cx, best_cy, best_r = avg2_cx, avg2_cy, mradius
                best_score = s

    return (best_cx, best_cy, best_r), best_score


def detect_ball(img_path):
    """
    Hybrid detection:
    1. Super-smooth to eliminate dimple texture
    2. Gradient magnitude for Hough-based detection
    3. Color distance from background for contour-based detection
    4. Pick the better result, then refine with radial edge scan
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    oh, ow = img.shape[:2]
    scale = WORK_SIZE / max(oh, ow)
    img_s = cv2.resize(img, (int(ow * scale), int(oh * scale)))
    sh, sw = img_s.shape[:2]

    # Super-smooth
    smooth = img_s.copy()
    for _ in range(5):
        smooth = cv2.bilateralFilter(smooth, 9, 75, 75)
    smooth = cv2.GaussianBlur(smooth, (25, 25), 0)
    smooth = cv2.GaussianBlur(smooth, (25, 25), 0)

    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)

    # Gradient magnitude
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    grad = np.sqrt(sobelx**2 + sobely**2)
    grad_norm = (grad / (grad.max() + 1e-6) * 255).astype(np.uint8)

    # Method 1: Hough on gradient
    hough_result, hough_score = hough_detect(grad_norm, sh, sw)

    # Method 2: Color distance from background
    color_result, color_score = color_distance_detect(smooth, grad_norm, sh, sw)

    # Pick the better one, with tiebreaker logic
    if hough_result and color_result:
        hi_score = max(hough_score, color_score)
        lo_score = min(hough_score, color_score)

        if hi_score > 0 and lo_score / hi_score > 0.85:
            # Scores are close — combine both methods with image center
            # as stabilizing anchor.
            hcx, hcy, hr = hough_result
            ccx, ccy, cr = color_result
            img_cx, img_cy = sw / 2, sh / 2

            center_dist = math.sqrt((hcx - ccx)**2 + (hcy - ccy)**2)

            # When the two methods disagree significantly on center
            # (>15% of radius), both are probably biased in different
            # directions. Use the image center as a third anchor since
            # the ball is typically the main subject of the photo.
            if center_dist > cr * 0.15:
                # When methods disagree, both are biased. The image
                # center is a strong prior because the ball is typically
                # the subject of the photo. Use average of the two
                # detections as a "detected center", then blend heavily
                # toward image center.
                det_cx = (ccx + hcx) / 2
                det_cy = (ccy + hcy) / 2
                # 40% detected average, 60% image center
                cx = det_cx * 0.40 + img_cx * 0.60
                cy = det_cy * 0.40 + img_cy * 0.60
                method = "blended(3way)"
            else:
                # Methods agree on center — use color center (more
                # reliable for full ball boundary)
                cx = ccx * 0.7 + hcx * 0.3
                cy = ccy * 0.7 + hcy * 0.3
                method = "blended(tie)"
            r = cr  # Color radius is more reliable for ball size
        elif hough_score >= color_score:
            cx, cy, r = hough_result
            method = "hough"
        else:
            cx, cy, r = color_result
            method = "color"
    elif hough_result:
        cx, cy, r = hough_result
        method = "hough"
    elif color_result:
        cx, cy, r = color_result
        method = "color"
    else:
        return None

    print(f"  Method: {method} (hough={hough_score:.3f}, color={color_score:.3f})")

    # Refine with radial edge scan.
    # Skip refinement for 3-way blends — the pre-refinement center
    # (weighted toward image center) is more reliable than what the
    # gradient-based refinement produces, because the gradient scan
    # gets confused by carpet/towel texture beyond the blending edge.
    if "3way" not in method:
        cx, cy, r = refine_circle(grad_norm, cx, cy, r, sw, sh)

    cx = int(cx / scale)
    cy = int(cy / scale)
    r = int(r / scale)

    # Return confidence flag: "3way" means methods disagreed significantly
    return cx, cy, r, method


def crop_to_circle(img_path, output_large, output_thumb, cx, cy, r, method=""):
    """Crop ball with transparent circular mask."""
    # Use tighter crop when detection was uncertain (3-way blend
    # means Hough and color disagreed, so edges may be imprecise)
    if "3way" in method:
        shrink = 0.90
    else:
        shrink = 0.94
    r = int(r * shrink)

    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img)  # Match OpenCV's auto-rotation
    img = img.convert("RGBA")
    iw, ih = img.size

    left = max(0, cx - r)
    top = max(0, cy - r)
    right = min(iw, cx + r)
    bottom = min(ih, cy + r)

    cropped = img.crop((left, top, right, bottom))
    cw, ch = cropped.size

    size = min(cw, ch)
    ox = (cw - size) // 2
    oy = (ch - size) // 2
    cropped = cropped.crop((ox, oy, ox + size, oy + size))

    aa = 4
    mask_big = Image.new("L", (size * aa, size * aa), 0)
    ImageDraw.Draw(mask_big).ellipse(
        [0, 0, size * aa - 1, size * aa - 1], fill=255
    )
    mask = mask_big.resize((size, size), Image.LANCZOS)
    cropped.putalpha(mask)

    large = cropped.resize((LARGE_SIZE, LARGE_SIZE), Image.LANCZOS)
    large.save(output_large, "WEBP", quality=90)

    thumb = cropped.resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
    thumb.save(output_thumb, "WEBP", quality=85)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = [
        f for f in INPUT_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ]

    if not images:
        print("No images found in", INPUT_DIR)
        return

    print(f"Found {len(images)} images to process.\n")

    ok = 0
    for img_path in sorted(images):
        stem = img_path.stem.lower()
        out_large = OUTPUT_DIR / f"{stem}.webp"
        out_thumb = OUTPUT_DIR / f"{stem}-thumb.webp"

        print(f"Processing: {img_path.name}")
        result = detect_ball(img_path)
        if result is None:
            print("  FAILED\n")
            continue

        cx, cy, r, method = result
        print(f"  Ball: center=({cx},{cy}), r={r}")
        crop_to_circle(img_path, out_large, out_thumb, cx, cy, r, method)
        ok += 1
        print(f"  -> {out_large.name}, {out_thumb.name}\n")

    print(f"Done! {ok}/{len(images)} processed.")


if __name__ == "__main__":
    main()
