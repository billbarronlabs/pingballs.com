import cv2
import numpy as np
import math

img = cv2.imread("ballimages/225.JPG")
oh, ow = img.shape[:2]
scale = 800 / max(oh, ow)
img_s = cv2.resize(img, (int(ow * scale), int(oh * scale)))
sh, sw = img_s.shape[:2]

print(f"Work image: {sw}x{sh}")

# Super-smooth  
smooth = img_s.copy()
for _ in range(5):
    smooth = cv2.bilateralFilter(smooth, 9, 75, 75)
smooth = cv2.GaussianBlur(smooth, (25, 25), 0)
smooth = cv2.GaussianBlur(smooth, (25, 25), 0)

# Background color
m = 20
strips = []
strips.append(smooth[0:m, :, :].reshape(-1, 3))
strips.append(smooth[sh-m:sh, :, :].reshape(-1, 3))
strips.append(smooth[:, 0:m, :].reshape(-1, 3))
strips.append(smooth[:, sw-m:sw, :].reshape(-1, 3))
strip_colors = [np.median(s, axis=0) for s in strips]
all_colors = np.array(strip_colors)
median_color = np.median(all_colors, axis=0)
dists = [np.linalg.norm(c - median_color) for c in strip_colors]
good_strips = [s for s, d in zip(strips, dists) if d < 50]
if len(good_strips) >= 2:
    combined = np.concatenate(good_strips, axis=0)
else:
    combined = np.concatenate(strips, axis=0)
bg_color = np.median(combined, axis=0)
print(f"Background color (BGR): {bg_color}")

# Color distance
smooth_f = smooth.astype(np.float32)
diff = smooth_f - bg_color.reshape(1, 1, 3)
color_dist = np.sqrt(np.sum(diff**2, axis=2))
color_dist_norm = (color_dist / (color_dist.max() + 1e-6) * 255).astype(np.uint8)
_, mask = cv2.threshold(color_dist_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest = max(contours, key=cv2.contourArea)
(mcx, mcy), mradius = cv2.minEnclosingCircle(largest)

M = cv2.moments(largest)
cent_x = M["m10"] / M["m00"]
cent_y = M["m01"] / M["m00"]

print(f"\nColor method:")
print(f"  Enclosing circle: center=({mcx:.0f},{mcy:.0f}), r={mradius:.0f}")
print(f"  Centroid: ({cent_x:.0f},{cent_y:.0f})")
print(f"  Drift: ({mcx-cent_x:.0f},{mcy-cent_y:.0f})")

# How the mask looks — check where it extends
# Find top, bottom, left, right extents of mask
mask_pts = np.where(mask > 0)
if len(mask_pts[0]) > 0:
    top_y = mask_pts[0].min()
    bot_y = mask_pts[0].max()
    left_x = mask_pts[1].min()
    right_x = mask_pts[1].max()
    print(f"  Mask extents: top={top_y}, bottom={bot_y}, left={left_x}, right={right_x}")
    print(f"  Mask size: {right_x-left_x}w x {bot_y-top_y}h")
    
    # Where SHOULD the center be? (midpoint of extents)
    ideal_cx = (left_x + right_x) / 2
    ideal_cy = (top_y + bot_y) / 2
    print(f"  Mask midpoint: ({ideal_cx:.0f},{ideal_cy:.0f})")
    print(f"  Center vs midpoint: ({mcx-ideal_cx:.0f},{mcy-ideal_cy:.0f})")

# Hough method
gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
grad = np.sqrt(sobelx**2 + sobely**2)
grad_norm = (grad / (grad.max() + 1e-6) * 255).astype(np.uint8)

min_r = int(min(sh, sw) * 0.15)
max_r = int(min(sh, sw) * 0.55)
best = None
best_score = -1
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
                cx, cy, r = float(c[0]), float(c[1]), float(c[2])
                dx = (cx - sw/2) / (sw/2)
                dy = (cy - sh/2) / (sh/2)
                if best is None or math.sqrt(dx**2 + dy**2) < math.sqrt(((best[0]-sw/2)/(sw/2))**2 + ((best[1]-sh/2)/(sh/2))**2):
                    best = (cx, cy, r)

if best:
    print(f"\nHough best candidate: center=({best[0]:.0f},{best[1]:.0f}), r={best[2]:.0f}")
    print(f"  Image center: ({sw/2:.0f},{sh/2:.0f})")
    print(f"  Hough offset from image center: ({best[0]-sw/2:.0f},{best[1]-sh/2:.0f})")
    
# Current blended result
hcx, hcy = best[0], best[1]
ccx, ccy, cr = mcx, mcy, mradius
bx = ccx * 0.7 + hcx * 0.3
by = ccy * 0.7 + hcy * 0.3
print(f"\nBlended (70/30): center=({bx:.0f},{by:.0f}), r={cr:.0f}")
print(f"  Blended offset from image center: ({bx-sw/2:.0f},{by-sw/2:.0f})")

# What if we use the mask midpoint instead of the enclosing circle center?
if len(mask_pts[0]) > 0:
    mid_cx = (left_x + right_x) / 2
    mid_cy = (top_y + bot_y) / 2
    mid_r = max(right_x-left_x, bot_y-top_y) / 2
    print(f"\nMask midpoint approach: center=({mid_cx:.0f},{mid_cy:.0f}), r={mid_r:.0f}")
    
    # Blend mask midpoint with Hough
    bx2 = mid_cx * 0.7 + hcx * 0.3
    by2 = mid_cy * 0.7 + hcy * 0.3
    print(f"Blended midpoint (70/30): center=({bx2:.0f},{by2:.0f}), r={mid_r:.0f}")
