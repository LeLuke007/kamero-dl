import cv2
import numpy as np


def compute_detection_alpha_map(bgr_img, ksize):
    """Compute a high-contrast normalized alpha map for finding 'TIET' locations."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    img_f = bgr_img.astype(np.float32)

    alpha_channels = []
    for c in range(3):
        ch = img_f[:, :, c]
        bg = cv2.morphologyEx(ch, cv2.MORPH_OPEN, kernel)
        bg = cv2.GaussianBlur(bg, (3, 3), 0)
        diff = np.maximum(ch - bg, 0.0)
        denom = np.maximum(255.0 - bg, 25.0)
        alpha_channels.append(diff / denom)

    alpha_min = np.minimum(np.minimum(alpha_channels[0], alpha_channels[1]), alpha_channels[2])
    alpha_med = np.median(np.stack(alpha_channels, axis=-1), axis=-1)
    return np.clip(0.65 * alpha_min + 0.35 * alpha_med, 0.0, 1.0).astype(np.float32)


def synthesize_seed_template(scale, angle_deg):
    """Create a synthetic rotated 'TIET' template for initial scale/angle discovery."""
    size = int(round(140 * scale))
    size = max(48, size + (size % 2))
    canvas = np.zeros((size, size), dtype=np.float32)

    font_scale = 0.80 * scale
    thickness = max(1, int(round(1.8 * scale)))
    (tw, th), _ = cv2.getTextSize("TIET", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx = (size - tw) // 2
    ty = (size + th) // 2

    cv2.putText(
        canvas, "TIET", (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1.0, thickness, cv2.LINE_AA
    )

    M = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), angle_deg, 1.0)
    rotated = cv2.warpAffine(canvas, M, (size, size), flags=cv2.INTER_LINEAR)

    ys, xs = np.where(rotated > 0.05)
    if len(xs) == 0:
        return rotated
    pad = max(4, int(round(5 * scale)))
    x1, x2 = max(0, xs.min() - pad), min(size, xs.max() + pad + 1)
    y1, y2 = max(0, ys.min() - pad), min(size, ys.max() + pad + 1)
    return rotated[y1:y2, x1:x2]


def find_best_seed_and_corr(alpha_map, img_w, img_h):
    """Find the scale and rotation angle of 'TIET' using variance-gated NMS peak scoring."""
    local_mean = cv2.blur(alpha_map, (31, 31))
    local_sq = cv2.blur(alpha_map * alpha_map, (31, 31))
    local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 0.0))

    ref_scale = (img_w / 720.0) * 0.62
    best_score = -1.0
    best_tmpl = None
    best_corr = None
    nms_r = max(25, int(min(img_h, img_w) * 0.08)) | 1
    nms_kernel = np.ones((nms_r, nms_r), np.uint8)

    for scale_mult in np.linspace(0.78, 1.25, 9):
        for angle in (26.5, 28.0, 29.5, 31.0):
            tmpl = synthesize_seed_template(ref_scale * scale_mult, angle)
            th, tw = tmpl.shape[:2]
            if th >= img_h // 2 or tw >= img_w // 2 or th < 8 or tw < 12:
                continue

            corr = cv2.matchTemplate(alpha_map, tmpl, cv2.TM_CCOEFF_NORMED)
            std_crop = local_std[th // 2 : th // 2 + corr.shape[0], tw // 2 : tw // 2 + corr.shape[1]]
            corr[(std_crop < 0.012) | (corr > 0.98) | ~np.isfinite(corr)] = 0.0

            dil = cv2.dilate(corr, nms_kernel)
            pks = np.sort(corr[(corr == dil) & (corr > 0.22)])
            if len(pks) >= 6:
                score = float(np.mean(pks[-min(15, len(pks)):]))
            else:
                flat_top = np.partition(corr.ravel(), -25)[-25:]
                score = float(np.mean(flat_top)) * 0.5

            if score > best_score:
                best_score = score
                best_tmpl = tmpl
                best_corr = corr

    return best_tmpl, best_corr


def subpixel_peak_offset(corr_patch):
    """Fit a 2D parabola to a 3x3 correlation neighborhood for sub-pixel precision."""
    if corr_patch.shape != (3, 3):
        return 0.0, 0.0
    dx_num = float(corr_patch[1, 2]) - float(corr_patch[1, 0])
    dx_den = 2.0 * (2.0 * float(corr_patch[1, 1]) - float(corr_patch[1, 2]) - float(corr_patch[1, 0]))
    dy_num = float(corr_patch[2, 1]) - float(corr_patch[0, 1])
    dy_den = 2.0 * (2.0 * float(corr_patch[1, 1]) - float(corr_patch[2, 1]) - float(corr_patch[0, 1]))

    dx = dx_num / dx_den if abs(dx_den) > 1e-5 else 0.0
    dy = dy_num / dy_den if abs(dy_den) > 1e-5 else 0.0
    return float(np.clip(dx, -0.6, 0.6)), float(np.clip(dy, -0.6, 0.6))


def gauss_reduce_basis(v1, v2):
    """2D Gauss lattice basis reduction to obtain the two shortest orthogonal basis vectors."""
    v1 = v1.astype(np.float64).copy()
    v2 = v2.astype(np.float64).copy()
    for _ in range(16):
        if np.linalg.norm(v2) < np.linalg.norm(v1):
            v1, v2 = v2, v1
        mu = round(float(np.dot(v1, v2)) / (float(np.dot(v1, v1)) + 1e-8))
        if mu == 0:
            break
        v2 = v2 - mu * v1
    return v1, v2


def fit_lattice_ransac_lsq(corr_map, orig_w, orig_h):
    """
    Find all strong correlation peaks across the image and solve for the exact
    2D periodic lattice (p0, v1, v2) via RANSAC + Global Linear Least Squares + Gauss reduction.
    Uses pseudoinverse and 2D-span rank checks to guarantee zero Singular Matrix errors.
    """
    ch, cw = corr_map.shape
    nms_radius = max(15, int(min(orig_w, orig_h) * 0.05)) | 1
    dilated = cv2.dilate(corr_map, np.ones((nms_radius, nms_radius), np.uint8))
    thresh = max(0.20, float(np.percentile(corr_map, 98.5) * 0.55))
    py, px = np.where((corr_map == dilated) & (corr_map >= thresh))

    peaks = []
    for y, x in zip(py, px):
        if 1 <= x < cw - 1 and 1 <= y < ch - 1:
            dx, dy = subpixel_peak_offset(corr_map[y - 1:y + 2, x - 1:x + 2])
            peaks.append((float(x) + dx, float(y) + dy, float(corr_map[y, x])))
    peaks.sort(key=lambda t: t[2], reverse=True)
    peaks = peaks[:85]

    L_default = orig_w / 6.0
    rad30 = np.deg2rad(30.0)
    v1_def = np.array([L_default * np.sin(rad30), L_default * np.cos(rad30)], dtype=np.float64)
    v2_def = np.array([-L_default * np.cos(rad30), L_default * np.sin(rad30)], dtype=np.float64)

    if len(peaks) < 6:
        p0_fallback = np.array([peaks[0][0], peaks[0][1]]) if len(peaks) else np.array([cw / 2.0, ch / 2.0])
        return p0_fallback, v1_def, v2_def

    pts = np.array([[p[0], p[1]] for p in peaks], dtype=np.float64)
    weights = np.array([p[2] for p in peaks], dtype=np.float64)

    cand_bases = [(v1_def, v2_def)]
    L_alt = orig_h / 6.0
    cand_bases.append((
        np.array([L_alt * np.sin(rad30), L_alt * np.cos(rad30)], dtype=np.float64),
        np.array([-L_alt * np.cos(rad30), L_alt * np.sin(rad30)], dtype=np.float64)
    ))

    min_d = min(orig_w, orig_h) * 0.08
    max_d = max(orig_w, orig_h) * 0.35
    top_n = min(20, len(pts))
    for i in range(top_n):
        for j in range(i + 1, top_n):
            dv = pts[j] - pts[i]
            d = np.hypot(dv[0], dv[1])
            if min_d <= d <= max_d:
                v_ortho = np.array([-dv[1], dv[0]], dtype=np.float64)
                cand_bases.append((dv, v_ortho))

    best_score = -1.0
    best_p0 = pts[0].copy()
    best_v1 = v1_def.copy()
    best_v2 = v2_def.copy()

    for cv1, cv2_vec in cand_bases:
        B_mat = np.column_stack([cv1, cv2_vec])
        if abs(np.linalg.det(B_mat)) < 100.0:
            continue
        B_inv = np.linalg.pinv(B_mat)
        for idx in range(min(15, len(pts))):
            p_ref = pts[idx]
            rel = pts - p_ref
            mn_float = rel @ B_inv.T
            mn_int = np.round(mn_float)
            recon = p_ref + mn_int @ B_mat.T
            err = np.linalg.norm(pts - recon, axis=1)
            inliers = np.where(err < 8.5)[0]
            if len(inliers) >= 5:
                if np.ptp(mn_int[inliers, 0]) >= 1 and np.ptp(mn_int[inliers, 1]) >= 1:
                    sc = float(np.sum(weights[inliers]))
                    if sc > best_score:
                        best_score = sc
                        best_p0 = p_ref.copy()
                        best_v1 = cv1.copy()
                        best_v2 = cv2_vec.copy()

    p0 = best_p0
    v1, v2 = gauss_reduce_basis(best_v1, best_v2)

    for tol in (8.0, 5.0, 3.0, 1.5, 1.0):
        B_mat = np.column_stack([v1, v2])
        if abs(np.linalg.det(B_mat)) < 100.0:
            break
        B_inv = np.linalg.pinv(B_mat)
        rel = pts - p0
        mn_float = rel @ B_inv.T
        mn_int = np.round(mn_float)
        recon = p0 + mn_int @ B_mat.T
        err = np.linalg.norm(pts - recon, axis=1)
        inliers = np.where(err < tol)[0]
        if len(inliers) < 6:
            break
        if np.ptp(mn_int[inliers, 0]) < 1 or np.ptp(mn_int[inliers, 1]) < 1:
            break

        A = np.column_stack([mn_int[inliers, 0], mn_int[inliers, 1], np.ones(len(inliers))])
        if np.linalg.matrix_rank(A) < 3:
            break
        W = np.sqrt(weights[inliers])[:, None]
        sol, _, _, _ = np.linalg.lstsq(A * W, pts[inliers] * W, rcond=None)
        new_v1, new_v2, new_p0 = sol[0], sol[1], sol[2]
        new_v1, new_v2 = gauss_reduce_basis(new_v1, new_v2)
        if abs(np.linalg.det(np.column_stack([new_v1, new_v2]))) < 100.0:
            break
        v1, v2, p0 = new_v1, new_v2, new_p0

    return p0, v1, v2


def warp_lanczos(matte, dx, dy):
    """Shift a 2D matte by sub-pixel offset (dx, dy) using Lanczos-4 interpolation."""
    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return matte
    h, w = matte.shape[:2]
    M = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    return cv2.warpAffine(
        matte, M, (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0
    )


def enumerate_lattice_stamps(p0, v1, v2, tw, th, orig_w, orig_h):
    """Generate all lattice positions overlapping [0:orig_w, 0:orig_h] with sub-pixel offsets."""
    stamps = []
    for m in range(-25, 26):
        for n in range(-25, 26):
            gx = float(p0[0] + m * v1[0] + n * v2[0])
            gy = float(p0[1] + m * v1[1] + n * v2[1])
            if gx + tw <= 0 or gx >= orig_w or gy + th <= 0 or gy >= orig_h:
                continue
            ix, iy = int(round(gx)), int(round(gy))
            dx_sub, dy_sub = gx - ix, gy - iy
            stamps.append((ix, iy, dx_sub, dy_sub))
    return stamps


def fill_small_holes(bin_mask, max_hole_area):
    """Fill enclosed interior pinholes (e.g. inside 'E') without bridging gaps between letters."""
    inv = (bin_mask == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    out = bin_mask.copy()
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] <= max_hole_area:
            out[labels == lbl] = 255
    return out


def remove_tiet_watermark_auto(bgr_img):
    """
    High-precision automatic TIET watermark removal:
    1. Multi-scale seed template discovery & singularity-free RANSAC + Least-Squares 2D lattice fit.
    2. Unclipped true-font discovery from padded morphological alpha patches.
    3. Sub-pixel refined lattice fit (<0.02px accuracy) with Gauss-reduced basis vectors.
    4. Unbiased stroke-inpainted background & Lanczos-4 true alpha matte extraction.
    5. 4:2:0 Chroma-aware inverse alpha unblending (Cw = 251.2).
    6. First-order shading-guided Bilateral Onion-Peel + True-Edge Preservation + Two-Scale Micro-Texture.
    """
    orig_h, orig_w = bgr_img.shape[:2]
    ksize = max(19, int(round(orig_w * 0.016)) | 1)

    alpha_det = compute_detection_alpha_map(bgr_img, ksize)
    seed_tmpl, seed_corr = find_best_seed_and_corr(alpha_det, orig_w, orig_h)
    th_s, tw_s = seed_tmpl.shape[:2]
    p0_init, v1_init, v2_init = fit_lattice_ransac_lsq(seed_corr, orig_w, orig_h)
    init_stamps = enumerate_lattice_stamps(p0_init, v1_init, v2_init, tw_s, th_s, orig_w, orig_h)

    pad = max(12, int(round(max(th_s, tw_s) * 0.22)))
    patches_det = []
    for (ix, iy, dx_sub, dy_sub) in init_stamps:
        if pad <= ix and ix + tw_s + pad <= orig_w and pad <= iy and iy + th_s + pad <= orig_h:
            p = alpha_det[iy - pad:iy + th_s + pad, ix - pad:ix + tw_s + pad]
            patches_det.append(warp_lanczos(p, -dx_sub, -dy_sub))

    if len(patches_det) >= 6:
        raw_med = np.median(np.stack(patches_det, axis=0), axis=0)
        baseline = np.percentile(raw_med, 30)
        init_matte = np.maximum(raw_med - baseline, 0.0)
        mask_cc = (init_matte > (init_matte.max() * 0.08)).astype(np.uint8)
        mask_cc = cv2.dilate(mask_cc, np.ones((5, 5), np.uint8), iterations=1)
        init_matte = init_matte * mask_cc

        ys, xs = np.where(mask_cc > 0)
        if len(xs) > 0:
            y1_m, y2_m = max(0, ys.min() - 4), min(init_matte.shape[0], ys.max() + 5)
            x1_m, x2_m = max(0, xs.min() - 4), min(init_matte.shape[1], xs.max() + 5)
            matte_1 = init_matte[y1_m:y2_m, x1_m:x2_m]
        else:
            matte_1 = seed_tmpl
    else:
        matte_1 = seed_tmpl

    matte_1_unit = matte_1 / (matte_1.max() + 1e-6)
    th, tw = matte_1_unit.shape[:2]
    scale_factor = max(1.0, max(tw, th) / 73.0)

    local_mean = cv2.blur(alpha_det, (31, 31))
    local_sq = cv2.blur(alpha_det * alpha_det, (31, 31))
    local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 0.0))

    corr_refined = cv2.matchTemplate(alpha_det, matte_1_unit, cv2.TM_CCOEFF_NORMED)
    std_crop = local_std[th // 2 : th // 2 + corr_refined.shape[0], tw // 2 : tw // 2 + corr_refined.shape[1]]
    corr_refined[(std_crop < 0.012) | ~np.isfinite(corr_refined)] = 0.0

    p0, v1, v2 = fit_lattice_ransac_lsq(corr_refined, orig_w, orig_h)
    stamps = enumerate_lattice_stamps(p0, v1, v2, tw, th, orig_w, orig_h)

    max_pinhole = int(round(22 * (scale_factor ** 2)))
    m_bin = ((matte_1_unit > 0.03) * 255).astype(np.uint8)
    m_bin = cv2.dilate(m_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    m_bin = fill_small_holes(m_bin, max_pinhole)

    stroke_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    for (ix, iy, dx_sub, dy_sub) in stamps:
        x1, y1 = max(0, ix), max(0, iy)
        x2, y2 = min(orig_w, ix + tw), min(orig_h, iy + th)
        if x2 > x1 and y2 > y1:
            tx1, ty1 = x1 - ix, y1 - iy
            tx2, ty2 = tx1 + (x2 - x1), ty1 + (y2 - y1)
            stroke_mask[y1:y2, x1:x2] = np.maximum(stroke_mask[y1:y2, x1:x2], m_bin[ty1:ty2, tx1:tx2])

    inp_r = max(4, int(round(4 * scale_factor)))
    bg_unbiased = cv2.inpaint(bgr_img, stroke_mask, inpaintRadius=inp_r, flags=cv2.INPAINT_TELEA).astype(np.float32)
    img_f = bgr_img.astype(np.float32)

    Cw = 251.2
    true_alpha_full = np.median(
        np.clip((img_f - bg_unbiased) / np.maximum(Cw - bg_unbiased, 15.0), -0.25, 1.0),
        axis=-1
    ).astype(np.float32)

    outside = m_bin == 0
    cand_patches = []
    for (ix, iy, dx_sub, dy_sub) in stamps:
        if 0 <= ix and ix + tw <= orig_w and 0 <= iy and iy + th <= orig_h:
            patch_bg = bg_unbiased[iy:iy + th, ix:ix + tw]
            bg_mean = float(np.mean(patch_bg[outside]))
            bg_spatial_std = float(np.mean(np.std(patch_bg[outside], axis=0)))
            if bg_mean < 228.0:
                patch = true_alpha_full[iy:iy + th, ix:ix + tw]
                cand_patches.append((bg_spatial_std, warp_lanczos(patch, -dx_sub, -dy_sub)))

    cand_patches.sort(key=lambda t: t[0])
    n_keep = max(8, min(len(cand_patches), int(round(len(cand_patches) * 0.70))))
    aligned_patches = [t[1] for t in cand_patches[:n_keep]]

    gate = (m_bin > 0).astype(np.float32)
    if len(aligned_patches) >= 6:
        raw_matte = np.median(np.stack(aligned_patches, axis=0), axis=0)
        outside_level = np.median(raw_matte[outside]) if np.any(outside) else 0.0
        true_matte = np.clip((raw_matte - outside_level) * gate, 0.0, 0.75).astype(np.float32)
    else:
        true_matte = (matte_1_unit * 0.60).astype(np.float32)

    result_f = img_f.copy()
    full_footprint_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    c_med_k = 5 if scale_factor > 1.3 else 3

    for (ix, iy, dx_sub, dy_sub) in stamps:
        warped_matte = warp_lanczos(true_matte, dx_sub, dy_sub)
        x1, y1 = max(0, ix), max(0, iy)
        x2, y2 = min(orig_w, ix + tw), min(orig_h, iy + th)
        if x2 <= x1 or y2 <= y1:
            continue
        tx1, ty1 = x1 - ix, y1 - iy
        tx2, ty2 = tx1 + (x2 - x1), ty1 + (y2 - y1)

        sub_matte = warped_matte[ty1:ty2, tx1:tx2]
        if not np.any(sub_matte > 0.005):
            continue

        patch_img = result_f[y1:y2, x1:x2]
        alpha_luma = np.clip(sub_matte, 0.0, 0.75)
        alpha_chroma = np.clip(cv2.GaussianBlur(sub_matte, (5, 5), 1.2), 0.0, 0.75)

        luma_obs = 0.114 * patch_img[:, :, 0] + 0.587 * patch_img[:, :, 1] + 0.299 * patch_img[:, :, 2]
        chroma_obs = patch_img - luma_obs[:, :, None]

        luma_unb = (luma_obs - alpha_luma * Cw) / np.maximum(1.0 - alpha_luma, 0.25)
        chroma_unb = chroma_obs / np.maximum(1.0 - alpha_chroma[:, :, None], 0.25)
        chroma_unb_med = cv2.medianBlur(
            np.clip(chroma_unb + 128.0, 0, 255).astype(np.uint8), c_med_k
        ).astype(np.float32) - 128.0

        unblended = luma_unb[:, :, None] + chroma_unb_med
        result_f[y1:y2, x1:x2] = np.clip(unblended, 0.0, 255.0)

        footprint = ((sub_matte > (true_matte.max() * 0.02)) * 255).astype(np.uint8)
        footprint = cv2.dilate(footprint, kernel_dilate, iterations=1)
        footprint = fill_small_holes(footprint, max_pinhole)
        footprint = cv2.dilate(footprint, kernel_dilate, iterations=1)
        full_footprint_mask[y1:y2, x1:x2] = np.maximum(full_footprint_mask[y1:y2, x1:x2], footprint)

    R = 4
    sigma_s = 2.0
    sigma_r = 16.0

    pad_result = cv2.copyMakeBorder(result_f, R, R, R, R, cv2.BORDER_REFLECT_101)
    pad_mask = cv2.copyMakeBorder(full_footprint_mask, R, R, R, R, cv2.BORDER_CONSTANT, value=0)
    cleaned_pad = pad_result.copy()

    offsets = []
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            if dx == 0 and dy == 0:
                continue
            d2 = dx * dx + dy * dy
            if d2 <= R * R:
                ws = np.exp(-d2 / (2.0 * sigma_s * sigma_s))
                offsets.append((dy, dx, ws))

    kernel_erode = np.ones((3, 3), np.uint8)
    rng = np.random.default_rng(42)
    med_k = 5 if scale_factor > 1.3 else 3

    for (ix, iy, dx_sub, dy_sub) in stamps:
        ox1, oy1 = max(0, ix - 4), max(0, iy - 4)
        ox2, oy2 = min(orig_w, ix + tw + 4), min(orig_h, iy + th + 4)
        if ox2 <= ox1 or oy2 <= oy1:
            continue

        px1, py1 = ox1, oy1
        px2, py2 = ox2 + 2 * R, oy2 + 2 * R

        work_img = cleaned_pad[py1:py2, px1:px2].copy()
        orig_local_mask = (pad_mask[py1:py2, px1:px2] > 0).astype(np.uint8)
        if not np.any(orig_local_mask):
            continue

        bh, bw = work_img.shape[:2]
        center_unb_orig = work_img[R:bh - R, R:bw - R].copy()

        guide_med_u8 = cv2.medianBlur(np.clip(np.round(work_img), 0, 255).astype(np.uint8), med_k)
        guide_med = guide_med_u8.astype(np.float32)
        guide = cv2.bilateralFilter(guide_med, d=5, sigmaColor=22.0, sigmaSpace=3.0)

        luma_g = 0.114 * guide[:, :, 0] + 0.587 * guide[:, :, 1] + 0.299 * guide[:, :, 2]
        gx = cv2.Sobel(luma_g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(luma_g, cv2.CV_32F, 0, 1, ksize=3)
        grad_guide = cv2.dilate(np.hypot(gx, gy), np.ones((3, 3), np.uint8))

        work_clamped = np.clip(work_img, guide_med - 16.0, guide_med + 16.0)
        edge_anchor = (grad_guide > 110.0).astype(np.uint8)
        work_img[edge_anchor > 0] = work_clamped[edge_anchor > 0]
        rem_mask = orig_local_mask & (1 - edge_anchor)

        center_grad = grad_guide[R:bh - R, R:bw - R]
        beta_shade = np.clip((center_grad - 22.0) / 28.0, 0.0, 0.85)[:, :, None]

        for _ in range(64):
            center_rem = rem_mask[R:bh - R, R:bw - R]
            if not np.any(center_rem):
                break
            eroded = cv2.erode(rem_mask, kernel_erode, iterations=1)
            shell = (rem_mask > 0) & (eroded == 0)
            center_shell = shell[R:bh - R, R:bw - R]
            if not np.any(center_shell):
                center_shell = center_rem > 0
                eroded[:] = 0

            clean_weight = (rem_mask == 0).astype(np.float32)
            center_guide = guide[R:bh - R, R:bw - R]
            center_unb = work_img[R:bh - R, R:bw - R]

            c_acc_color = center_unb * 1e-4
            c_acc_weight = np.full((bh - 2 * R, bw - 2 * R), 1e-4, dtype=np.float32)

            for dy, dx, ws in offsets:
                ny1, ny2 = R + dy, bh - R + dy
                nx1, nx2 = R + dx, bw - R + dx
                n_clean = clean_weight[ny1:ny2, nx1:nx2]
                n_color = work_img[ny1:ny2, nx1:nx2]
                n_guide = guide[ny1:ny2, nx1:nx2]
                diff2 = np.sum((center_guide - n_guide) ** 2, axis=-1) / 3.0
                wr = np.exp(-diff2 / (2.0 * sigma_r * sigma_r))
                w_tot = ws * n_clean * wr
                n_transport = n_color + beta_shade * (center_guide - n_guide)
                c_acc_color += w_tot[:, :, None] * n_transport
                c_acc_weight += w_tot

            recon = np.clip(c_acc_color / c_acc_weight[:, :, None], 0.0, 255.0)
            center_work = work_img[R:bh - R, R:bw - R]
            center_work[center_shell] = recon[center_shell]
            rem_mask = eroded

        center_orig_mask = orig_local_mask[R:bh - R, R:bw - R] > 0
        c_acc_color2 = work_img[R:bh - R, R:bw - R].copy()
        c_acc_weight2 = np.ones((bh - 2 * R, bw - 2 * R), dtype=np.float32)
        center_guide2 = guide[R:bh - R, R:bw - R]
        for dy, dx, ws in offsets:
            ny1, ny2 = R + dy, bh - R + dy
            nx1, nx2 = R + dx, bw - R + dx
            n_color = work_img[ny1:ny2, nx1:nx2]
            n_guide2 = guide[ny1:ny2, nx1:nx2]
            diff2 = np.sum((center_guide2 - n_guide2) ** 2, axis=-1) / 3.0
            wr = np.exp(-diff2 / (2.0 * 18.0 * 18.0))
            w_tot = ws * wr
            c_acc_color2 += w_tot[:, :, None] * n_color
            c_acc_weight2 += w_tot
        polished_work = c_acc_color2 / c_acc_weight2[:, :, None]

        clean_local = orig_local_mask[R:bh - R, R:bw - R] == 0
        if np.any(clean_local):
            blur_clean = cv2.GaussianBlur(center_unb_orig, (5, 5), 1.0)
            hf = np.abs(center_unb_orig - blur_clean)
            grain_std = float(np.clip(np.median(hf[clean_local]) * 1.20, 0.35, 2.4))
            n1 = rng.normal(0.0, grain_std * 0.70, size=(bh - 2 * R, bw - 2 * R, 1)).astype(np.float32)
            n2_raw = rng.normal(0.0, grain_std * 1.05, size=(bh - 2 * R, bw - 2 * R, 1)).astype(np.float32)
            n2 = cv2.GaussianBlur(n2_raw, (3, 3), 0.65)[:, :, None]
            polished_work = np.clip(polished_work + n1 + n2, 0.0, 255.0)

        w_edge = np.clip((center_grad - 95.0) / 55.0, 0.0, 1.0)
        w_edge = cv2.GaussianBlur(w_edge, (3, 3), 0)[:, :, None]
        center_clamped = work_clamped[R:bh - R, R:bw - R]
        blended = polished_work * (1.0 - w_edge) + center_clamped * w_edge

        cleaned_pad[py1 + R:py2 - R, px1 + R:px2 - R][center_orig_mask] = blended[center_orig_mask]

    cleaned_f = cleaned_pad[R:R + orig_h, R:R + orig_w]
    return np.clip(np.round(cleaned_f), 0, 255).astype(np.uint8)
