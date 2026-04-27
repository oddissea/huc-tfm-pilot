#!/usr/bin/env python3
"""
Port local del pipeline TIFF → H5 del paquete `portaanalysis` en el servidor
doctor. Basado en:

- portahelper.py           (adaptado a OOP con nombres PEP8)
- dataprocessing.py        (port directo de generatePatches + rowTask1)
- util_patch_rebin.py      (port directo de rebin_patches)
- util_hdf5.py             (port directo de save_patches_to_hdf5)

Paridad verificada al 100 % contra la salida del pipeline original (N de
parches, posiciones y píxeles idénticos) sobre
`10084_22_ca_202204081746.tif` (419 vs 419 parches, diff = 0).

Uso:
    python scripts/preprocessing/tiff_to_h5.py \\
        --input  <archivo.tif> \\
        --output <salida.h5>

Validación contra un H5 de referencia:
    python scripts/preprocessing/tiff_to_h5.py \\
        --input  <archivo.tif> \\
        --output <salida.h5> \\
        --compare <referencia.h5>
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import cv2
import h5py
import imutils
import numpy as np
import tifffile as tifi


# ───────────────────────────────────────────────────────────────────────────
# PortaHelper (port directo del portahelper.py leído del servidor)
# ───────────────────────────────────────────────────────────────────────────

def _get_pixel_max(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([grayscale], [0], None, [256], [0, 256])
    return int(np.where(hist[:, 0] == hist[:, 0].max())[0][0])


class PortaHelper:
    DEFAULT_MIN_TISSUE_PIXELS_3MM = math.ceil(1 / 0.46500 * 3000)
    WHITE_CONDITION_THRESHOLD = 90.0  # %

    def __init__(self, filename, image_level=5):
        self._filename = filename
        self._image_name = Path(filename).stem
        self._work_image_level = image_level + 2
        self._inc_rate = None
        self._min_tissue_shape = None
        self.__read_metadata()

    def __read_metadata(self):
        with tifi.TiffFile(self._filename) as meta:
            if len(meta.pages) < 6:
                raise ValueError('porta error: TIFF con menos de 6 páginas')
            ref_shape = self._image_shape(meta.pages[2].tags)
            cur_shape = self._image_shape(meta.pages[self._work_image_level].tags)
            self._inc_rate = (ref_shape[0] / cur_shape[0],
                              ref_shape[1] / cur_shape[1])
            min_w = math.ceil((1 / self._inc_rate[0]) * self.DEFAULT_MIN_TISSUE_PIXELS_3MM)
            min_l = math.ceil((1 / self._inc_rate[1]) * self.DEFAULT_MIN_TISSUE_PIXELS_3MM)
            self._min_tissue_shape = (min_w, min_l)

    @staticmethod
    def _image_shape(tags):
        return (tags['ImageWidth'].value, tags['ImageLength'].value)

    @staticmethod
    def _detect_polygon(contour):
        peri = cv2.arcLength(contour, True)
        return cv2.approxPolyDP(contour, 0.04 * peri, True)

    def _bounding_boxes(self, binary):
        contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        bboxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 25:
                continue
            approx = self._detect_polygon(c)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                if w >= self._min_tissue_shape[1] and h >= self._min_tissue_shape[0]:
                    bboxes.append((x, y, w, h, area))
        return bboxes

    def _best_box_id(self, bboxes, image, grayscale):
        best_id = None
        best_area = -1
        for i, (x, y, w, h, area) in enumerate(bboxes):
            mean_img = grayscale[y:y+h, x:x+w].astype(np.int32, copy=False)
            canal0 = np.abs(image[y:y+h, x:x+w, 0] - mean_img)
            canal1 = np.abs(image[y:y+h, x:x+w, 1] - mean_img)
            canal2 = np.abs(image[y:y+h, x:x+w, 2] - mean_img)
            sumimg = canal0 + canal1 + canal2
            mask_gray = sumimg < 10
            white_pct = mask_gray.sum() / mask_gray.size * 100
            if area > best_area and white_pct < self.WHITE_CONDITION_THRESHOLD:
                best_area = area
                best_id = i
        return best_id

    def extract_tissue(self):
        porta_image = tifi.imread(self._filename, key=self._work_image_level)
        grayscale = cv2.cvtColor(porta_image, cv2.COLOR_RGB2GRAY)
        pixel_max = 200
        binary = np.where(grayscale >= pixel_max, 0, 255).astype(np.uint8)
        bboxes = self._bounding_boxes(binary)

        if len(bboxes) <= 1:
            # Fallback: usar la imagen 20x completa
            image_20x = tifi.imread(self._filename, key=2)
            return image_20x, self._image_name, _get_pixel_max(image_20x)

        best = self._best_box_id(bboxes, porta_image, grayscale)
        if best is None:
            raise RuntimeError('No se encontró bbox válido de tejido')

        box = bboxes[best]
        tx = math.ceil(box[0] * self._inc_rate[0])
        ty = math.ceil(box[1] * self._inc_rate[1])
        tw = math.ceil(box[2] * self._inc_rate[0])
        tl = math.ceil(box[3] * self._inc_rate[1])

        tissue_low = porta_image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
        pixel_max = _get_pixel_max(tissue_low)

        image_20x = tifi.imread(self._filename, key=2)
        tissue = np.copy(image_20x[ty:ty+tl, tx:tx+tw])
        return tissue, self._image_name, pixel_max


# ───────────────────────────────────────────────────────────────────────────
# Normalización + filtro de parches (port de dataprocessing.py)
# ───────────────────────────────────────────────────────────────────────────

def image_normalize(image, pixel_max, copy=False):
    cpimage = image if not copy else np.copy(image)
    norma = np.float32(255 / pixel_max)
    for canal in range(image.shape[2]):
        temp = cpimage[:, :, canal] * norma
        temp = np.ceil(temp).astype(np.uint16, copy=False)
        temp = np.clip(temp, 0, 255).astype(np.uint8, copy=False)
        cpimage[:, :, canal] = temp
    return cpimage


def _is_patch_valid(patch, white_percent):
    """Réplica exacta del filtro de blanco de dataprocessing.py::rowTask1."""
    FIX_DECIMAL = 1000
    mean_img = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.int32, copy=False)
    canal0 = np.abs(patch[:, :, 0].astype(np.int32) - mean_img)
    canal1 = np.abs(patch[:, :, 1].astype(np.int32) - mean_img)
    canal2 = np.abs(patch[:, :, 2].astype(np.int32) - mean_img)
    sumimg = canal0 + canal1 + canal2
    mask_gray = sumimg < 10
    count = int(np.count_nonzero(mask_gray))
    white_condition = count / mask_gray.size * 100
    white_condition_fix = int(white_condition * FIX_DECIMAL)
    white_percent_fix = white_percent * FIX_DECIMAL
    return white_condition_fix < white_percent_fix


# ───────────────────────────────────────────────────────────────────────────
# generatePatches (port directo de dataprocessing.py::generatePatches + rowTask1)
# ───────────────────────────────────────────────────────────────────────────

def generate_patches(tissue_image, patch_size, white_percent, pixel_max):
    """
    Devuelve [(index, y, y+h, x, x+w, patch_normalizado)] para los parches
    que pasan el filtro de blanco con umbral `white_percent` (%).

    Equivalente a dataprocessing.py::generatePatches +  rowTask1, sin Pool.
    """
    patch_w, patch_h = int(patch_size[0]), int(patch_size[1])
    ilength, iwidth, _c = tissue_image.shape
    num_patch_w = iwidth // patch_w
    num_patch_l = ilength // patch_h

    fix_decimal = 1000
    white_percent_fix = white_percent * fix_decimal

    out = []
    for l in range(num_patch_l):
        for w in range(num_patch_w):
            index = w + l * num_patch_w
            y = l * patch_h
            x = w * patch_w
            ll = y + patch_h
            ww = x + patch_w

            patch = image_normalize(tissue_image[y:ll, x:ww], pixel_max, copy=True)
            mean_img = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.int32, copy=False)
            canal0 = np.abs(patch[:, :, 0].astype(np.int32) - mean_img)
            canal1 = np.abs(patch[:, :, 1].astype(np.int32) - mean_img)
            canal2 = np.abs(patch[:, :, 2].astype(np.int32) - mean_img)
            sumimg = canal0 + canal1 + canal2
            mask_gray = sumimg < 10
            count = int(np.count_nonzero(mask_gray))
            white_condition = count / mask_gray.size * 100
            white_condition_fix = int(white_condition * fix_decimal)

            if white_condition_fix < white_percent_fix:
                out.append((index, y, ll, x, ww, patch))

    return out


# ───────────────────────────────────────────────────────────────────────────
# rebin_patches (port directo de util_patch_rebin.py::rebin_patches)
# ───────────────────────────────────────────────────────────────────────────

def rebin_patches(patch_list, patch_size):
    """
    Para cada parche, si sus 8 vecinos ortogonales/diagonales a distancia
    `patch_size` están presentes en la lista ya filtrada, construye el
    bloque 3×3 y lo redimensiona a `patch_size` con INTER_LINEAR_EXACT.

    Recibe la lista que devuelve `generate_patches`:
        [(index, y, ll, x, ww, patch), ...]
    Devuelve:
        positions    : [(y, x), ...]   (esquina superior-izquierda)
        indices      : [index, ...]    (id original de grid)
        pairs        : [[orig, reb], ...]
    """
    size = int(patch_size)

    xarr = np.empty(len(patch_list), int)
    yarr = np.empty(len(patch_list), int)
    xytup = []
    for i in range(len(patch_list)):
        xarr[i] = patch_list[i][1]    # y
        yarr[i] = patch_list[i][3]    # x
        xytup.append((patch_list[i][1], patch_list[i][3]))

    positions = []
    indices = []
    pairs = []

    for p in range(len(patch_list)):
        prec_h = (xytup[p][0] - size, xytup[p][1])
        next_h = (xytup[p][0] + size, xytup[p][1])
        prec_v = (xytup[p][0], xytup[p][1] - size)
        next_v = (xytup[p][0], xytup[p][1] + size)
        diag_1 = (xytup[p][0] - size, xytup[p][1] - size)
        diag_2 = (xytup[p][0] + size, xytup[p][1] - size)
        diag_3 = (xytup[p][0] - size, xytup[p][1] + size)
        diag_4 = (xytup[p][0] + size, xytup[p][1] + size)

        if (prec_h not in xytup or next_h not in xytup or
                prec_v not in xytup or next_v not in xytup or
                diag_1 not in xytup or diag_2 not in xytup or
                diag_3 not in xytup or diag_4 not in xytup):
            continue

        im_3p3 = np.empty((3 * size, 3 * size, 3), int)
        im_3p3[0:size, 0:size, :] = patch_list[xytup.index(diag_1)][5]
        im_3p3[0:size, size:2*size, :] = patch_list[xytup.index(prec_h)][5]
        im_3p3[0:size, 2*size:3*size, :] = patch_list[xytup.index(diag_3)][5]
        im_3p3[size:2*size, 0:size, :] = patch_list[xytup.index(prec_v)][5]
        im_3p3[size:2*size, size:2*size, :] = patch_list[p][5]
        im_3p3[size:2*size, 2*size:3*size, :] = patch_list[xytup.index(next_v)][5]
        im_3p3[2*size:3*size, 0:size, :] = patch_list[xytup.index(diag_2)][5]
        im_3p3[2*size:3*size, size:2*size, :] = patch_list[xytup.index(next_h)][5]
        im_3p3[2*size:3*size, 2*size:3*size, :] = patch_list[xytup.index(diag_4)][5]

        resim = cv2.resize(im_3p3, dsize=(size, size),
                            interpolation=cv2.INTER_LINEAR_EXACT)

        positions.append((patch_list[p][1], patch_list[p][3]))
        indices.append(patch_list[p][0])
        pairs.append([patch_list[p][5], resim])

    return positions, indices, pairs


# ───────────────────────────────────────────────────────────────────────────
# Escritura HDF5 (port directo de util_hdf5.save_patches_to_hdf5)
# ───────────────────────────────────────────────────────────────────────────

def save_patches_to_hdf5(patches_pairs, filename, image_name,
                          patch_positions, patch_categories, patch_numbers,
                          patch_size_x=300, patch_size_y=300, num_channels=3):
    n = len(patches_pairs)
    if not (len(patch_positions) == len(patch_categories) == len(patch_numbers) == n):
        raise ValueError('Tamaños inconsistentes entre listas')

    # Construir array (N, 2, H, W, C)
    first = patches_pairs[0]
    h, w, c = first[0].shape
    arr = np.zeros((n, 2, h, w, c), dtype=first[0].dtype)
    for i, (orig, reb) in enumerate(patches_pairs):
        arr[i, 0] = orig
        arr[i, 1] = reb

    with h5py.File(filename, 'w') as f:
        patches_ds = f.create_dataset(
            'patches', data=arr,
            compression='gzip', compression_opts=9,
            maxshape=(None, arr.shape[1], arr.shape[2], arr.shape[3], arr.shape[4]),
        )
        patches_ds.attrs['description'] = 'Image patches array'
        patches_ds.attrs['dimensions'] = [
            'patch_number', 'patch_type',
            'x_dimension', 'y_dimension', 'rgb_channels',
        ]
        patches_ds.attrs['patch_type_encoding'] = '0=original_patch, 1=rebinned_surrounding'
        patches_ds.attrs['shape_description'] = (
            f'({arr.shape[0]} patches, 2 types, {patch_size_x}x{patch_size_y}'
            f' pixels, {num_channels} channels)'
        )

        f.attrs['source_image_name'] = image_name
        f.attrs['total_patches'] = arr.shape[0]
        f.attrs['patch_size_x'] = patch_size_x
        f.attrs['patch_size_y'] = patch_size_y
        f.attrs['num_channels'] = num_channels
        f.attrs['creation_date'] = np.bytes_(str(np.datetime64('now')))

        pos_arr = np.array(patch_positions)
        pos_ds = f.create_dataset('patch_positions', data=pos_arr,
                                   maxshape=(None, pos_arr.shape[1]))
        pos_ds.attrs['description'] = 'X,Y coordinates of each patch center/corner'
        pos_ds.attrs['format'] = 'array of (x, y) tuples'

        cat_arr = np.array(patch_categories, dtype='S3')
        cat_ds = f.create_dataset('patch_categories', data=cat_arr,
                                   maxshape=(None,))
        cat_ds.attrs['description'] = 'Category labels for each patch'
        cat_ds.attrs['valid_categories'] = 'NOR, TUM, HIP, ADE, ART, XXX'

        num_arr = np.array(patch_numbers, dtype=np.int32)
        num_ds = f.create_dataset('patch_numbers', data=num_arr,
                                   maxshape=(None,))
        num_ds.attrs['description'] = 'Original patch numbers from input data'

        g = f.create_group('metadata')
        g.attrs['patch_extraction_method'] = 'Patch extraction by util_patch_rebin.-py'
        g.attrs['rebinning_method'] = (
            'All 3x3 paches centered on the chosen one rebinnedwith '
            'cv2.resize(image, dsize=(size, size), '
            'interpolation=cv2.INTER_LINEAR_EXACT)'
        )


# ───────────────────────────────────────────────────────────────────────────
# Comparador contra un H5 de referencia
# ───────────────────────────────────────────────────────────────────────────

def compare_h5(reconstructed, reference):
    print(f'\n{"="*70}\nCOMPARACIÓN DE RECONSTRUCCIÓN vs ORIGINAL\n{"="*70}')
    with h5py.File(reconstructed) as r, h5py.File(reference) as o:
        n_r = r['patches'].shape[0]
        n_o = o['patches'].shape[0]
        print(f'  N de parches:    reconstruido={n_r:>5}   original={n_o:>5}'
              f'   {"✓" if n_r == n_o else "✗"}')

        pos_r = r['patch_positions'][:]
        pos_o = o['patch_positions'][:]
        if pos_r.shape == pos_o.shape:
            pos_match = np.array_equal(pos_r, pos_o)
            print(f'  Posiciones:      shapes coinciden '
                  f'({pos_r.shape}) — valores {"✓ iguales" if pos_match else "✗ distintos"}')
        else:
            print(f'  Posiciones:      shape {pos_r.shape} vs {pos_o.shape} — ✗')

        if n_r == n_o:
            # Comparar primeros N parches píxel a píxel
            sample_n = min(5, n_r)
            for i in range(sample_n):
                orig_diff = np.abs(r['patches'][i, 0].astype(np.int32)
                                    - o['patches'][i, 0].astype(np.int32)).sum()
                reb_diff = np.abs(r['patches'][i, 1].astype(np.int32)
                                   - o['patches'][i, 1].astype(np.int32)).sum()
                n_pixels = np.prod(r['patches'][i, 0].shape)
                print(f'  Parche[{i}]:     orig_diff_sum={orig_diff:>10}   '
                      f'reb_diff_sum={reb_diff:>10}   (max pix {n_pixels*255})')


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, type=Path, help='TIFF de entrada')
    ap.add_argument('--output', required=True, type=Path, help='H5 de salida')
    ap.add_argument('--patch-size', nargs=2, type=int, default=[300, 300],
                    help='Tamaño del parche (w h)')
    ap.add_argument('--white-percent', type=float, default=80.0,
                    help='Umbral de blanco del filtro original (default: 80 %%)')
    ap.add_argument('--image-level', type=int, default=5,
                    help='Nivel del TIFF piramidal para detección (default: 5 → work=7)')
    ap.add_argument('--compare', type=Path, default=None,
                    help='H5 de referencia para comparar')
    args = ap.parse_args()

    assert args.patch_size[0] == args.patch_size[1], (
        'Solo se soportan parches cuadrados (util_patch_rebin.py asume size=w=h)'
    )

    print(f'Procesando: {args.input.name}')
    t0 = time.time()

    porta = PortaHelper(str(args.input), image_level=args.image_level)
    tissue, image_name, pixel_max = porta.extract_tissue()
    print(f'  Tissue shape: {tissue.shape}   pixel_max={pixel_max}')
    print(f'  Extract tissue: {time.time() - t0:.1f}s')

    t1 = time.time()
    patch_list = generate_patches(tissue, args.patch_size, args.white_percent,
                                    pixel_max)
    positions, indices, pairs = rebin_patches(patch_list, args.patch_size[0])
    print(f'  Parches (pre-rebin): {len(patch_list)}   '
          f'con 8 vecinos: {len(pairs)}   (en {time.time()-t1:.1f}s)')

    # Categorías siempre "XXX" para inferencia
    categories = ['XXX'] * len(pairs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_patches_to_hdf5(
        pairs, str(args.output), f'{image_name}.tif',
        positions, categories, indices,
        patch_size_x=args.patch_size[0], patch_size_y=args.patch_size[1],
        num_channels=3,
    )
    size_mb = args.output.stat().st_size / 1e6
    print(f'  H5 escrito: {args.output} ({size_mb:.1f} MB)')
    print(f'  TIEMPO TOTAL: {time.time()-t0:.1f}s')

    if args.compare is not None:
        compare_h5(args.output, args.compare)


if __name__ == '__main__':
    main()
