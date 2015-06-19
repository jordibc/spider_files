#!/usr/bin/env python

"""
Do stuff with images or volumes in spider format (read them, plot
them, save them).
"""

# Format: http://spider.wadsworth.org/spider_doc/spider/docs/image_doc.html

import sys
from struct import pack, unpack
import numpy as np
import matplotlib.pyplot as plt

# From the spider format:
labels = {i-1: v for i, v in [
    (1, 'NZ'),
    (2, 'NY'),
    (3, 'IREC'),
    (5, 'IFORM'),
    (6, 'IMAMI'),
    (7, 'FMAX'),
    (8, 'FMIN'),
    (9, 'AV'),
    (10, 'SIG'),
    (12, 'NX'),
    (13, 'LABREC'),
    (14, 'IANGLE'),
    (15, 'PHI'),
    (16, 'THETA'),
    (17, 'GAMMA'),
    (18, 'XOFF'),
    (19, 'YOFF'),
    (20, 'ZOFF'),
    (21, 'SCALE'),
    (22, 'LABBYT'),
    (23, 'LENBYT'),
    (24, 'ISTACK/MAXINDX'),
    (26, 'MAXIM'),
    (27, 'IMGNUM'),
    (28, 'LASTINDX'),
    (31, 'KANGLE'),
    (32, 'PHI1'),
    (33, 'THETA1'),
    (34, 'PSI1'),
    (35, 'PHI2'),
    (36, 'THETA2'),
    (37, 'PSI2'),
    (38, 'PIXSIZ'),
    (39, 'EV'),
    (40, 'PROJ'),
    (41, 'MIC'),
    (42, 'NUM'),
    (43, 'GLONUM'),
    (101, 'PSI3'),
    (102, 'THETA3'),
    (103, 'PHI3'),
    (104, 'LANGLE')]}

# Inverse.
locations = {v: k for k, v in labels.iteritems()}


def open_volume(filename, endianness='ieee-le'):
    """Read a volume in spider format and return a numpy array with it."""
    f = open(filename)
    e = {'ieee-le': '<', 'ieee-be': '>'}[endianness]
    fields = unpack('%s13f' % e, f.read(4 * 13))
    nz, ny, nx, labrec = [int(fields[i]) for i in [0, 1, 11, 12]]
    f.seek(4 * nx * labrec)  # go to end of header
    return np.fromfile(f, dtype='%sf4' % e).reshape((nx, ny, nz))
    # .transpose(1, 0, 2) ?


def open_image(filename, n, endianness='ieee-le'):
    """Read an image from a file in spider format and return a numpy array."""
    f = open(filename)
    e = {'ieee-le': '<', 'ieee-be': '>'}[endianness]
    fields = unpack('%s256f' % e, f.read(4 * 256))
    nz, ny, nx, labrec = [int(fields[i]) for i in [0, 1, 11, 12]]
    size = nx * labrec
    if size != 256:
        f.seek(4 * size)
    f.seek(4 * n * (size + nx * ny * nz), 1)
    return np.fromfile(f, dtype='%sf4' % e, count=nx*ny).reshape((nx, ny))


def save_volume(vol, filename):
    """Save volume vol into a file, with the spider format."""
    nx, ny, nz = vol.shape
    fields = [0.0] * nx
    values = {
        'NZ': nz, 'NY': ny,
        'IREC': 3,  # number of records (including header records)
        'IFORM': 3,  # 3D volume
        'FMAX': vol.max(), 'FMIN': vol.min(),
        'AV': vol.mean(), 'SIG': vol.std(),
        'NX': nx,
        'LABREC': 1,  # number of records in file header (label)
        'SCALE': 1,
        'LABBYT': 4 * nx,  # number of bytes in header
        'LENBYT': 4 * nx,  # record length in bytes (only 1 in our header)
    }
    for label, value in values.items():
        fields[locations[label]] = float(value)
    header = pack('%df' % nx, *fields)
    with open(filename, 'w') as f:
        f.write(header)
        vol.tofile(f)


def show_slices(vol):
    """Interactively show requested slices of volume vol."""
    while True:
        layer = raw_input('Layer ("q" to quit): ')
        if layer.lower() == 'q':
            break
        elif not layer.isdigit() or not 0 <= int(layer) <= vol.shape[2]:
            continue
        try:
            plt.imshow(vol[:,:,int(layer)])
            plt.colorbar()
            plt.show()
        except Exception as e:
            print(e)


def show_header(filename, endianness='ieee-le'):
    """Show the header information of volume in file filename."""
    print('Reading header of %s ...' % filename)
    f = open(filename)
    e = {'ieee-le': '<', 'ieee-be': '>'}[endianness]
    fields = unpack('%s256f' % e, f.read(4 * 256))
    size = int(fields[11] * fields[12])
    if size != 256:
        f.seek(0)
        fields = unpack('%s%df' % (e, size), f.read(4 * size))

    for i, x in enumerate(fields):
        if x != 0:
            print('%5d: %-12g (%s)' % (i+1, x, labels.get(i, 'unknown')))
    print('All other values are 0.\nFor reference see '
          'http://spider.wadsworth.org/spider_doc/spider/docs/image_doc.html')

    # If it only contains one header (a volume, for example) we are done.
    if fields[4] != 1:
        return

    def get_header():
        """Return list with all the header fields and advance the file."""
        pos = f.tell()
        new_fields = unpack('%s256f' % e, f.read(4 * 256))
        size = int(new_fields[11] * new_fields[12])
        if size != 256:
            f.seek(pos)
            new_fields = unpack('%s%df' % (e, size), f.read(4 * size))
        nz, ny, nx = [int(new_fields[i]) for i in [0, 1, 11]]

        f.seek(4 * nx * ny * nz, 1)  # go to the next header (or EOF)

        return new_fields

    print('\nShowing the next header and all the subsequent ones that are '
          'different.')
    maxim = fields[25]
    stack = 0
    while True:
        stack += 1
        if stack > maxim:
            break
        new_fields = get_header()
        if new_fields == fields:
            sys.stdout.write('.')
            sys.stdout.flush()
            continue
        print('\nStack %d' % stack)
        for i, x in enumerate(new_fields):
            if x != 0:
                print('%5d: %-12g (%s)' % (i+1, x, labels.get(i, 'unknown')))
        fields = new_fields
    print('\nNumber of stacks processed: %d' % (stack - 1))


def ftype(filename, endianness='ieee-le'):
    """Return 1 if filename is a stack of 2D images, 3 if a volume."""
    e = {'ieee-le': '<', 'ieee-be': '>'}[endianness]
    return int(unpack('%s5f' % e, open(filename).read(4 * 5))[-1])

    

if __name__ == '__main__':
    # Read file.
    if len(sys.argv) < 2:
        sys.exit('usage: %s <filename>' % sys.argv[0])

    fname = sys.argv[1]
    show_header(fname)

    if ftype(fname) == 1:
        while True:
            stack = raw_input('Stack (from 0 to MAXIM-1) ("q" to quit): ')
            if stack.lower() == 'q':
                break
            elif not stack.isdigit() or not 0 <= int(stack):
                continue
            try:
                img = open_image(fname, int(stack))
                plt.imshow(np.log(np.abs(img) + 1e-10))
                plt.colorbar()
                plt.show()
            except Exception as e:
                print(e)
    else:
        print('Opening file %s' % fname)
        vol = open_volume(fname)

        # Play with it.
        show_slices(vol)

        # # Modify volume.
        # print("Processing volume...")

        # # for i in range(150, 256):
        # #     vol[:,:,i] *= 10

        # vol2 = np.abs(np.fft.fftshift(np.fft.fftn(vol))).astype('f4')

        # # Save it.
        # print('Saving modified volume in test.vol')
        # save_volume(vol2, 'test.vol')
        # #show_header('test.vol')
