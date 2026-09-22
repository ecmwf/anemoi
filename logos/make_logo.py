#!/usr/bin/env python3
# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Render a logo SVG from docs-logo-template.svg with a custom sub-title string.

Reads ``docs-logo-template.svg``, replaces the ``template`` text with the given
string, and horizontally centres it using the glyph-width table embedded below.

The centre line is taken from the "anemoi" text element in the template, so the
sub-title always sits centred under the logo word — even if the template moves.

Usage:
    ./make_logo.py metadata                  # -> metadata.svg
    ./make_logo.py "inference" -o foo.svg
"""

import argparse
import re
import sys

TEMPLATE = "docs-logo-template.svg"

# Advance widths in em units (fraction of font size), keyed by codepoint.

_WIDTHS = {
    32: 0.296,
    33: 0.444,
    34: 0.52,
    35: 0.592,
    36: 0.592,
    37: 0.907,
    38: 0.759,
    39: 0.296,
    40: 0.314,
    41: 0.314,
    42: 0.482,
    43: 0.666,
    44: 0.296,
    45: 0.296,
    46: 0.296,
    47: 0.407,
    48: 0.592,
    49: 0.592,
    50: 0.592,
    51: 0.592,
    52: 0.592,
    53: 0.592,
    54: 0.592,
    55: 0.592,
    56: 0.592,
    57: 0.592,
    58: 0.296,
    59: 0.296,
    60: 0.666,
    61: 0.666,
    62: 0.666,
    63: 0.556,
    64: 0.8,
    65: 0.741,
    66: 0.648,
    67: 0.685,
    68: 0.759,
    69: 0.63,
    70: 0.593,
    71: 0.778,
    72: 0.759,
    73: 0.296,
    74: 0.5,
    75: 0.722,
    76: 0.537,
    77: 0.944,
    78: 0.815,
    79: 0.832,
    80: 0.63,
    81: 0.852,
    82: 0.648,
    83: 0.574,
    84: 0.574,
    85: 0.741,
    86: 0.704,
    87: 1.0,
    88: 0.722,
    89: 0.648,
    90: 0.63,
    91: 0.296,
    92: 0.407,
    93: 0.296,
    94: 0.666,
    95: 0.5,
    96: 0.26,
    97: 0.537,
    98: 0.63,
    99: 0.482,
    100: 0.63,
    101: 0.574,
    102: 0.37,
    103: 0.63,
    104: 0.574,
    105: 0.26,
    106: 0.26,
    107: 0.556,
    108: 0.26,
    109: 0.87,
    110: 0.574,
    111: 0.61,
    112: 0.63,
    113: 0.63,
    114: 0.407,
    115: 0.463,
    116: 0.407,
    117: 0.574,
    118: 0.556,
    119: 0.833,
    120: 0.556,
    121: 0.556,
    122: 0.5,
    123: 0.37,
    124: 0.222,
    125: 0.37,
    126: 0.666,
}


def text_width(string, size):
    missing = [c for c in string if ord(c) not in _WIDTHS]
    if missing:
        sys.exit(f"error: no width for {missing!r}; only printable ASCII is supported")
    return sum(_WIDTHS[ord(c)] for c in string) * size


def attr(tag, name):
    m = re.search(rf'{name}="([^"]*)"', tag)
    if not m:
        sys.exit(f"error: attribute {name!r} not found in: {tag}")
    return m.group(1)


def find_text(svg, ident):
    """Return the full <text id="ident" ...>content</text> element."""
    m = re.search(rf'<text id="{ident}"[^>]*>.*?</text>', svg)
    if not m:
        sys.exit(f'error: no <text id="{ident}"> in template')
    return m.group(0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("text", help="sub-title string to render")
    p.add_argument("-o", "--output", help="output SVG (default: <text>.svg)")
    p.add_argument("--white", action="store_true", help="render sub-title text in white (for dark mode)")
    p.add_argument("--template", default=TEMPLATE)
    args = p.parse_args(argv)

    svg = open(args.template, encoding="utf-8").read()

    # Centre line = centre of the "anemoi" logo word.
    anemoi = find_text(svg, "anemoi")
    a_x = float(attr(anemoi, "x"))
    a_size = float(attr(anemoi, "font-size"))
    a_content = re.search(r">([^<]*)</text>", anemoi).group(1)
    centre = a_x + text_width(a_content, a_size) / 2

    # Replace the template sub-title, re-centred on `centre`.
    tmpl = find_text(svg, "template")
    t_size = float(attr(tmpl, "font-size"))
    new_x = centre - text_width(args.text, t_size) / 2

    new_tag = re.sub(r'x="[^"]*"', f'x="{new_x:.4f}"', tmpl, count=1)
    new_tag = re.sub(r">([^<]*)</text>", f">{args.text}</text>", new_tag, count=1)
    if args.white:
        new_tag = re.sub(r'fill="[^"]*"', 'fill="#ffffff"', new_tag, count=1)

    out_svg = svg.replace(tmpl, new_tag, 1)
    out_path = args.output or f"{args.text}.svg"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(out_svg)
    print(f"wrote {out_path}: {args.text!r} centred on x={centre:.3f} (x={new_x:.4f})", file=sys.stderr)


if __name__ == "__main__":
    main()
