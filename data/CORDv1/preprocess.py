#
#  This script assumes that you've downloaded CORD cord-19_2021-04-26.tar.gz and decompressed it.
#

import os
import csv
import ujson
import random

from argparse import ArgumentParser
from collections import defaultdict


def main(args):
    metadata = defaultdict(list)  # available for all papers
    fulltext = {}  # only for full-text papers
    fulltext_has_bg = {}  # only for papers with BG section(s)

    with open(os.path.join(args.data, 'metadata.csv'), newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')

        for idx, row in enumerate(reader):
            if idx % 20_000 == 0:
                print(idx)

            cid = row['cord_uid']
            metadata[cid].append(row)

            parses = [row['pdf_json_files'], row['pmc_json_files']]
            parse_path = next((p for p in parses if len(p)), None)

            if parse_path is None:
                continue

            # if there's multiple, we'll take the first one
            parse_path = parse_path.split(';')[0]

            with open(os.path.join(args.data, parse_path)) as g:
                paper = ujson.load(g)
                body = paper['body_text']

                paper['has_bg'] = any(
                    p['section'].lower() in BackgroundHeadings for p in body)

                fulltext[cid] = paper

                if paper['has_bg']:
                    fulltext_has_bg[cid] = paper

    os.makedirs(args.output)

    with open(os.path.join(args.output, 'metadata.json'), 'w') as f:
        print(f"#> Writing to {f.name}...")
        ujson.dump(metadata, f)

    with open(os.path.join(args.output, 'fulltext_has_bg.json'), 'w') as f:
        print(f"#> Writing to {f.name}...")
        ujson.dump(fulltext_has_bg, f)

    with open(os.path.join(args.output, 'fulltext.json'), 'w') as f:
        print(f"#> Writing to {f.name}...")
        ujson.dump(fulltext, f)

    print("#> Done.")


if __name__ == "__main__":
    parser = ArgumentParser(description=".")

    # Input Arguments.
    parser.add_argument('--data', dest='data', required=True, type=str)

    args = parser.parse_args()
    args.output = os.path.join(args.data, 'cleaned')

    assert not os.path.exists(args.output), args.output

    main(args)
