#
#  This script assumes that you've downloaded CORD cord-19_2021-04-26.tar.gz and decompressed it.
#

import os
import csv
import ujson
import random

from argparse import ArgumentParser
from collections import defaultdict


# Every heading that contains "background" or "literaure" from within
# the top-2000 section headings (by # of paragraphs, not unique papers).
BackgroundHeadings = {'background', 'related work', 'literature review', 'related works', 'background:',
                      'theoretical background', 'literature search', 'ii. related work', 'related literature',
                      'background and related work', 'literature survey', 'historical background', '| background',
                      'literature', 'background and rationale {6a}', 'research background', 'review of literature',
                      'background information', 'comparison with existing literature', 'abstract background',
                      'background & summary', 'background and rationale'}


def main(args):
    metadata = defaultdict(list)  # available for all papers, including those without full-text
    fulltext = {}  # only for full-text papers
    fulltext_has_bg = {}

    with open(os.path.join(args.data, 'metadata.csv'), newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')

        for idx, row in enumerate(reader):
            if idx % 20_000 == 0:
                print(idx)
            
            sha = row['sha']

            metadata[sha].append(row)

            for ftype in ['pdf_json', 'pmc_json']:
                try:
                    with open(os.path.join(args.data, 'document_parses', ftype, f'{sha}.json')) as g:
                        paper = ujson.load(g)
                        body = paper['body_text']

                        paper['has_bg'] = any(p['section'].lower() in BackgroundHeadings for p in body)

                        fulltext[sha] = paper

                        if paper['has_bg']:
                            fulltext_has_bg[sha] = paper

                        break
                except:
                    pass
    
    os.makedirs(args.output)

    with open(os.path.join(args.output, 'metadata.json'), 'w') as f:
        print("#> Writing to {f.name}...")
        ujson.dump(metadata, f)

    with open(os.path.join(args.output, 'fulltext_has_bg.json'), 'w') as f:
        print("#> Writing to {f.name}...")
        ujson.dump(fulltext_has_bg, f)

    with open(os.path.join(args.output, 'fulltext.json'), 'w') as f:
        print("#> Writing to {f.name}...")
        ujson.dump(fulltext, f)


if __name__ == "__main__":
    parser = ArgumentParser(description=".")

    # Input Arguments.
    parser.add_argument('--data', dest='data', required=True, type=str)

    args = parser.parse_args()
    args.output = os.path.join(args.data, 'cleaned')

    assert not os.path.exists(args.output), args.output

    main(args)
