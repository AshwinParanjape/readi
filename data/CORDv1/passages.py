#
#  This script assumes that you've downloaded CORD cord-19_2021-04-26.tar.gz and decompressed it.
#  Further, it assumes that you've used preprocess.py to preprocess the data.
#  Lastly, it assumes that you've used corpus.py and examples.py to create the corpus.
#

import os
import tqdm
import ujson
import pathlib


from argparse import ArgumentParser
from data.CORDv1.utils import print_message


def main(args):
    print_message("#> Starting..")

    with open(os.path.join(pathlib.Path(args.output).parent, 'metadata.json')) as f:
        metadata = ujson.load(f)

    with open(os.path.join(args.output, 'papers.json')) as f:
        papers = ujson.load(f)

    output_path = os.path.join(args.output, 'collection.jsonl')
    assert not os.path.exists(output_path), output_path

    with open(output_path, 'w') as f:
        print_message(f"#> Writing to {f.name}...")

        f.write('\n')  # Empty first line.

        PID = 1
        for cid, paper in papers.items():
            meta = metadata[cid][0]
            title, abstract, date = meta['title'], meta['abstract'], meta['publish_time']

            for p in paper['passages']:
                line = [p['heading'], date, p['text'], abstract]
                line = ' | '.join(line)
                line = ' '.join(line.strip().split())

                passage = {'pid': PID, 'title': title, 'cid': cid, 'heading': p['heading'], 'line': line,
                           'only_text': p['text'], 'citations': p['citations'], 'date': date, 'abstract': abstract}

                f.write(ujson.dumps(passage) + '\n')

                PID += 1

    print("#> Done.")


if __name__ == "__main__":
    parser = ArgumentParser(description=".")

    # Input Arguments.
    parser.add_argument('--corpus', dest='corpus', required=True, type=str)

    args = parser.parse_args()
    args.output = args.corpus

    main(args)
