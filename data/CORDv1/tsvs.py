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

    output_path = os.path.join(args.output, 'collection.tsv')
    assert not os.path.exists(output_path), output_path

    with open(output_path, 'w') as f:
        print_message(f"#> Writing to {f.name}...")

        line = ['id', 'text', 'title', 'cid']
        line = '\t'.join(map(str, line))
        f.write(line + '\n')

        PID = 1
        for cid, paper in papers.items():
            meta = metadata[cid][0]
            title, abstract, date = meta['title'], meta['abstract'], meta['publish_time']

            for p in paper['passages']:
                line = [p['heading'], date, p['text'], abstract]
                line = ' | '.join(line)
                line = ' '.join(line.strip().split())
                line = [PID, line, title, cid]
                line = '\t'.join(map(str, line))
                f.write(line + '\n')

                PID += 1

    for split in ['train', 'dev', 'test']:
        input_path = os.path.join(args.output, f'{split}.jsonl')

        output_path = os.path.join(args.output, f'{split}.source')
        assert not os.path.exists(output_path), output_path

        with open(input_path) as f:
            with open(output_path, 'w') as g:
                print_message(f"#> Writing to {g.name}...")
                for line in f:
                    example = ujson.loads(line)
                    line = [example['title'], example['date'], example['abstract']]
                    line = ' | '.join(line)
                    line = ' '.join(line.strip().split())

                    for passage in example['background']:  # Repeat source for each target
                        g.write(line + '\n')

        output_path = os.path.join(args.output, f'{split}.target')
        assert not os.path.exists(output_path), output_path

        with open(input_path) as f:
            with open(output_path, 'w') as g:
                print_message(f"#> Writing to {g.name}...")
                for line in f:
                    example = ujson.loads(line)

                    for passage in example['background']:
                        passage = passage['text']
                        passage = ' '.join(passage.strip().split())
                        g.write(passage + '\n')
        
        output_path = os.path.join(args.output, f'{split}.meta')
        assert not os.path.exists(output_path), output_path

        with open(input_path) as f:
            with open(output_path, 'w') as g:
                print_message(f"#> Writing to {g.name}...")
                for line in f:
                    example = ujson.loads(line)
                    cid = example['cid']
                    meta = metadata[cid][0]

                    for passage in example['background']:
                        line = ujson.dumps(meta)
                        g.write(line + '\n')

    print("#> Done.")


if __name__ == "__main__":
    parser = ArgumentParser(description=".")

    # Input Arguments.
    parser.add_argument('--corpus', dest='corpus', required=True, type=str)

    args = parser.parse_args()
    args.output = args.corpus

    main(args)
