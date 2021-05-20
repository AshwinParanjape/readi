#
#  This script assumes that you've downloaded CORD cord-19_2021-04-26.tar.gz and decompressed it.
#  Further, it assumes that you've used preprocess.py to preprocess the data.
#  Lastly, it assumes that you've used corpus.py to create the corpus.
#

import os
import ujson

from argparse import ArgumentParser
from data.CORDv1.utils import BackgroundHeadings, DEV_SIZE, TEST_SIZE, print_message


class ExampleBuilder:
    def __init__(self, path):
        self.papers = self.load_papers(os.path.join(path, 'papers.json'))

    def run(self):
        self.examples = self.create_examples()
        self.train, self.dev, self.test = self.create_splits()

        return self.train, self.dev, self.test
    
    def load_papers(self, path):
        with open(path) as f:
            print_message(f"#> Load {f.name}..")
            papers = ujson.load(f)

        return papers

    def create_examples(self):
        papers = self.papers
        examples = []

        for cid, paper in papers.items():
            example = self.create_example(cid, paper)

            if example:
                examples.append(example)

        return examples

    def create_example(self, cid,  paper):
        raw = paper['raw']
        passages = paper['passages']
        background = []

        for p in passages:
            if p['heading'] not in BackgroundHeadings:
                continue

            if len(p['citations']) == 0:
                continue

            background.append(p)

        if len(background) == 0:
            return None

        example = {}
        example['cid'] = cid
        example['title'] = raw['metadata']['title']
        example['date'] = raw['metadata']['publish_time']
        example['abstract'] = raw['metadata']['abstract']
        example['background'] = background

        return example

    def create_splits(self):
        self.examples = sorted(self.examples, key=lambda ex: ex['date'])

        train_size = len(self.examples) - DEV_SIZE - TEST_SIZE
        assert train_size > DEV_SIZE, (len(self.examples), DEV_SIZE, TEST_SIZE)

        train = self.examples[:train_size]
        dev = self.examples[train_size:train_size+DEV_SIZE]
        test = self.examples[train_size+DEV_SIZE:]

        return train, dev, test


def main(args):
    print_message("#> Starting..")

    builder = ExampleBuilder(args.corpus)
    splits = builder.run()

    names = ['train', 'dev', 'test']
    for name, data in zip(names, splits):
        path = os.path.join(args.output, f'{name}.jsonl')
        assert not os.path.exists(path), path

        with open(path, 'w') as f:
            print_message(f"#> Writing to {f.name}...")
            for example in data:
                line = ujson.dumps(example) + '\n'
                f.write(line)

    print("#> Done.")


if __name__ == "__main__":
    parser = ArgumentParser(description=".")

    # Input Arguments.
    parser.add_argument('--corpus', dest='corpus', required=True, type=str)

    args = parser.parse_args()
    args.output = args.corpus

    main(args)
