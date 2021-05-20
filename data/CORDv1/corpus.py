#
#  This script assumes that you've downloaded CORD cord-19_2021-04-26.tar.gz and decompressed it.
#  Further, it assumes that you've used preprocess.py to preprocess the data.
#

import os
import csv
import ujson
import random

from argparse import ArgumentParser
from collections import defaultdict
from data.CORDv1.utils import MIN_CITATIONS, bib_to_key, bib_to_citation, list_of_unique_dicts


class CorpusBuilder:
    def __init__(self, path):
        self.fulltext = self.load_fulltext(path)

    def run(self):
        fulltext = self.fulltext
        anthology = self.count_citations(fulltext)
        papers = self.fix_citations(fulltext, anthology)

        return anthology, papers

    def load_fulltext(self, path):
        with open(path) as f:
            fulltext = ujson.load(f)

        return fulltext

    def count_citations(self, fulltext):
        anthology = defaultdict(list)
        anthology_bibs = defaultdict(list)

        for cid, paper in fulltext.items():
            for _, bib in paper['bib_entries'].items():
                key = bib_to_key(bib)

                if key:
                    anthology[key].append(cid)
                    anthology_bibs[key].append(bib)

        # Drop rare citations
        for key in anthology:
            if len(set(anthology[key])) < MIN_CITATIONS:
                del anthology[key]
                del anthology_bibs[key]
            else:
                unique_bibs = list_of_unique_dicts(anthology_bibs[key])
                anthology[key] = (anthology[key], unique_bibs)

        return anthology

    def fix_citations(self, fulltext, anthology):
        papers = {}

        for cid, paper in fulltext.items():
            sections = defaultdict(list)
            bibs = paper['bib_entries']
            passages = []

            for p in paper['body_text']:
                heading = p['section'].lower()
                sections[heading].append(p)

            for heading, section in sections:
                section_passages = self.fix_citations_in_section(
                    section, bibs, anthology)
                passages.extend(section_passages)

            papers[cid] = {'raw': paper, 'passages': passages}

        return papers

    def fix_citations_in_section(self, section, bibs, anthology):
        passages = []

        for p in section:
            text, new_text, offset = p['text'], [], 0
            p_citations = []

            for cite in p['cite_spans']:
                start, end, refid = cite['start'], cite['end'], cite['ref_id']

                more_text = text[offset:start]
                more_text = more_text.replace('{', ' ')
                more_text = more_text.replace('}', ' ')
                new_text.append(more_text)

                offset = end

                if refid:
                    bib = bibs[refid]
                    key = bib_to_key(bib)
                    citation = bib_to_citation(bib)

                    if citation and (key in anthology):
                        new_text.append(citation)
                        p_citations.append((key, citation, bib))

            new_text.append(text[offset:])
            new_text = ' '.join(new_text)

            psg = {}
            psg['heading'] = section
            psg['text'] = new_text
            psg['raw'] = p
            psg['citations'] = p_citations

            passages.append(psg)

        return passages


def main(args):
    builder = CorpusBuilder(args.fulltext)
    anthology, papers = builder.run()

    os.makedirs(args.output)

    with open(os.path.join(args.output, 'anthology.json'), 'w') as f:
        print(f"#> Writing to {f.name}...")
        ujson.dump(anthology, f)

    with open(os.path.join(args.output, 'papers.json'), 'w') as f:
        print(f"#> Writing to {f.name}...")
        ujson.dump(papers, f)

    print("#> Done.")


if __name__ == "__main__":
    parser = ArgumentParser(description=".")

    # Input Arguments.
    parser.add_argument('--fulltext', dest='fulltext', required=True, type=str)

    args = parser.parse_args()
    args.output = os.path.join(os.path.dirname(args.fulltext), 'corpus')

    assert not os.path.exists(args.output), args.output

    main(args)


# TODO/NOTE: To avoid cutting anthology halfway and to keep this simple, I'll keep the original passages uncut!
# I think they're generally short enough.

# passages = [p['text'] for p in passages]
# passages = ' '.join(passages)

# passages = passages.replace('\t', ' ')
# passages = passages.replace('\n', ' ')
# passages = passages.replace('\r', ' ')

# words = passages.split()
# passages = [words[offset:offset + MAX_LIMIT] for offset in range(0, len(words), MAX_LIMIT)]
# passages = [' '.join(psg) for psg in passages]
