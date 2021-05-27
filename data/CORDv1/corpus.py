#
#  This script assumes that you've downloaded CORD cord-19_2021-04-26.tar.gz and decompressed it.
#  Further, it assumes that you've used preprocess.py to preprocess the data.
#

import os
import ujson
import tqdm

from argparse import ArgumentParser
from collections import defaultdict

from data.CORDv1.utils import MIN_CITATIONS, bib_to_key, bib_to_citation, list_of_unique_dicts, print_message


class CorpusBuilder:
    def __init__(self, path):
        self.fulltext = self.load_fulltext(path)

    def run(self):
        fulltext = self.fulltext
        self.anthology = self.build_anthology(fulltext)
        self.papers = self.normalize_citations(fulltext)

        return self.anthology, self.papers

    def load_fulltext(self, path):
        with open(path) as f:
            print_message(f"#> Load {f.name}..")
            fulltext = ujson.load(f)

        return fulltext

    def build_anthology(self, fulltext):
        print_message("#> Build anthology..")

        anthology = defaultdict(list)
        anthology_bibs = defaultdict(list)

        for cid, paper in tqdm.tqdm(fulltext.items()):
            for _, bib in paper['bib_entries'].items():
                key = bib_to_key(bib)

                if key:
                    anthology[key].append(cid)
                    anthology_bibs[key].append(bib)

        # Drop rare citations
        anthology_keys = list(anthology.keys())
        for key in anthology_keys:
            if len(set(anthology[key])) < MIN_CITATIONS:
                del anthology[key]
                del anthology_bibs[key]
            else:
                unique_bibs = list_of_unique_dicts(anthology_bibs[key])
                anthology[key] = (anthology[key], unique_bibs)

        return anthology

    def normalize_citations(self, fulltext):
        print_message("#> Normalize all citations in the text..")

        papers = {}

        for cid, paper in tqdm.tqdm(fulltext.items()):
            sections = defaultdict(list)
            bibs = paper['bib_entries']
            passages = []

            for p in paper['body_text']:
                heading = p['section'].lower()
                sections[heading].append(p)

            for heading, section in sections.items():
                section_psgs = self.fix_citations_in_section(heading, section, bibs)
                passages.extend(section_psgs)

            papers[cid] = {'raw': paper, 'passages': passages}

        return papers

    def fix_citations_in_section(self, heading, section, bibs):
        anthology = self.anthology
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
                        p_citations.append((key, citation, refid))

            more_text = text[offset:]

            # TODO: These would improve quality, but not currently done!
            # more_text = more_text.replace('{', ' ')
            # more_text = more_text.replace('}', ' ')
            new_text.append(more_text)

            new_text = ' '.join(new_text)

            psg = {}
            psg['heading'] = heading
            psg['text'] = new_text
            psg['raw'] = p
            psg['citations'] = p_citations

            passages.append(psg)

        return passages


def main(args):
    print_message("#> Starting..")

    builder = CorpusBuilder(args.fulltext)
    anthology, papers = builder.run()

    os.makedirs(args.output)

    with open(os.path.join(args.output, 'anthology.json'), 'w') as f:
        print_message(f"#> Writing to {f.name}...")
        ujson.dump(anthology, f)

    with open(os.path.join(args.output, 'papers.json'), 'w') as f:
        print_message(f"#> Writing to {f.name}...")
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
