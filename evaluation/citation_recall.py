import ujson

from argparse import ArgumentParser
from collections import defaultdict

def main(args):
    GoldCitations_by_paperID = defaultdict(list)
    GoldCitations_by_QID = defaultdict(list)
    Ranking_by_QID = defaultdict(list)

    CollectionCitations = {}

    with open(args.collection_jsonl) as f:
        for line in f:
            if len(line.strip()):
                line = ujson.loads(line)
                pid, citations = line['pid'], line['citations']
                CollectionCitations[pid] = [cite for _, cite, _ in citations]

    with open(args.slice_jsonl) as f:
        for line_idx, line in enumerate(f):
            line = ujson.loads(line)

            for p in line['background']:
                for _, citation_text, _ in p['citations']:
                    GoldCitations_by_paperID[line_idx].append(citation_text)

    for idx in GoldCitations_by_paperID:
        GoldCitations_by_paperID[idx] = list(set(GoldCitations_by_paperID[idx]))

    with open(args.slice_jsonl) as f:
        current_qid = 0

        for line_idx, line in enumerate(f):
            line = ujson.loads(line)
            gold_citations = GoldCitations_by_paperID[line_idx]

            for p in line['background']:
                GoldCitations_by_QID[current_qid] = gold_citations
                current_qid += 1

    with open(args.ranking_passages) as f:
        # assert f.readline().strip() == '\t'.join(['qid', 'pid', 'rank', 'score', 'text', 'title', 'cid'])

        for line in f:
            # qid, pid, rank, score, text, title, cid = line.strip().split('\t')
            qid, pid, *_ = line.strip().split('\t')
            qid, pid = int(qid), int(pid)

            Ranking_by_QID[qid].append(pid)
            # Ranking_by_QID[qid].append(text)
            
    assert GoldCitations_by_QID.keys() == Ranking_by_QID.keys(), (len(GoldCitations_by_QID), len(Ranking_by_QID))

    for cutoff in [10, 20, 50, 100]:
        Recall = []

        for qid, gold in GoldCitations_by_QID.items():
            ranking = Ranking_by_QID[qid]
            citations_seen_so_far = 0
            recalled = [False for _ in gold]

            # for text in ranking:
            #     if text.count('{') != text.count('}'):
            #         print('\n\n\n')
            #         print(text.count('{'), text.count('}'))
            #         print(text)
            #         print('\n\n\n')
            #     citations_seen_so_far += text.count('{')

            for pid in ranking:
                pid_citations = CollectionCitations[pid]

                for idx, gold_citation in enumerate(gold):
                    recalled[idx] = recalled[idx] or (gold_citation in pid_citations)

                citations_seen_so_far += len(pid_citations)
                if citations_seen_so_far >= cutoff:
                    break
            
            Recall.append(sum(recalled) / len(recalled))
        
        recall = sum(Recall) / len(Recall)
        recall = round(100.0 * recall, 3)

        print(f"CitationRecall@{cutoff} = {recall}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description=".")

    # Input Arguments.
    parser.add_argument('--collection-jsonl', dest='collection_jsonl', required=True)
    parser.add_argument('--slice-jsonl', dest='slice_jsonl', required=True, type=str)
    parser.add_argument('--ranking', dest='ranking', required=True)

    args = parser.parse_args()

    main(args)
