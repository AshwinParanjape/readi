import ujson

from argparse import ArgumentParser
from collections import defaultdict
import pickle, io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def main(args):
    GoldCitations_by_paperID = defaultdict(list)
    GoldCitations_by_QID = defaultdict(list)
    Ranking_by_QID = defaultdict(list)

    CollectionCitations = {}
    Pid2Cid = {}
    Qid2Cid = {}

    with open(args.collection_jsonl) as f:
        for line in f:
            if len(line.strip()):
                line = ujson.loads(line)
                pid, citations = line['pid'], line['citations']
                CollectionCitations[pid] = [cite for _, cite, _ in citations]
                Pid2Cid[pid] = line['cid']

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
                Qid2Cid[current_qid] = line['cid']
                current_qid += 1

    
    if args.scores is not None:
        with open(args.scores) as f:
            assert f.readline()
            #assert f.readline().strip() == '\t'.join(['stage','epoch', 'q_id', 'doc_id', 'p_score']
            Scores_by_QID = defaultdict(list)

            for line in f:
                stage, epoch, qid, pid, score = line.strip().split('\t')
                if stage==args.stage:
                    qid, pid = int(qid), int(pid)

                    Scores_by_QID[qid].append((pid, score))
                # Ranking_by_QID[qid].append(text)
        for qid, pids_scores in Scores_by_QID.items():
            pids_scores = sorted(pids_scores, key=lambda p: p[1], reverse=True)
            pids = [p for p, s in pids_scores]
            Ranking_by_QID[qid] = pids

    if args.rescored_pkl is not None:
        with open(args.rescored_pkl, 'rb') as f:
            print(f"Loading {f.name}...")
            Ranking = CPU_Unpickler(f).load()
            Ranking_by_QID = {q['qid']: [d['doc_id'] for d in q['retrievals']] for q in Ranking}

    else:
        with open(args.ranking) as f:
            assert f.readline()
            # assert f.readline().strip() == '\t'.join(['qid', 'pid', 'rank', 'score', 'text', 'title', 'cid'])

            for line in f:
                # qid, pid, rank, score, text, title, cid = line.strip().split('\t')
                qid, pid, *_ = line.strip().split('\t')
                qid, pid = int(qid), int(pid)

                Ranking_by_QID[qid].append(pid)
                # Ranking_by_QID[qid].append(text)
            
    assert GoldCitations_by_QID.keys() == Ranking_by_QID.keys() or args.ranking is None, (len(GoldCitations_by_QID), len(Ranking_by_QID))

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
                if Qid2Cid[qid] == Pid2Cid[pid]:
                    continue

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


# FIXME: NOTE: TODO:  Taking full passage even if exceeds citation cutoff in the last passage!!!! Intentional for now.

if __name__ == "__main__":
    parser = ArgumentParser(
        description=".")

    # Input Arguments.
    parser.add_argument('--collection-jsonl', dest='collection_jsonl', required=True)
    parser.add_argument('--slice-jsonl', dest='slice_jsonl', required=True, type=str)
    parser.add_argument('--ranking', dest='ranking')
    parser.add_argument('--scores', dest='scores') 
    parser.add_argument('--rescored_pkl', dest='rescored_pkl') 
    parser.add_argument('--stage', dest='stage', type=str, default='val') 

    args = parser.parse_args()

    main(args)
