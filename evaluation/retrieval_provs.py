import pickle

from argparse import ArgumentParser
from collections import defaultdict

def main(args):
    Provs = {}

    with open(args.provenance) as f:
        print(f"Loading {f.name}...")
        _ = f.readline()  # Header
        for line_idx, line in enumerate(f):
            line = line.strip().split('\t')
            qid, pid, *_ = line
            qid, pid = map(int, [qid, pid])
            assert qid == line_idx, (qid, line_idx, line)

            Provs[qid] = pid

    for path in args.ranking:
        if path.endswith('.tsv'):
            Ranking = defaultdict(list)

            with open(path) as f:
                print(f"Loading {f.name}...")
                for line in f:
                    line = line.strip().split('\t')
                    qid, pid, *_ = line
                    qid, pid = map(int, [qid, pid])

                    Ranking[qid].append(pid)
            
            for qid in Ranking:
                Ranking[qid] = {'qid': qid, 'retrievals': Ranking[qid]}
                
        else:
            with open(path, 'rb') as f:
                print(f"Loading {f.name}...")
                Ranking = pickle.load(f)

        assert len(Ranking) == len(Provs), (len(Ranking), len(Provs))

        Success = {k: 0.0 for k in [1, 5, 10, 20, 50, 100]}
        MRR_100 = 0.0
        MRR_Denom = 0

        for qid, prov_pid in Provs.items():
            if prov_pid == -1:
                continue

            ranking = Ranking[qid]
            assert ranking['qid'] == qid

            MRR_Denom += 1

            # target = ranking['target']
            ranking = ranking['retrievals'][:100]
            ranking_pids = [x['doc_id'] for x in ranking]

            try:
                rank = ranking_pids.index(prov_pid)
            except ValueError:
                rank = None

            if rank is not None:
                MRR_100 += 1.0 / (rank + 1)

                for k in Success:
                    if rank < k:
                        Success[k] += 1.0

        MRR_100 = MRR_100 / MRR_Denom
        MRR_100 = round(100 * MRR_100, 2)

        for k in Success:
            Success[k] = Success[k] / MRR_Denom
            Success[k] = round(100 * Success[k], 2)
            print(f"Success@{k} = {Success[k]}%")

        print(f"MRR@100 = {MRR_100}%")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Evaluate KILT-provenance retrieval via MRR@100.")

    # Input Arguments.
    parser.add_argument('--ranking', dest='ranking', required=True, type=str, nargs='+')
    parser.add_argument('--provenance', dest='provenance', required=True)

    args = parser.parse_args()

    main(args)
