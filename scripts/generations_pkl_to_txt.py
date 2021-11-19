from argparse import ArgumentParser
import pickle as pkl

parser = ArgumentParser( description=".")

# Input Arguments.
parser.add_argument('--generations_pkl', dest='generations_pkl', required=True)
args = parser.parse_args()

txt_path = '/'.join(args.generations_pkl.split('/')[:-1])+'/readable_generations.txt'
with open(txt_path, 'w') as fout:
    with open(args.generations_pkl, 'rb') as f:
        gens = pkl.load(f)
    for gen in gens:
        print('Conv:', ' | '.join(gen['source'].split('|')[-2:]), file=fout)
        print('target:', gen['target'], file=fout)
        if 'retrievals' in gen:
            for ret in gen['retrievals']:
                print('\t Doc:', ret['doc_score'], ret['doc_text'], file=fout)
                if 'generator_output' in ret:
                    if 'log_liklihood' in ret['generator_output'] and 'baseline_log_liklihood' in ret['generator_output']:
                        for s, ll, bll in zip(ret['generator_output']['strings'], ret['generator_output']['log_liklihood'], ret['generator_output']['baseline_log_liklihood']):
                            print('\t\t Gen output:', ll, bll, s, file=fout)
                    else:
                        for s in zip(ret['generator_output']['strings']):
                            print('\t\t Gen output:', s, file=fout)
                    print('-'*100, file=fout)
        else:
            if 'log_liklihood' in gen['generator_output'] and 'baseline_log_liklihood' in gen['generator_output']:
                for s, ll, bll in zip(gen['generator_output']['strings'], gen['generator_output']['log_liklihood'], gen['generator_output']['baseline_log_liklihood']):
                    print('\t\t Gen output:', ll, bll, s, file=fout)
            else:
                for s in zip(gen['generator_output']['strings']):
                    print('\t\t Gen output:', s, file=fout)
            print('-'*100, file=fout)
        print('='*100, file=fout)
