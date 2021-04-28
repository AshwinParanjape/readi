import argparse
import torch

parser = argparse.ArgumentParser(description='Script to extract p (and q) scorers from training checkpoint in a format usable for colbert')
parser.add_argument('--checkpoint', type=str, help="MarginalizedLoss or ELBOloss training checkpoint")

args = parser.parse_args()
ckpt = torch.load(args.checkpoint, map_location=torch.device('cpu'))
state_dict = {('module.'+k[9:]):v for k, v in ckpt['state_dict'].items() if k.startswith('p_scorer')}
assert len(state_dict) > 0, "No keys starting with p_scorer in the state_dict; aborting"
torch.save({'model_state_dict': state_dict}, args.checkpoint+'.p_scorer')

state_dict = {('module.'+k[9:]):v for k, v in ckpt['state_dict'].items() if k.startswith('q_scorer')}
if len(state_dict) <= 0:
    print("No keys starting with q_scorer in the state_dict; but that's fine if the checkpoint was for Marginalized Loss")
else:
    torch.save({'model_state_dict': state_dict}, args.checkpoint+'.q_scorer')


