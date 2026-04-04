from huggingface_hub import hf_hub_download
import os

repo_id = 'stabilityai/TripoSR'
filenames = ['config.yaml', 'model.ckpt']

out_dir = os.path.join(os.getcwd(), 'triposr_weights')
os.makedirs(out_dir, exist_ok=True)

for fn in filenames:
    print(f'Downloading {fn} from {repo_id}...')
    path = hf_hub_download(repo_id=repo_id, filename=fn, cache_dir=out_dir)
    print('Saved to:', path)
print('Done')
