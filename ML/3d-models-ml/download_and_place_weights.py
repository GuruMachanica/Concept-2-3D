from huggingface_hub import hf_hub_download
import os
import shutil
repo_id = 'stabilityai/TripoSR'
filenames = ['config.yaml', 'model.ckpt']

out_dir = os.path.join(os.getcwd(), 'triposr','local_pretrained')
os.makedirs(out_dir, exist_ok=True)

for fn in filenames:
    print(f'Downloading {fn} from {repo_id}...')
    path = hf_hub_download(repo_id=repo_id, filename=fn)
    print('Downloaded to cache:', path)
    dest = os.path.join(out_dir, fn)
    print('Copying to', dest)
    shutil.copy(path, dest)
    print('Copied')

print('All done. Local pretrained dir:', out_dir)
