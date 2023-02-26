import wandb
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("tiewa_enguin/tpu_ldm_ddpm")

for run in runs:
    files = run.files()
    for file in files:
        fname = file._attrs['name']
        if 'pth' in fname:
            epoch = int(fname[-8:-4])
            if epoch < 15 or epoch % 10 != 0:
                print(f'Deleting {fname}')
                file.delete()
            else:
                print(f'Keeping {fname}')
