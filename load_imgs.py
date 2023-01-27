from threading import Thread
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from glob import glob
import pandas as pd
import requests, os, shutil, time

IMG_SIZE = 256
out_folder = 'dataset'
if out_folder not in os.listdir():
    os.mkdir(out_folder)
if 'imgs' not in os.listdir(out_folder):
    os.mkdir(f'{out_folder}/imgs')

df = pd.read_parquet('train-00000-of-00001-6f24a7497df494ae.parquet')
df = df[df['punsafe'] < 0.7]
df = df[df['HEIGHT'] >= IMG_SIZE]
df = df[df['WIDTH'] >= IMG_SIZE]

STOP = False
pbar = tqdm(total=len(df))
def process_df(df, t_i=0, return_arr={}):
    global pbar
    img_id = 0
    img_to_text = {}
    return_arr[t_i] = img_to_text

    thread_out = f'{t_i}'
    if thread_out not in os.listdir(out_folder):
        os.mkdir(f'{out_folder}/{thread_out}')

    #for index, row in tqdm(df.iterrows(), total=len(df)):
    for index, row in df.iterrows():
        if STOP:
            break

        pbar.update(1)
        url = row.URL
        img = None

        try:
            response = requests.get(url, timeout=5) # 5 second timeout
            img = Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            pass

        if img is None:
            continue

        # resize
        w, h = img.size
        min_dim = min(w, h)
        scale = IMG_SIZE / min_dim
        new_w, new_h = round(scale * w), round(scale * h)
        img = img.resize((new_w, new_h))

        # center crop
        w, h = img.size
        s_y = (h - IMG_SIZE) // 2
        s_x = (w - IMG_SIZE) // 2
        img = img.crop((s_x, s_y, s_x+IMG_SIZE, s_y+IMG_SIZE))
        img_filename = f'{img_id:0>9}.jpg'
        img_to_text[img_filename] = row.TEXT
        img.save(f'{out_folder}/{thread_out}/{img_filename}')
        img_id += 1

N_THREADS = 128
sub_df_len = len(df) // N_THREADS
df_remainder = len(df) % N_THREADS

id_to_text_batches = {}
ts = []
s_i = e_i = 0
try:
    for t_i in range(N_THREADS):
        s_i, e_i = e_i, sub_df_len * t_i + (1 if t_i < df_remainder else 0)
        t = Thread(target=process_df, args=(df.iloc[s_i:e_i], t_i, id_to_text_batches), daemon=True)
        ts.append(t)
        t.start()

    for t in ts:
        t.join()
except KeyboardInterrupt:
    STOP = True
    print('Keyboard interrupt, waiting 2 seconds for threads to close.')
    time.sleep(2)

id_to_text = {}
img_id = 0
for t_i, id_to_text_batch in id_to_text_batches.items():
    for img_filename, img_caption in id_to_text_batch.items():
        shutil.move(f'{out_folder}/{t_i}/{img_filename}', f'{out_folder}/imgs/{img_id:0>9}.jpg')
        id_to_text[img_id] = img_caption
        img_id += 1

id_to_text_out = open(f'{out_folder}/id_to_text.txt', 'w')
for i in range(len(id_to_text)):
    id_to_text_out.write(id_to_text[i] + '\n')
id_to_text_out.close()
os.system(f'rmdir {out_folder}/*')
