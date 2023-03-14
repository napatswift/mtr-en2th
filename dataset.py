import pandas as pd
import random

def write_file(split, pair_list):
  dataset = dict(en=list(), th=list())
  for en, th in pair_list:
    dataset['en'].append(en)
    dataset['th'].append(th)
  
  print(split, 'total:', len(pair_list))

  pd.DataFrame(dataset,).to_csv(f'dataset/translate_{split}.csv', index=False)

def _manootchecklist_data():
  csv_path = 'dataset/translate.csv'
  ds_df = pd.read_csv(csv_path)
  ds_df = ds_df[ds_df['Matched?'] & (
      ds_df.Thai != '<song title>') & (~ds_df.Thai.isna())]
  ds_df.loc[:, 'Thai'] = ds_df.Thai.astype(str)
  ds_df.loc[:, 'English'] = ds_df.English.astype(
      str).apply(lambda x: x.lower())
  text_pairs = ds_df[['English', 'Thai']].values.tolist()
  random.shuffle(text_pairs)
  num_val_samples = int(0.15 * len(text_pairs))
  num_train_samples = len(text_pairs) - 2 * num_val_samples
  train_pairs = text_pairs[:num_train_samples]
  val_pairs = text_pairs[num_train_samples: num_train_samples + num_val_samples]
  test_pairs = text_pairs[num_train_samples + num_val_samples:]
  return train_pairs, val_pairs, test_pairs

def _scb_mt_en_th_data():
  csv_path = 'dataset/en-th.merged_stratified.train.csv'
  ds_df = pd.read_csv(csv_path, index_col=0)
  text_pairs = ds_df.loc[:100_000, ['en', 'th']].values.tolist()
  return text_pairs, [], []

train_pairs = []
val_pairs   = []
test_pairs  = []

_train_pairs, _val_pairs, _test_pairs = _manootchecklist_data()
train_pairs += _train_pairs
val_pairs   += _val_pairs
test_pairs  += _test_pairs

_train_pairs, _val_pairs, _test_pairs = _scb_mt_en_th_data()
train_pairs += _train_pairs
val_pairs   += _val_pairs
test_pairs  += _test_pairs


write_file('train', train_pairs)
write_file('val', val_pairs)
write_file('test', test_pairs)
write_file('all', train_pairs+test_pairs+val_pairs)