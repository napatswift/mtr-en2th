import pandas as pd
import sentencepiece as spm

def create_temp_file() -> dict:
  dataset_df = pd.read_csv('dataset/translate_all.csv')
  temp_files = {}
  for lang in ['th', 'en']:
    sentence_lines = dataset_df[lang].tolist()
    total_line_count = 0
    fname = f'dataset/.lines_{lang.lower()}.txt'
    temp_files[lang] = fname
    with open(fname, 'w') as f:
      string_lines = [str(l) for l in sentence_lines]
      for l in string_lines:
        f.write(l)
        f.write('\n')
        total_line_count += 1
      print('total_line_count', total_line_count)
  return temp_files

def build_spmodel(lang, fpath):
  spm.SentencePieceTrainer.train(
    input=fpath,
    model_prefix=f'spmodel/{lang}',
    vocab_size=2000,
    pad_id=3)


if __name__ == '__main__':
  for lang, fpath in create_temp_file().items():
    build_spmodel(lang, fpath)