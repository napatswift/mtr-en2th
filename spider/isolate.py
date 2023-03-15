import json
import re
import pandas
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-s', '--source', help='Input JSON file path to isolate English and Thai', required=True)
  parser.add_argument('-d', '--destination', help='Outpus CSV file path', required=True)
  args = parser.parse_args()

  m_lyrics = json.load(open(args.source))

  lyrics_pair = []
  for song in m_lyrics[:]:
      if 'แปล' not in song['title']:
          continue
      song_detail_count = 0
      lyrics = []
      lang_tag = []
      for line in song['lyrics']:
          found_detail = re.match('^[A-Za-z][A-Za-z0-9 ]*:', line) is not None
          if found_detail:
              song_detail_count += 1
          found_interlude = re.match('\[.*\]', line) is not None
          if not found_detail and song_detail_count and not found_interlude:
              lyrics.append(line)
              th_match=re.findall('[\u0E00-\u0E7F]', line)
              lang_tag.append(1 if th_match else 0)
      th_len = sum(lang_tag)
      en_len = len(lang_tag)-th_len

      lyrics_pair.append(('<song title>', f"<{song['title']}>"))

      en_lyrics = [l for l,lang in zip(lyrics,lang_tag) if lang == 0]
      if th_len > en_len:
          en_lyrics += [''] * (th_len - en_len)
          
      th_lyrics = [l for l,lang in zip(lyrics,lang_tag) if lang == 1]
      if en_len > th_len:
          th_lyrics += [''] * (en_len - th_len)
      lyrics_pair.extend(list(zip(th_lyrics,en_lyrics)))

  pandas.DataFrame(lyrics_pair).to_csv(args.destination)