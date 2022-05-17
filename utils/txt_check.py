import tqdm
import os
from scipy.io import wavfile
mixLst = open(r'../TalkSet/lists/lists_out/train.txt').read().splitlines()
data_path = r'/mnt/lustre02/jiangsu/aispeech/home/flc23/DataSets/Talkset'

list_out_train = r'train_check.txt'
train_file = open(list_out_train, "a")
for i, line in enumerate(tqdm.tqdm(mixLst)):
    if i>67500:
        class_, audio_data, video_data = line.split(' ')[:3]
        audio1, audio_2, audio_3 = audio_data.split('/')
        video1, video_2, video_3 = video_data.split('/')
        data_name = audio1 + '_' + audio_2 + '_' + audio_3 + '_' + video1 + '_' + video_2 + '_' + video_3

        audio_path = os.path.join(data_path, class_, audio1, data_name + '.wav')
        video_path = os.path.join(data_path, class_, audio1, data_name + '.mp4')

        # print(video_path)
        # print(audio_path)
        if os.path.exists(audio_path) and os.path.exists(video_path):
            # print(line)
            print(i)
            try:
                _, _ = wavfile.read(audio_path)
            except:
                print('wrong line')
                continue

            train_file.write(line + '\r\n')


