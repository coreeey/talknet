import os, subprocess, glob, pandas, tqdm, cv2, numpy, json
from scipy.io import wavfile
import shutil
class args():
    a = 1

def init_args(args):
    # The details for the following folders/files can be found in the annotation of the function 'preprocess_AVA' below
    # args.modelSavePath    = os.path.join(args.savePath, 'model')
    # args.scoreSavePath    = os.path.join(args.savePath, 'score.txt')
    # args.trialPathAVA     = os.path.join(args.dataPath, 'csv')
    args.trialPathAVA     = r'/mnt/lustre02/jiangsu/aispeech/home/xt415/remote/Datasets/Talkies'

    args.audioOrigPathAVA     = os.path.join(r'/mnt/lustre02/jiangsu/aispeech/home/flc23/DataSets/Talkies', 'audio')
    args.visualOrigPathAVA   = os.path.join(r'/mnt/lustre02/jiangsu/aispeech/home/flc23/DataSets/Talkies', 'clips')

    args.audioPathAVA     = os.path.join(r'/mnt/lustre02/jiangsu/aispeech/home/xt415/remote/Datasets/Talkies', 'audio_generate')
    args.visualPathAVA    = os.path.join(r'/mnt/lustre02/jiangsu/aispeech/home/xt415/remote/Datasets/Talkies', 'clips_generate')

    # args.visualPathAVA    = os.path.join(args.dataPathAVA, 'clips_videos')
    # args.trainTrialAVA    = os.path.join(args.trialPathAVA, 'train_loader.csv')

    # if args.evalDataType == 'val':
    #     # args.evalTrialAVA = os.path.join(args.trialPathAVA, 'val_loader.csv')
    #     # args.evalOrig     = os.path.join(args.trialPathAVA, 'val_orig.csv')
    #     args.evalCsvSave  = os.path.join(args.savePath,     'val_res.csv')
    # else:
    #     # args.evalTrialAVA = os.path.join(args.trialPathAVA, 'test_loader.csv')
    #     # args.evalOrig     = os.path.join(args.trialPathAVA, 'test_orig.csv')
    #     args.evalCsvSave  = os.path.join(args.savePath,     'test_res.csv')
    #
    # os.makedirs(args.modelSavePath, exist_ok = True)
    # os.makedirs(args.dataPathAVA, exist_ok = True)
    return args


def preprocess_talkies(args):
    # This preprocesstion is modified based on this [repository](https://github.com/fuankarion/active-speakers-context).
    # The required space is 302 G. 
    # If you do not have enough space, you can delate `orig_videos`(167G) when you get `clips_videos(85G)`.
    #                             also you can delate `orig_audios`(44G) when you get `clips_audios`(6.4G).
    # So the final space is less than 100G.
    # The AVA dataset will be saved in 'AVApath' folder like the following format:
    # ```
    # ├── clips_audios  (The audio clips cut from the original movies)
    # │   ├── test
    # │   ├── train
    # │   └── val
    # ├── clips_videos (The face clips cut from the original movies, be save in the image format, frame-by-frame)
    # │   ├── test
    # │   ├── train
    # │   └── val
    # ├── csv
    # │   ├── test_file_list.txt (name of the test videos)
    # │   ├── test_loader.csv (The csv file we generated to load data for testing)
    # │   ├── test_orig.csv (The combination of the given test csv files)
    # │   ├── train_loader.csv (The csv file we generated to load data for training)
    # │   ├── train_orig.csv (The combination of the given training csv files)
    # │   ├── trainval_file_list.txt (name of the train/val videos)
    # │   ├── val_loader.csv (The csv file we generated to load data for validation)
    # │   └── val_orig.csv (The combination of the given validation csv files)
    # ├── orig_audios (The original audios from the movies)
    # │   ├── test
    # │   └── trainval
    # └── orig_videos (The original movies)
    #     ├── test
    #     └── trainval
    # ```

    # download_csv(args) # Take 1 minute
    # exit(0)
    # download_videos(args) # Take 6 hours
    # extract_audio(args) # Take 1 hour
    extract_audio_clips(args) # Take 3 minutes
    extract_video_clips(args) # Take about 2 days

# def download_csv(args):
#     # Take 1 minute to download the required csv files
#     Link = "1C1cGxPHaJAl1NQ2i7IhRgWmdvsPhBCUy"
#     cmd = "gdown --id %s -O %s"%(Link, args.dataPathAVA + '/csv.tar.gz')
#     subprocess.call(cmd, shell=True, stdout=None)
#     cmd = "tar -xzvf %s -C %s"%(args.dataPathAVA + '/csv.tar.gz', args.dataPathAVA)
#     subprocess.call(cmd, shell=True, stdout=None)
#     os.remove(args.dataPathAVA + '/csv.tar.gz')
#
# def download_videos(args):
#     # Take 6 hours to download the original movies, follow this repository: https://github.com/cvdfoundation/ava-dataset
#     for dataType in ['trainval', 'test']:
#         fileList = open('%s/%s_file_list.txt'%(args.trialPathAVA, dataType)).read().splitlines()
#         outFolder = '%s/%s'%(args.visualOrigPathAVA, dataType)
#         for fileName in fileList:
#             cmd = "wget -P %s https://s3.amazonaws.com/ava-dataset/%s/%s"%(outFolder, dataType, fileName)
#             subprocess.call(cmd, shell=True, stdout=None)
#
# def extract_audio(args):
#     # Take 1 hour to extract the audio from movies
#     for dataType in ['trainval', 'test']:
#         inpFolder = '%s/%s'%(args.visualOrigPathAVA, dataType)
#         outFolder = '%s/%s'%(args.audioOrigPathAVA, dataType)
#         os.makedirs(outFolder, exist_ok = True)
#         videos = glob.glob("%s/*"%(inpFolder))
#         for videoPath in tqdm.tqdm(videos):
#             audioPath = '%s/%s'%(outFolder, videoPath.split('/')[-1].split('.')[0] + '.wav')
#             cmd = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads 8 %s -loglevel panic" % (videoPath, audioPath))
#             subprocess.call(cmd, shell=True, stdout=None)


def extract_audio_clips(args):
    # Take 3 minutes to extract the audio clips
    dic = {'train':'train', 'val':'val'}
    for dataType in ['train']:
        df = pandas.read_csv(os.path.join(args.trialPathAVA, 'talkies_%s_check.csv'%(dataType)), engine='python')
        df_ = df
        video_name_ls = df['clip_id'].unique().tolist()
        print(video_name_ls)
        for i, video_id in enumerate(video_name_ls):
            df = df[df['clip_id'] == video_id]
            dfNeg = df[df['label'] == 0]
            dfPos = df[df['label'] == 1]
            insNeg = dfNeg['speaker_id'].unique().tolist()
            insPos = dfPos['speaker_id'].unique().tolist()
            df = pandas.concat([dfPos, dfNeg]).reset_index(drop=True)
            df = df.sort_values(['id', 'timestamp']).reset_index(drop=True)
            entityList = df['id'].unique().tolist()
            df = df.groupby('id')
            audioFeatures = {}
            outDir = os.path.join(args.audioPathAVA, dataType)
            audioDir = args.audioOrigPathAVA
            for l in df['clip_id'].unique().tolist():
                d = os.path.join(outDir, l[0])
                if not os.path.isdir(d):
                    os.makedirs(d)
            for entity in tqdm.tqdm(entityList, total = len(entityList)):
                insData = df.get_group(entity)
                videoKey = insData.iloc[0]['clip_id']

                start = insData.iloc[0]['timestamp']
                end = insData.iloc[-1]['timestamp']
                entityID = insData.iloc[0]['id']
                insPath = os.path.join(outDir, videoKey, 'id_' + entityID.split()[-1]+'.wav')
                # print(insPath)
                if videoKey not in audioFeatures.keys():
                    audioFile = os.path.join(audioDir, videoKey+'.wav')
                    try:
                        sr, audio = wavfile.read(audioFile)
                    except:
                        with open('problem_train.txt', 'a+') as f:
                            f.writelines(['audio__', audioFile])
                        continue
                    audioFeatures[videoKey] = audio
                audioStart = int(float(start)*sr)
                audioEnd = int(float(end)*sr)
                audioData = audioFeatures[videoKey][audioStart:audioEnd]
                wavfile.write(insPath, sr, audioData)
            df = df_
'''
clip_id,clip_url,clip_metadata,label,speaker_id,timestamp,id,x1,y1,x2,y2
->
video_id, _, _, label_id, entity_id, frame_timestamp, instance_id, entity_box_x1, entity_box_y1, entity_box_x2, entity_box_y2
'''
def extract_video_clips(args):
    # Take about 2 days to crop the face clips.
    # You can optimize this code to save time, while this process is one-time.
    # If you do not need the data for the test set, you can only deal with the train and val part. That will take 1 day.
    # This procession may have many warning info, you can just ignore it.
    dic = {'train':'train', 'val':'trainval'}
    for dataType in ['train']:
        df = pandas.read_csv(os.path.join(args.trialPathAVA, 'talkies_%s_check.csv'%(dataType)))

        df_ = df
        video_name_ls = df['clip_id'].unique().tolist()
        print(video_name_ls)
        for i, video_id in enumerate(video_name_ls):
            df = df[df['clip_id'] == video_id]

            # df = df[df['video_id'] == '5milLu-6bWI']

            dfNeg = df[df['label'] == 0]
            dfPos = df[df['label'] == 1]

            insNeg = dfNeg['speaker_id'].unique().tolist()
            insPos = dfPos['speaker_id'].unique().tolist()
            df = pandas.concat([dfPos, dfNeg]).reset_index(drop=True)
            df = df.sort_values(['id', 'timestamp']).reset_index(drop=True)
            entityList = df['id'].unique().tolist()
            df = df.groupby('id')
            outDir = os.path.join(args.visualPathAVA, dataType)
            for l in df['clip_id'].unique().tolist():
                # print(df['video_id'])
                # if '5milLu' not in df['video_id'][l]:
                #     continue
                d = os.path.join(outDir, l[0])
                if not os.path.isdir(d):
                    os.makedirs(d)

            for entity in tqdm.tqdm(entityList, total = len(entityList)):
                # print(entity)
                insData = df.get_group(entity)
                videoKey = insData.iloc[0]['clip_id']
                entityID = insData.iloc[0]['id']
                # videoDir = os.path.join(args.visualOrigPathAVA, dic[dataType])
                videoDir = args.visualOrigPathAVA

                videoFile = glob.glob(os.path.join(videoDir, '{}.*'.format(videoKey)))[0]
                #cat org video

                try:
                    V = cv2.VideoCapture(videoFile)
                except:
                    with open('problem_train.txt', 'a+') as f:
                        f.writelines(['video__', videoFile])
                    continue
                insDir = os.path.join(os.path.join(outDir, videoKey))#, entityID))

                # if not os.path.exists(os.path.join(insDir, f'org.mp4')):
                #     shutil.copy(videoFile, os.path.join(insDir, f'org.mp4'),)

                if not os.path.isdir(insDir):
                    os.makedirs(insDir)
                j = 0
                #generate crop video
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_video = cv2.VideoWriter(os.path.join(insDir, f'id_{entityID.split()[-1]}.mp4'), fourcc, 25.0, (112, 112),
                                            True)
                # print(videoFile)

                for _, row in insData.iterrows():
                    #video_path+id        assume each id speak one time
                    # if row['clip_id'] == videoKey:
                    # imageFilename = os.path.join(insDir, str("%.2f"%row['timestamp']) + f'{entityID.split()[-1]}'+'.jpg')
                    V.set(cv2.CAP_PROP_POS_MSEC, row['timestamp'] * 1e3)
                    # print(row, row['timestamp'] )
                    _, frame = V.read()
                    try:
                        h = numpy.size(frame, 0)
                    except:
                        with open('problem.txt', 'a+') as f:
                            f.writelines(['video', videoFile])
                        continue
                    w = numpy.size(frame, 1)
                    x1 = int(row['x1'] * w)
                    y1 = int(row['y1'] * h)
                    x2 = int(row['x2'] * w)
                    y2 = int(row['y2'] * h)
                    # face = frame[:, x1:x2, :]
                    face = frame[y1:y2, x1:x2, :]
                    face = cv2.resize(face, (112, 112))
                    out_video.write(face)
                    j = j+1
                    # cv2.imwrite(imageFilename, face)
                print('save video')
                out_video.release()
                V.release()
            df = df_

if __name__ == "__main__":
    init_args(args)
    preprocess_talkies(args)