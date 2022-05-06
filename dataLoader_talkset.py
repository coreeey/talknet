import os, torch, numpy, cv2, random, glob, python_speech_features
from scipy.io import wavfile
from torchvision.transforms import RandomCrop
from albumentations import (Compose, Rotate, VerticalFlip, RandomCrop, OneOf)
def generate_audio_set(dataPath, batchList):
    audioSet = {}
    for line in batchList:
        data = line.split(' ')
        audioName = data[3]
        # videoName = data[0][:11]
        dataName = data[1]
        _, audio = wavfile.read(os.path.join(dataPath, audioName + '.wav'))
        audioSet[dataName] = audio
    return audioSet

def overlap(dataName, audio, audioSet):   
    noiseName =  random.sample(set(list(audioSet.keys())) - {dataName}, 1)[0]
    noiseAudio = audioSet[noiseName]    
    snr = [random.uniform(-5, 5)]
    if len(noiseAudio) < len(audio):
        shortage = len(audio) - len(noiseAudio)
        noiseAudio = numpy.pad(noiseAudio, (0, shortage), 'wrap')
    else:
        noiseAudio = noiseAudio[:len(audio)]
    noiseDB = 10 * numpy.log10(numpy.mean(abs(noiseAudio ** 2)) + 1e-4)
    cleanDB = 10 * numpy.log10(numpy.mean(abs(audio ** 2)) + 1e-4)
    noiseAudio = numpy.sqrt(10 ** ((cleanDB - noiseDB - snr) / 10)) * noiseAudio
    audio = audio + noiseAudio    
    return audio.astype(numpy.int16)


def load_visual_labels(data_path, class_, visualAug=False):
    labels = []
    faces = []

    H = 112
    if visualAug == True:
        new = int(H*random.uniform(0.7, 1))
        x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
        M = cv2.getRotationMatrix2D((H/2,H/2), random.uniform(-15, 15), 1)
        augType = random.choice(['orig', 'flip', 'crop', 'rotate'])
    else:
        augType = 'orig'

    video = cv2.VideoCapture(data_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    # cot_frame = 0
    while video.isOpened():
        ret, frames = video.read()
        if ret == True :
            face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (224, 224))
            face = face[int(112 - (112 / 2)):int(112 + (112 / 2)), int(112 - (112 / 2)):int(112 + (112 / 2))]
            if augType == 'orig':
                faces.append(face)
            elif augType == 'flip':
                faces.append(cv2.flip(face, 1))
            elif augType == 'crop':
                faces.append(cv2.resize(face[y:y + new, x:x + new], (H, H)))
            elif augType == 'rotate':
                faces.append(cv2.warpAffine(face, M, (H, H)))

            if class_ == 'TAudio':
                labels.append(1)
            else:
                labels.append(0)
        else:
            break
    video.release()
    faces = numpy.array(faces)
    labels = numpy.array(labels)

    return faces, labels, num_frames, fps

def dynamic_batch(visualFeatures, audioFeatures, labels, length):
    dynamic_bs = random.choice([1, 2, 4, 8, 16])
    cut = length * 25 // dynamic_bs
    visual_, audio_, labels_ = [], [], []
    for i in range(dynamic_bs):
        visual_.append(visualFeatures[int(i * cut): int((i + 1) * cut), ...])  # .unsqueeze(0))
        audio_.append(audioFeatures[int(i * cut * 4): int((i + 1) * cut * 4), ...])  # .unsqueeze(0))
        labels_.append(labels[int(i * cut): int((i + 1) * cut), ...])  # .unsqueeze(0))

    return visual_, audio_, labels_



class train_loader(object):
    def __init__(self, trialFileName, data_path):#, batchSize, **kwargs):
        # self.audioPath  = audioPath
        # self.visualPath = visualPath
        self.miniBatch = []
        self.data_path = data_path
        self.mixLst = open(trialFileName).read().splitlines()


    def __getitem__(self, index):
        print(self.mixLst[index])
        class_, audio_data, video_data = self.mixLst[index].split(' ')[:3]
        audio1, audio_2, audio_3 = audio_data.split('/')
        video1, video_2, video_3 = video_data.split('/')
        data_name = audio1 + '_' + audio_2 + '_' + audio_3 + '_' + video1 + '_' + video_2 + '_' + video_3

        audio_path = os.path.join(self.data_path, class_, audio1, data_name + '.wav')
        video_data = os.path.join(self.data_path, class_, audio1, data_name + '.mp4')
        #load visual and labels
        visualFeatures, labels, frames, fps = load_visual_labels(video_data, class_, True)
        #load audio
        _, audio = wavfile.read(audio_path)
        audioFeatures = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025 * 25 / fps, winstep=0.010 * 25 / fps)
        #align the feature size
        length = min((audioFeatures.shape[0] - audioFeatures.shape[0] % 4) / 100, visualFeatures.shape[0])
        visualFeatures = visualFeatures[:int(round(length * 25)), :, :]
        labels = labels[:int(round(length * 25))]  # , :]
        audioFeatures = audioFeatures[:labels.shape[0] * 4, :]
        #dynamic batch size
        print('dataloadernumpysize', audioFeatures.shape, visualFeatures.shape, labels.shape)
        visualFeatures, audioFeatures, labels = dynamic_batch(visualFeatures, audioFeatures, labels, length)

        audioFeatures = torch.FloatTensor(numpy.array(audioFeatures))
        visualFeatures = torch.FloatTensor(numpy.array(visualFeatures))
        labels = torch.LongTensor(numpy.array(labels))
        print('dataloadersize', audioFeatures.size(), visualFeatures.size(), labels.size())
        return audioFeatures, visualFeatures, labels


    def __len__(self):
        return len(self.mixLst)


class val_loader(object):
    def __init__(self, trialFileName, data_path):#, batchSize, **kwargs):
        # self.audioPath  = audioPath
        # self.visualPath = visualPath
        self.miniBatch = []
        self.data_path = data_path
        self.mixLst = open(trialFileName).read().splitlines()
        # sort the training set by the length of the videos, shuffle them to make more videos in the same batch belong to different movies
        # sortedMixLst = sorted(mixLst, key=lambda data: (int(data.split('\t')[1]), int(data.split('\t')[-1])), reverse=True)
        # sortedMixLst = sorted(mixLst, key=lambda data: (float(data.split(' ')[3]), int(data.split(' ')[-1])), reverse=True)

        # sortedMixLst = mixLst
        start = 0
        # print(mixLst)
        # while True:
        #   length = float(sortedMixLst[start].split(' ')[3])
        #   end = min(len(sortedMixLst), start + max(int(batchSize / length), 1))
        #   self.miniBatch.append(sortedMixLst[start:end])
        #   if end == len(sortedMixLst):
        #       break
        #   start = end

        # while True:
        #   length = int(sortedMixLst[start].split('\t')[1])
        #   end = min(len(sortedMixLst), start + max(int(batchSize / length), 1))
        #   self.miniBatch.append(sortedMixLst[start:end])
        #   if end == len(sortedMixLst):
        #       break
        #   start = end

    def __getitem__(self, index):
        print(self.mixLst[index])
        class_, audio_data, video_data = self.mixLst[index].split(' ')[:3]
        audio1, audio_2, audio_3 = audio_data.split('/')
        video1, video_2, video_3 = video_data.split('/')
        data_name = audio1 + '_' + audio_2 + '_' + audio_3 + '_' + video1 + '_' + video_2 + '_' + video_3

        audio_path = os.path.join(self.data_path, class_, audio1, data_name + '.wav')
        video_data = os.path.join(self.data_path, class_, audio1, data_name + '.mp4')

        visualFeatures, labels = [], []

        # python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)

        video = cv2.VideoCapture(video_data)
        while video.isOpened():
            ret, frames = video.read()
            if ret == True:
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224, 224))
                face = face[int(112 - (112 / 2)):int(112 + (112 / 2)), int(112 - (112 / 2)):int(112 + (112 / 2))]
                visualFeatures.append(face)
                if class_ == 'TAudio':
                    labels.append(1)
                else:
                    labels.append(0)
            else:
                break
        fps = video.get(cv2.CAP_PROP_FPS)
        frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        video.release()
        visualFeatures = numpy.array(visualFeatures)
        labels = numpy.array(labels)
        _, audio = wavfile.read(audio_path)
        audioFeatures = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025 * 25 / fps, winstep=0.010 * 25 / fps)

        length = min((audioFeatures.shape[0] - audioFeatures.shape[0] % 4) / 100, visualFeatures.shape[0])
        visualFeatures = visualFeatures[:int(round(length * 25)), :, :]
        labels = labels[:int(round(length * 25))]  # , :]
        audioFeatures = audioFeatures[:labels.shape[0] * 4, :]

        # audioSet = generate_audio_set(self.audioPath, batchList) # load the audios in this batch to do augmentation


        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels))

    def __len__(self):
        return len(self.mixLst)