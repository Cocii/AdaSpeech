import os
import numpy as np




feature = ["avg_mel_phon", "duration", "energy", "mel", "pitch", "speaker_embed"]
#           0               1            2        3       4        5


# path = "/workspace/nartts/AdaSpeech/preprocessed_data/libri_spkr/"
# a = os.path.join(path, feature[2], "0020-{}-205_000004_000002.npy".format(feature[2]))
# b = os.path.join(path, feature[4], "0020-{}-205_000004_000002.npy".format(feature[4]))



path = "/workspace/nartts/AdaSpeech/preprocessed_data/cctv_212018_yw/"
a = os.path.join(path, feature[2], "2020-{}-699b78d0ce8945ab8126ebb4c4d8339b_2000_h264_1872_aac_128_s97.npy".format(feature[2]))
b = os.path.join(path, feature[4], "2020-{}-699b78d0ce8945ab8126ebb4c4d8339b_2000_h264_1872_aac_128_s97.npy".format(feature[4]))


# path = "/workspace/nartts/AdaSpeech/preprocessed_data/libri_spkr/"
# a = os.path.join(path, feature[2], "0572-{}-128861_000004_000003.npy".format(feature[2]))
# b = os.path.join(path, feature[4], "0572-{}-128861_000004_000003.npy".format(feature[4]))


np.set_printoptions(suppress=True)
a = np.load(a)
b = np.load(b)
print("energy: ", a)
print("pitch: ", b)

# print("pitch: ", (a - np.mean(a))/np.std(a))
# print("energy: ", (b - np.mean(b))/np.std(b))

print("energy ==== min: ", np.min(a), ", max: ",np.max(a), ", mean: ", np.mean(a), "std: ", np.std(a))
print("pitch === min: ", np.min(b), ", max: ",np.max(b), ", mean: ", np.mean(b),  "std: ", np.std(b))