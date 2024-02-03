# BigVGAN mel self
# CUDA_VISIBLE_DEVICES=0 python /workspace/nartts/AdaSpeech/vocoder_from_mel.py --mel_path "/workspace/nartts/AdaSpeech/output/output_en_cn/result/adaspeech_en_cn/mel/1162_Down! Shit, Man! 我是 Gangster trump, You know what I'm saying? Fuck! You're son of a bitch! Gang, gan_5.npy" -p /workspace/nartts/AdaSpeech/config/en_cn_spkr/preprocess.yaml -m /workspace/nartts/AdaSpeech/config/en_cn_spkr/model.yaml -t /workspace/nartts/AdaSpeech/config/en_cn_spkr/train.yaml --vocoder_checkpoint /workspace/nartts/AdaSpeech/BigVGAN/g_05000000.zip --vocoder_config /workspace/nartts/AdaSpeech/BigVGAN/config.json


# original mel
# (1, 80, 843)
# "/workspace/nartts/AdaSpeech/output/output_en_cn/result/adaspeech_en_cn/mel/275_我们在Madrid发现了一个叫johnny的programmer, 他很爱programming，然后发布到internet上，让people from all of the world来各种refe.npy"

# [1, 222, 80]
# "/data/speech_data/libri_cctv_vocoder_fintune/tests/0047_122796_000148_000000.npy"

CUDA_VISIBLE_DEVICES=1 python /workspace/nartts/AdaSpeech/BigVGAN/vocoder_from_mel.py --mel_path "/data/speech_data/libri_cctv_vocoder_fintune/tests/0428_125879_000038_000001.npy" -p /workspace/nartts/AdaSpeech/config/en_cn_spkr/preprocess.yaml -m /workspace/nartts/AdaSpeech/config/en_cn_spkr/model.yaml -t /workspace/nartts/AdaSpeech/config/en_cn_spkr/train.yaml --vocoder_checkpoint /workspace/nartts/AdaSpeech/BigVGAN/g_05000000_large.zip --vocoder_config /workspace/nartts/AdaSpeech/BigVGAN/config_large.json
