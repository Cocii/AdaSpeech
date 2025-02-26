# girl /data/LibriTTS_R/train-other-500/1051/133883/1051_133883_000004_000001.wav
# girl /data/LibriTTS_R/train-other-500/52/123202/52_123202_000004_000000.wav
# boy /data/LibriTTS_R/train-other-500/1094/157768/1094_157768_000001_000001.wav
# boy /data/LibriTTS_R/train-other-500/1403/135907/1403_135907_000002_000001.wav
# man31="/data/LibriTTS_R/train-other-500/31/121970/31_121970_000004_000001.wav"
# woman20="/data/LibriTTS_R/train-other-500/20/205/20_205_000002_000002.wav"
# woman29="/data/LibriTTS_R/train-other-500/29/123027/29_123027_000001_000001.wav"
# man46="/data/LibriTTS_R/train-other-500/46/127996/46_127996_000000_000001.wav"
# woman47="/data/LibriTTS_R/train-other-500/47/122796/47_122796_000001_000000.wav" # nice
# yaemiko="/data/dataset/YaeMiko/vo_DQAQ104_1_yaeMiko_35.wav"

gorou="ref_audios/gorou_origin.wav"
g_V='/Users/coc/Downloads/jiayan_code/bigvgan/g_V'
g_VR='/Users/coc/Downloads/jiayan_code/bigvgan/g_VR'
g_VRT='/Users/coc/Downloads/jiayan_code/bigvgan/g_VRT'
vocoder=$g_VRT
ref_audio_path=$gorou
speaker_id="Gorou"
CUDA_VISIBLE_DEVICES=0 python3 inference.py \
    --language_id "0" \
    --speaker_id $speaker_id \
    --reference_audio $ref_audio_path \
    --text "$(cat test.txt)" \
    -p config/en_cn_spk_gan/preprocess.yaml \
    -m config/en_cn_spk_gan/model.yaml \
    -t config/en_cn_spk_gan/train.yaml \
    --restore_step 96000 \
    --vocoder_checkpoint $vocoder \
    --vocoder_config BigVGAN/config_large.json