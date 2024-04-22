# hifigan
# --vocoder_checkpoint /workspace/nartts/AdaSpeech/hifigan/g_finetune_02695000.pth.tar --vocoder_config /workspace/nartts/AdaSpeech/hifigan/config.json 

# bigvgan 
# --vocoder_checkpoint /workspace/nartts/AdaSpeech/BigVGAN/g_05000000_large.zip --vocoder_config /workspace/nartts/AdaSpeech/BigVGAN/config_large.json 

# man en 85
# --reference_audio "/data/speech_data/LibriTTS_R/train-other-500/85/121551/85_121551_000006_000000.wav"

# man en 1844
# --reference_audio "/data/speech_data/LibriTTS_R/train-other-500/1844/145713/1844_145713_000002_000000.wav"

# man en 1868
# --reference_audio "/data/speech_data/LibriTTS_R/train-other-500/1868/145473/1868_145473_000008_000001.wav"

# woman en 0052  perfect 
# --reference_embedding "/data/speech_data/ref_audios/genshin/vo-spk-XMAQ305_11_dehya_08.npy"
# --reference_audio "/data/speech_data/LibriTTS_R/train-other-500/92/6488/92_6488_000000_000000.wav"
# --reference_embedding "/data/speech_data/ref_audios/genshin/vo-spk-XMAQ305_8_dunyarzad_04.npy"
# --reference_audio "/data/speech_data/LibriTTS_R/train-other-500/52/121057/52_121057_000036_000004.wav"

# woman perfect 0366
# --reference_audio "/data/speech_data/LibriTTS_R/train-other-500/366/127793/366_127793_000002_000002.wav"

# woman cn  2021  1162
# --reference_audio "/data/speech_data/ref_audios/2021_42ff8b4b99ad41afa83c1cc418485b30_2000_h264_1872_aac_128_s14.wav"

# batch mode 
# CUDA_VISIBLE_DEVICES=0 python /workspace/nartts/AdaSpeech/mel-inference.py --mode "batch" --source "/workspace/nartts/AdaSpeech/test.txt" -p /workspace/nartts/AdaSpeech/config/en_cn_spkr/preprocess.yaml -m /workspace/nartts/AdaSpeech/config/en_cn_spkr/model.yaml -t /workspace/nartts/AdaSpeech/config/en_cn_spkr/train.yaml --vocoder_checkpoint /workspace/nartts/AdaSpeech/BigVGAN/g_05000000_large.zip --vocoder_config /workspace/nartts/AdaSpeech/BigVGAN/config_large.json --restore_step 162000

# trump  finetune 
# CUDA_VISIBLE_DEVICES=1 python /workspace/nartts/AdaSpeech/inference.py --language_id "1" --speaker_id "trump_trump" --reference_audio "/data/speech_data/ref_audios/Trump_WEF_2018/0-98/en/trump_3.wav" --text "$(cat inference.txt)" -p /workspace/nartts/AdaSpeech/config/finetune_trump/preprocess.yaml -m /workspace/nartts/AdaSpeech/config/finetune_trump/model.yaml -t /workspace/nartts/AdaSpeech/config/finetune_trump/train.yaml --vocoder_checkpoint /workspace/nartts/AdaSpeech/hifigan/g_finetune_02545000.pth.tar --vocoder_config /workspace/nartts/AdaSpeech/hifigan/config.json  --restore_step 1400 --duration_control 1.0 --pitch_control 1.0


# new xlsr inference
# CUDA_VISIBLE_DEVICES=0 python /workspace/nartts/AdaSpeech/inference.py --language_id "1" --speaker_id "2020" --reference_audio "/data/speech_data/cctv_cjy/2020/10/11/part14/8dff52b19821440fae08aea946dd8994_2000_h264_1872_aac_128_s09.wav" --reference_embedding '/workspace/nartts/AdaSpeech/preprocessed_data/cctv_212018_yw/xlsr/2020-xlsr-8dff52b19821440fae08aea946dd8994_2000_h264_1872_aac_128_s09.npy' --text "$(cat /workspace/nartts/AdaSpeech/test.txt)" -p /workspace/nartts/AdaSpeech/config/en_cn_xlsr/preprocess.yaml -m /workspace/nartts/AdaSpeech/config/en_cn_xlsr/model.yaml -t /workspace/nartts/AdaSpeech/config/en_cn_xlsr/train.yaml --vocoder_checkpoint /data/speech_data/cuijiayan/checkpoint/hifigan/g_finetune_02695000.pth.tar --vocoder_config /workspace/nartts/AdaSpeech/hifigan/config.json  --restore_step 88000



# orginal pyannote spk_emb inference
# CUDA_VISIBLE_DEVICES=1 python /workspace/nartts/AdaSpeech/inference.py --language_id "0" --speaker_id "2018" --reference_audio "/data/speech_data/Yuanshen/wav/VO_COOP/VO_gorou/vo_LYYCOP001_1905509_gorou_03b.wav" --text "$(cat /workspace/nartts/AdaSpeech/test.txt)" -p /workspace/nartts/AdaSpeech/config/en_cn_spk_1221/preprocess.yaml -m /workspace/nartts/AdaSpeech/config/en_cn_spk_1221/model.yaml -t /workspace/nartts/AdaSpeech/config/en_cn_spk_1221/train.yaml --vocoder_checkpoint /data/speech_data/cuijiayan/checkpoint/hifigan/g_finetune_02695000.pth.tar --vocoder_config /workspace/nartts/AdaSpeech/hifigan/config.json  --restore_step 116000

# GAN pyannote spk_emb inference
# CUDA_VISIBLE_DEVICES=1 python /workspace/nartts/AdaSpeech/inference.py --language_id "1" --speaker_id "Gorou" --reference_audio "/data/speech_data/Yuanshen/wav/VO_COOP/VO_gorou/vo_LYYCOP001_1905509_gorou_03b.wav" --text "$(cat /workspace/nartts/AdaSpeech/test.txt)" -p /workspace/nartts/AdaSpeech/config/en_cn_spk_gan/preprocess.yaml -m /workspace/nartts/AdaSpeech/config/en_cn_spk_gan/model.yaml -t /workspace/nartts/AdaSpeech/config/en_cn_spk_gan/train.yaml --vocoder_checkpoint /data/speech_data/cuijiayan/checkpoint/hifigan/g_finetune_02695000.pth.tar --vocoder_config /workspace/nartts/AdaSpeech/hifigan/config.json  --restore_step 44000

# 女生 /data/LibriTTS_R/train-other-500/1051/133883/1051_133883_000004_000001.wav
# 女生 /data/LibriTTS_R/train-other-500/52/123202/52_123202_000004_000000.wav
# 男生 /data/LibriTTS_R/train-other-500/1094/157768/1094_157768_000001_000001.wav
# 男生 /data/LibriTTS_R/train-other-500/1403/135907/1403_135907_000002_000001.wav
# /workspace/AdaSpeech/output/output_spk_gan/result/adaspeech_spk_gan

# CUDA_VISIBLE_DEVICES=0 python /workspace/AdaSpeech/inference.py --language_id "0" --speaker_id "0052" --reference_audio "/data/LibriTTS_R/train-other-500/52/123202/52_123202_000004_000000.wav" --text "$(cat /workspace/AdaSpeech/test.txt)" -p /workspace/AdaSpeech/config/en_cn_spk_gan/preprocess.yaml -m /workspace/AdaSpeech/config/en_cn_spk_gan/model.yaml -t /workspace/AdaSpeech/config/en_cn_spk_gan/train.yaml --vocoder_checkpoint /workspace/AdaSpeech/hifigan/g_finetune_02695000.pth.tar --vocoder_config /workspace/AdaSpeech/hifigan/config.json  --restore_step 96000

# CUDA_VISIBLE_DEVICES=0 python /workspace/AdaSpeech/inference.py --language_id "0" --speaker_id "1051" --reference_audio "/data/LibriTTS_R/train-other-500/1051/133883/1051_133883_000004_000001.wav" --text "$(cat /workspace/AdaSpeech/test.txt)" -p /workspace/AdaSpeech/config/en_cn_spk_gan/preprocess.yaml -m /workspace/AdaSpeech/config/en_cn_spk_gan/model.yaml -t /workspace/AdaSpeech/config/en_cn_spk_gan/train.yaml --vocoder_checkpoint /workspace/AdaSpeech/hifigan/g_finetune_02695000.pth.tar --vocoder_config /workspace/AdaSpeech/hifigan/config.json  --restore_step 96000

# CUDA_VISIBLE_DEVICES=0 python /workspace/AdaSpeech/inference.py --language_id "0" --speaker_id "1094" --reference_audio "/data/LibriTTS_R/train-other-500/1094/157768/1094_157768_000001_000001.wav" --text "$(cat /workspace/AdaSpeech/test.txt)" -p /workspace/AdaSpeech/config/en_cn_spk_gan/preprocess.yaml -m /workspace/AdaSpeech/config/en_cn_spk_gan/model.yaml -t /workspace/AdaSpeech/config/en_cn_spk_gan/train.yaml --vocoder_checkpoint /workspace/AdaSpeech/hifigan/g_finetune_02695000.pth.tar --vocoder_config /workspace/AdaSpeech/hifigan/config.json  --restore_step 96000




man31="/data/LibriTTS_R/train-other-500/31/121970/31_121970_000004_000001.wav"
# woman20="/data/LibriTTS_R/train-other-500/20/205/20_205_000002_000002.wav"
woman29="/data/LibriTTS_R/train-other-500/29/123027/29_123027_000001_000001.wav"
man46="/data/LibriTTS_R/train-other-500/46/127996/46_127996_000000_000001.wav"
woman47="/data/LibriTTS_R/train-other-500/47/122796/47_122796_000001_000000.wav" # nice
yaemiko="/data/dataset/YaeMiko/vo_DQAQ104_1_yaeMiko_35.wav"
gorou="/data/dataset/Gorou/vo_dialog_DQAQ010_gorou_03.wav"

baseG="/workspace/BigVGAN/BigVGAN/bigvgan_22khz_80band/fakeMel_realAudio_finetune/g_05700000"
testG="/workspace/BigVGAN/BigVGAN/bigvgan_22khz_80band/RealFakeAduio_rawnet_finetune/generator/g_09000000"
testG1='/workspace/BigVGAN/BigVGAN/bigvgan_22khz_80band/detect_experiment1/generator/g_06450000'

ref_audio_path=$woman47
speaker_id=$(echo "$ref_audio_path" | cut -d'/' -f5)
speaker_id=$(printf "%04d" "$speaker_id")

# ref_audio_path=$yaemiko
# speaker_id=$(echo "$ref_audio_path" | cut -d'/' -f4)

# CUDA_VISIBLE_DEVICES=0 python /workspace/AdaSpeech/inference.py \
#     --language_id "0" \
#     --speaker_id $speaker_id \
#     --reference_audio $ref_audio_path \
#     --text "$(cat /workspace/AdaSpeech/test.txt)" \
#     -p /workspace/AdaSpeech/config/en_cn_spk_gan/preprocess.yaml \
#     -m /workspace/AdaSpeech/config/en_cn_spk_gan/model.yaml \
#     -t /workspace/AdaSpeech/config/en_cn_spk_gan/train.yaml \
#     --restore_step 96000 \
#     --vocoder_checkpoint $baseG \
#     --vocoder_config /workspace/AdaSpeech/bigvgan_22khz_80band/config.json

CUDA_VISIBLE_DEVICES=0 python /workspace/AdaSpeech/inference.py \
    --language_id "0" \
    --speaker_id $speaker_id \
    --reference_audio $ref_audio_path \
    --text "$(cat /workspace/AdaSpeech/test.txt)" \
    -p /workspace/AdaSpeech/config/en_cn_spk_gan/preprocess.yaml \
    -m /workspace/AdaSpeech/config/en_cn_spk_gan/model.yaml \
    -t /workspace/AdaSpeech/config/en_cn_spk_gan/train.yaml \
    --restore_step 96000 \
    --vocoder_checkpoint $testG1 \
    --vocoder_config /workspace/AdaSpeech/bigvgan_22khz_80band/config.json

# CUDA_VISIBLE_DEVICES=0 python /workspace/AdaSpeech/inference.py \
#     --language_id "0" \
#     --speaker_id $speaker_id \
#     --reference_audio $ref_audio_path \
#     --text "$(cat /workspace/AdaSpeech/test.txt)" \
#     -p /workspace/AdaSpeech/config/en_cn_spk_gan/preprocess.yaml \
#     -m /workspace/AdaSpeech/config/en_cn_spk_gan/model.yaml \
#     -t /workspace/AdaSpeech/config/en_cn_spk_gan/train.yaml \
#     --restore_step 96000 \
#     --vocoder_checkpoint /workspace/AdaSpeech/bigvgan_22khz_80band/g_05000000 \
#     --vocoder_config /workspace/AdaSpeech/bigvgan_22khz_80band/config.json


# hifigan
# --vocoder_checkpoint /workspace/AdaSpeech/hifigan/g_finetune_02695000.pth.tar --vocoder_config /workspace/AdaSpeech/hifigan/config.json
# BigVGAN
# --vocoder_checkpoint /workspace/AdaSpeech/bigvgan_22khz_80band/g_05000000 --vocoder_config /workspace/AdaSpeech/bigvgan_22khz_80band/config.json
## fake real rawnet2 finetune
# --vocoder_checkpoint /workspace/BigVGAN/BigVGAN/bigvgan_22khz_80band/RealFakeAduio_rawnet_finetune/g_05250000 --vocoder_config /workspace/AdaSpeech/bigvgan_22khz_80band/config.json
