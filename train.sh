# continue
# CUDA_VISIBLE_DEVICES=0 python /workspace/nartts/AdaSpeech/train_en_cn.py -p /workspace/nartts/AdaSpeech/config/en_cn_spkr/preprocess.yaml -m /workspace/nartts/AdaSpeech/config/en_cn_spkr/model.yaml -t /workspace/nartts/AdaSpeech/config/en_cn_spkr/train.yaml --vocoder_checkpoint /workspace/nartts/AdaSpeech/BigVGAN/g_05000000_large.zip --vocoder_config /workspace/nartts/AdaSpeech/BigVGAN/config_large.json --restore_step 162000

# From   libri + cctv
CUDA_VISIBLE_DEVICES=0 python /workspace/AdaSpeech/train_en_cn.py -p /workspace/AdaSpeech/config/en_cn_spk_gan/preprocess.yaml -m /workspace/AdaSpeech/config/en_cn_spk_gan/model.yaml -t /workspace/AdaSpeech/config/en_cn_spk_gan/train.yaml --vocoder_checkpoint /workspace/AdaSpeech/hifigan/g_finetune_02695000.pth.tar --vocoder_config /workspace/AdaSpeech/hifigan/config.json --restore_step 0

# From   libri + cctv  xlsr
# CUDA_VISIBLE_DEVICES=0 python /workspace/nartts/AdaSpeech/train_en_cn.py -p /workspace/nartts/AdaSpeech/config/en_cn_spkr/preprocess.yaml -m /workspace/nartts/AdaSpeech/config/en_cn_spkr/model.yaml -t /workspace/nartts/AdaSpeech/config/en_cn_spkr/train.yaml --vocoder_checkpoint /data/speech_data/cuijiayan/checkpoint/hifigan/g_finetune_02695000.pth.tar --vocoder_config /workspace/nartts/AdaSpeech/hifigan/config.json --restore_step 0

# From   libri + cctv + aishell
# CUDA_VISIBLE_DEVICES=0 python /workspace/nartts/AdaSpeech/train_en_cn.py -p /workspace/nartts/AdaSpeech/config/libri_cctv_aishell/preprocess.yaml -m /workspace/nartts/AdaSpeech/config/libri_cctv_aishell/model.yaml -t /workspace/nartts/AdaSpeech/config/libri_cctv_aishell/train.yaml --vocoder_checkpoint /data/speech_data/cuijiayan/checkpoint/hifigan/g_finetune_02695000.pth.tar --vocoder_config /workspace/nartts/AdaSpeech/hifigan/config.json --restore_step 0

# output mels and wavs
# CUDA_VISIBLE_DEVICES=1 python /workspace/nartts/AdaSpeech/synthesis_mel.py -p /workspace/nartts/AdaSpeech/config/en_cn_spkr/preprocess.yaml -m /workspace/nartts/AdaSpeech/config/en_cn_spkr/model.yaml -t /workspace/nartts/AdaSpeech/config/en_cn_spkr/train.yaml --vocoder_checkpoint /workspace/nartts/AdaSpeech/BigVGAN/g_05000000_large.zip --vocoder_config /workspace/nartts/AdaSpeech/BigVGAN/config_large.json --restore_step 162000