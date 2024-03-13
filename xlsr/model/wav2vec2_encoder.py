from transformers import Wav2Vec2Config, Wav2Vec2Model
import torch
import os
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
model_path = os.path.join(parent_directory, 'xlsr_53_spkr.pt')

def load_model(ckpt_path="/data/speech_data/cuijiayan/tools/xlsr/xlsr_53_spkr.pt"):
    config, _ = Wav2Vec2Config.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53",
        cache_dir=None,
        return_unused_kwargs=True,
        force_download=False,
        resume_download=False,
        proxies=None,
        local_files_only=False,
        use_auth_token=None,
        revision=None,
        subfolder='',
        _from_auto=False,
        _from_pipeline=None,
    )

    model = Wav2Vec2Model(config)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)

    for param in model.parameters():
        param.requires_grad = False
        param.grad = None
    
    model.eval()
    return model


def extract_xlsr_spkr(model, input_values):
    '''
    input_values: [B, T]
    '''
    model.eval()
    
    all_hidden_states = ()
    with torch.no_grad():
        extract_features = model.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        hidden_states, extract_features = model.feature_projection(extract_features)
        hidden_states = model._mask_hidden_states(
            hidden_states, mask_time_indices=None, attention_mask=None
        )

        position_embeddings = model.encoder.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = model.encoder.dropout(hidden_states)

        for layer in model.encoder.layers[:2]:
            all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer(hidden_states,
                                attention_mask=None, output_attentions=False)
            hidden_states = layer_outputs[0]
    
    return all_hidden_states[1]
