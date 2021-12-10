local bert_model = "bert-base-uncased";
local max_length = 128;
local bert_dim = 768;
local lstm_dim = 200;
local batch_size = 1;

{
    "dataset_reader" : {
        "type": "si-reader",
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer_mismatched",
                "model_name": bert_model,
                "max_length": max_length
            }
        },
        "max_span_width": 20
    },
    "train_data_path": "data/train-si",
    "validation_data_path": "data/dev-si",
    "model": {
        "type": "span-identifier",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer_mismatched",
                    "model_name": bert_model,
                    "max_length": max_length
                }
            }
        },
        "context_layer": {
            "type": "lstm",
            "bidirectional": true,
            "hidden_size": lstm_dim,
            "input_size": bert_dim,
            "num_layers": 1
        },
        "feature_size": 20,
        "max_span_width": 20,
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "sorting_keys": ["batch_content"],
            "padding_noise": 0.0,
            "batch_size": batch_size,
        }
    },
    "trainer": {
        "num_epochs": 1,
        "grad_norm": 5.0,
        "patience" : 10,
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "mode": "max",
            "patience": 2
        },
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1e-3,
            "weight_decay": 0.01,
            "parameter_groups": [
                [[".*transformer.*"], {"lr": 1e-5}]
            ]
        }
    }
}