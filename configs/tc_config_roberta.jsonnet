local bert_model = "bert-base-uncased";
local si_model = "models/models_si/si_roberta/model.tar.gz";

local epochs = 1;
local max_span_width = 10;

local max_length = 128;
local bert_dim = 768;
local lstm_dim = 200;
local batch_size = 1;

{
    "dataset_reader" : {
        "type": "tc-reader",
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer_mismatched",
                "model_name": bert_model,
                "max_length": max_length
            }
        },
        "max_span_width": max_span_width,
        "si_model": {
            "type": "from_archive",
            "archive_file": si_model
        },
    },
    "train_data_path": "data/train-tc",
    "validation_data_path": "data/dev-tc",
    "model": {
        "type": "technique-classifier",
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
        "max_span_width": max_span_width,
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "sorting_keys": ["content"],
            "padding_noise": 0.0,
            "batch_size": batch_size
        }
    },
    "trainer": {
        "num_epochs": epochs,
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