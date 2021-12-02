local bert_model = "bert-base-uncased";
local max_length = 128;
local bert_dim = 768;
local lstm_dim = 200; 

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
        "data_directory_path": "data/",
        "max_span_width": 20
    },
    "train_data_path": "train-task-si.labels",
    "validation_data_path": "dev-task-si.labels",
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
        }
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "sorting_keys": ["text"],
            "padding_noise": 0.0,
            "batch_size": 1
        }
    },
    "trainer": {
        "num_epochs": 150,
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