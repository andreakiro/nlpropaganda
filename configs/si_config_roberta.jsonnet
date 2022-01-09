local debug = false;

local bert_model = "roberta-base";
local max_length = 128;
local max_span_width = 10;
local bert_dim = 768;
local lstm_dim = 200;
local batch_size = 1;
local epochs = 10;

local train_data_path = if debug then "data/debug-train-si" else "data/train-si";
local validation_data_path = if debug then "data/debug-dev-si" else "data/dev-si";


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
        "max_span_width": max_span_width
    },
    "train_data_path": train_data_path,
    "validation_data_path": validation_data_path,
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
        "max_span_width": max_span_width,
    },
    "data_loader": {
        // "max_instances_in_memory": batch_size * 4,
        "batch_sampler": {
            "type": "bucket",
            "sorting_keys": ["batch_all_spans"],
            "batch_size": batch_size,
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
        },
    }
}
