import torch
import torch.nn as nn
from models.layers.attention import BertCrossattLayer, LXRTXLayer
from models.time2vec import Time2Vec


class MultiModalTransformer(torch.nn.Module):
    def __init__(self, args):
        super(MultiModalTransformer, self).__init__()
        self.args = args

        image_embedding_size = self.args.IMAGE_EMBEDDING_SIZES[
            self.args.image_embeddings_type
        ]
        text_embedding_size = self.args.TEXT_EMBEDDING_SIZES[
            self.args.text_embeddings_type
        ]

        if self.args.position_embeddings == "time2vec":
            self.position_embeddings = Time2Vec(args)
        elif self.args.position_embeddings == "learned":
            self.position_embeddings = nn.Parameter(
                torch.randn(
                    1,
                    self.args.window_size,
                    self.args.cross_encoder_args["embedding_size"],
                )
            )

        self.image_projection = torch.nn.Linear(
            image_embedding_size, self.args.cross_encoder_args["embedding_size"]
        )
        self.text_projection = torch.nn.Linear(
            text_embedding_size, self.args.cross_encoder_args["embedding_size"]
        )

        # this is cross_encoder
        self.layers = torch.nn.ModuleList(
            [LXRTXLayer(args) for _ in range(self.args.cross_encoder_args["n_layers"])]
        )

        self.final_transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                self.args.final_encoder_args["embedding_size"],
                self.args.final_encoder_args["n_heads"],
                activation="gelu",
                batch_first=True,
                norm_first=True,
                dropout=self.args.final_encoder_args["dropout_prob"],
                dim_feedforward=4 * self.args.final_encoder_args["embedding_size"],
            ),
            self.args.final_encoder_args["n_layers"],
        )

        self.output_classification = torch.nn.Linear(
            self.args.final_encoder_args["embedding_size"], 1
        )

    def forward(self, batch):
        extended_image_attention_mask = (1.0 - batch["image_mask"]) * -10000.0
        extended_text_attention_mask = (1.0 - batch["text_mask"]) * -10000.0
        all_lang_feats = self.text_projection(batch["text_embeddings"])
        all_visn_feats = self.image_projection(batch["image_embeddings"])

        if self.args.position_embeddings == "time2vec":
            position_embeddings = self.position_embeddings[batch["time"]]
        elif self.args.position_embeddings == "learned":
            position_embeddings = self.position_embeddings
        else:
            position_embeddings = torch.zeros_like(all_lang_feats)

        lang_feats = all_lang_feats + position_embeddings
        visn_feats = all_visn_feats + position_embeddings

        for layer_module in self.layers:
            if self.args.modality == "text":
                lang_feats = layer_module.forward_single_stream(
                    lang_feats, extended_text_attention_mask, stream="text"
                )
            elif self.args.modality == "image":
                visn_feats = layer_module.forward_single_stream(
                    visn_feats, extended_image_attention_mask, stream="image"
                )
            else:
                lang_feats, visn_feats = layer_module(
                    lang_feats,
                    extended_text_attention_mask,
                    visn_feats,
                    extended_image_attention_mask,
                )

        if self.args.modality == "image":
            final_inputs = visn_feats
            valid_mask = batch["image_mask"].reshape(
                batch["image_mask"].shape[0], -1
            ).bool()
        else:
            final_inputs = lang_feats
            valid_mask = batch["text_mask"].reshape(
                batch["text_mask"].shape[0], -1
            ).bool()

        # TransformerEncoder produces NaNs when every token is masked. Use one
        # zero-valued dummy token internally, then pool with the original mask.
        empty_rows = ~valid_mask.any(dim=1)
        safe_valid_mask = valid_mask.clone()
        safe_final_inputs = final_inputs.clone()
        if empty_rows.any():
            safe_valid_mask[empty_rows, 0] = True
            safe_final_inputs[empty_rows, 0] = 0

        final_vector = self.final_transformer(
            safe_final_inputs,
            src_key_padding_mask=~safe_valid_mask,
        )

        pool_mask = valid_mask.unsqueeze(-1).type_as(final_vector)
        final_vector = final_vector * pool_mask
        denom = pool_mask.sum(dim=1).clamp_min(1.0)
        final_vector = final_vector.sum(dim=1) / denom

        output = self.output_classification(final_vector)
        output_proba = torch.sigmoid(output)

        return {"logits": output, "probas": output_proba}


def masked_mean(sequence, mask):
    mask = mask.reshape(mask.shape[0], -1).bool()
    pool_mask = mask.unsqueeze(-1).type_as(sequence)
    sequence = sequence * pool_mask
    denom = pool_mask.sum(dim=1).clamp_min(1.0)
    return sequence.sum(dim=1) / denom


class TextImageConcatBaseline(torch.nn.Module):
    def __init__(self, args):
        super(TextImageConcatBaseline, self).__init__()
        self.args = args

        image_embedding_size = self.args.IMAGE_EMBEDDING_SIZES[
            self.args.image_embeddings_type
        ]
        text_embedding_size = self.args.TEXT_EMBEDDING_SIZES[
            self.args.text_embeddings_type
        ]
        hidden_size = self.args.final_encoder_args["embedding_size"]

        if self.args.position_embeddings == "time2vec":
            self.position_embeddings = Time2Vec(args)
        elif self.args.position_embeddings == "learned":
            self.position_embeddings = nn.Parameter(
                torch.randn(1, self.args.window_size, hidden_size)
            )

        self.image_projection = torch.nn.Linear(image_embedding_size, hidden_size)
        self.text_projection = torch.nn.Linear(text_embedding_size, hidden_size)
        self.output_classification = torch.nn.Linear(hidden_size * 2, 1)

    def forward(self, batch):
        text_feats = self.text_projection(batch["text_embeddings"])
        image_feats = self.image_projection(batch["image_embeddings"])

        if self.args.position_embeddings == "time2vec":
            position_embeddings = self.position_embeddings[batch["time"]]
        elif self.args.position_embeddings == "learned":
            position_embeddings = self.position_embeddings
        else:
            position_embeddings = torch.zeros_like(text_feats)

        text_vector = masked_mean(text_feats + position_embeddings, batch["text_mask"])
        image_vector = masked_mean(image_feats + position_embeddings, batch["image_mask"])
        final_vector = torch.cat([text_vector, image_vector], dim=-1)

        output = self.output_classification(final_vector)
        return {"logits": output, "probas": torch.sigmoid(output)}


class LSTMBaseline(torch.nn.Module):
    def __init__(self, args):
        super(LSTMBaseline, self).__init__()
        self.args = args

        image_embedding_size = self.args.IMAGE_EMBEDDING_SIZES[
            self.args.image_embeddings_type
        ]
        text_embedding_size = self.args.TEXT_EMBEDDING_SIZES[
            self.args.text_embeddings_type
        ]
        hidden_size = self.args.final_encoder_args["embedding_size"]
        lstm_args = getattr(self.args, "lstm_args", {}) or {}
        lstm_hidden_size = lstm_args.get("hidden_size", hidden_size)
        n_layers = lstm_args.get("n_layers", 1)
        dropout = lstm_args.get("dropout_prob", 0.0) if n_layers > 1 else 0.0
        bidirectional = lstm_args.get("bidirectional", True)

        self.image_projection = torch.nn.Linear(image_embedding_size, hidden_size)
        self.text_projection = torch.nn.Linear(text_embedding_size, hidden_size)
        self.lstm = torch.nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=lstm_hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        output_size = lstm_hidden_size * (2 if bidirectional else 1)
        self.output_classification = torch.nn.Linear(output_size, 1)

    def forward(self, batch):
        text_feats = self.text_projection(batch["text_embeddings"])
        image_feats = self.image_projection(batch["image_embeddings"])
        sequence = torch.cat([text_feats, image_feats], dim=-1)

        valid_mask = (batch["text_mask"] + batch["image_mask"]).reshape(
            batch["text_mask"].shape[0], -1
        ) > 0
        sequence = sequence * valid_mask.unsqueeze(-1).type_as(sequence)
        output_sequence, _ = self.lstm(sequence)
        final_vector = masked_mean(output_sequence, valid_mask.float())

        output = self.output_classification(final_vector)
        return {"logits": output, "probas": torch.sigmoid(output)}
