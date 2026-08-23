import torch
import torch.nn as nn
import copy
import json
import math
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import nomenclature

device = "cuda" if torch.cuda.is_available() else "cpu"


class TextOnlyBERT(nn.Module):
    """Encode each post with a configurable BERT and reuse the temporal classifier."""

    def __init__(self, trans_args):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "The text-only BERT encoder requires the transformers package."
            ) from exc

        model_name = trans_args.bert_model_name
        self.bert_model_name = model_name
        print(f"Loading text encoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_text_encoder = AutoModel.from_pretrained(model_name)
        self.max_length = trans_args.bert_max_length
        self.post_batch_size = trans_args.bert_post_batch_size
        self.finetune_bert = trans_args.bert_finetune
        if not self.finetune_bert:
            self.bert_text_encoder.requires_grad_(False)
            self.bert_text_encoder.eval()

        hidden_size = self.bert_text_encoder.config.hidden_size
        expected_size = trans_args.TEXT_EMBEDDING_SIZES[
            trans_args.text_embeddings_type
        ]
        if hidden_size != expected_size:
            raise ValueError(
                f"BERT hidden size {hidden_size} does not match configured "
                f"text embedding size {expected_size}."
            )

        trans_model = nomenclature.MODELS[trans_args.model]
        self.multi_modal_transformer = trans_model(trans_args)

    def train(self, mode=True):
        super().train(mode)
        if not self.finetune_bert:
            self.bert_text_encoder.eval()
        return self

    @staticmethod
    def _normalize_text(value):
        if value is None:
            return ""
        if isinstance(value, str):
            return "" if value == "<PAD>" else value
        if isinstance(value, float) and math.isnan(value):
            return ""
        if isinstance(value, (list, tuple)):
            return " ".join(TextOnlyBERT._normalize_text(item) for item in value)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def forward(self, batch):
        batch_size = len(batch["texts"])
        window_size = len(batch["texts"][0])
        flat_texts = [
            self._normalize_text(text)
            for timeline in batch["texts"]
            for text in timeline
        ]
        feature_chunks = []
        for start in range(0, len(flat_texts), self.post_batch_size):
            text_chunk = flat_texts[start : start + self.post_batch_size]
            tokenized = self.tokenizer(
                text_chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokenized = {
                key: value.to(device) for key, value in tokenized.items()
            }
            if self.finetune_bert:
                bert_output = self.bert_text_encoder(**tokenized)
            else:
                with torch.no_grad():
                    bert_output = self.bert_text_encoder(**tokenized)
            feature_chunks.append(bert_output.last_hidden_state[:, 0, :])

        text_features = torch.cat(feature_chunks, dim=0)
        text_features = text_features.reshape(batch_size, window_size, -1)

        image_size = self.multi_modal_transformer.args.IMAGE_EMBEDDING_SIZES[
            self.multi_modal_transformer.args.image_embeddings_type
        ]
        new_batch = {
            "image_mask": torch.zeros_like(batch["image_mask"]).to(device),
            "text_mask": batch["text_mask"].to(device),
            "text_embeddings": text_features,
            "image_embeddings": torch.zeros(
                batch_size,
                window_size,
                image_size,
                device=device,
                dtype=text_features.dtype,
            ),
            "time": batch["time"].to(device),
        }
        return self.multi_modal_transformer(new_batch)


class BertImageCLIP(nn.Module):
    """Use BERT text features and CLIP image features with the existing fusion model."""

    def __init__(self, cfg, clip_model, trans_args):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "The BERT+CLIP multimodal encoder requires the transformers package."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(trans_args.bert_model_name)
        self.bert_text_encoder = AutoModel.from_pretrained(trans_args.bert_model_name)
        self.max_length = trans_args.bert_max_length
        self.post_batch_size = trans_args.bert_post_batch_size
        self.finetune_bert = trans_args.bert_finetune
        if not self.finetune_bert:
            self.bert_text_encoder.requires_grad_(False)
            self.bert_text_encoder.eval()

        hidden_size = self.bert_text_encoder.config.hidden_size
        expected_size = trans_args.TEXT_EMBEDDING_SIZES[
            trans_args.text_embeddings_type
        ]
        if hidden_size != expected_size:
            raise ValueError(
                f"BERT hidden size {hidden_size} does not match configured "
                f"text embedding size {expected_size}."
            )

        trans_model = nomenclature.MODELS[trans_args.model]
        self.prompt_learner = MultiModalPromptLearner(cfg, clip_model)
        self.image_encoder = clip_model.visual
        self.multi_modal_transformer = trans_model(trans_args)
        self.dtype = clip_model.dtype

    def train(self, mode=True):
        super().train(mode)
        if not self.finetune_bert:
            self.bert_text_encoder.eval()
        return self

    def _encode_texts(self, texts):
        batch_size = len(texts)
        window_size = len(texts[0])
        flat_texts = [
            TextOnlyBERT._normalize_text(text)
            for timeline in texts
            for text in timeline
        ]

        feature_chunks = []
        for start in range(0, len(flat_texts), self.post_batch_size):
            text_chunk = flat_texts[start : start + self.post_batch_size]
            tokenized = self.tokenizer(
                text_chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokenized = {
                key: value.to(device) for key, value in tokenized.items()
            }
            if self.finetune_bert:
                bert_output = self.bert_text_encoder(**tokenized)
            else:
                with torch.no_grad():
                    bert_output = self.bert_text_encoder(**tokenized)
            feature_chunks.append(bert_output.last_hidden_state[:, 0, :])

        text_features = torch.cat(feature_chunks, dim=0)
        return text_features.reshape(batch_size, window_size, -1)

    def _encode_images(self, texts, images):
        image_features_list = []
        for text, image in zip(texts, images):
            (
                _prompts,
                _tokenized_prompt,
                shared_ctx,
                _deep_compound_prompts_text,
                deep_compound_prompts_vision,
            ) = self.prompt_learner(text)
            image = torch.stack(image).to(device)
            image_features = self.image_encoder(
                image.type(self.dtype),
                shared_ctx,
                deep_compound_prompts_vision,
            )
            image_features_list.append(image_features)
        return torch.stack(image_features_list)

    def forward(self, batch):
        text_features = self._encode_texts(batch["texts"])
        image_features = self._encode_images(batch["texts"], batch["images"])
        new_batch = {
            "image_mask": batch["image_mask"].to(device),
            "text_mask": batch["text_mask"].to(device),
            "text_embeddings": text_features.to(device),
            "image_embeddings": image_features.to(device),
            "time": batch["time"].to(device),
        }
        return self.multi_modal_transformer(new_batch)


class TextEncoder(nn.Module):
    """
    A text encoder that processes input prompts using a transformer model from CLIP.

    Attributes:
        transformer (nn.Module): The transformer model from CLIP.
        positional_embedding (torch.Tensor): Positional embeddings for text inputs.
        ln_final (nn.LayerNorm): Final normalization layer.
        text_projection (torch.Tensor): Projection matrix for text features.
        dtype (torch.dtype): Data type for computations.
    """

    def __init__(self, clip_model):
        """
        Initializes the TextEncoder with components from the provided CLIP model.

        Args:
            clip_model (nn.Module): Pretrained CLIP model.
        """
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        """
        Forward pass of the text encoder.

        Args:
            prompts (torch.Tensor): Embedded text prompts.
            tokenized_prompts (torch.Tensor): Tokenized prompts.
            compound_prompts_deeper_text (torch.Tensor): Deeper-level prompt embeddings.

        Returns:
            torch.Tensor: Encoded text features.
        """
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        # Pass as a list, as nn.Sequential cannot process multiple arguments
        combined = [
            x,
            compound_prompts_deeper_text,
            0,  # Depth counter for prompt propagation
        ]
        outputs = self.transformer(combined)
        x = outputs[0]  # Extract the processed tensor

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )
        return x


class MultiModalPromptLearner(nn.Module):
    """
    A module for learning multi-modal prompts in MaPLe.

    Attributes:
        compound_prompts_depth (int): Depth of compound prompts.
        proj (nn.Linear): Linear layer to project prompt embeddings.
        ctx (nn.Parameter): Learnable context vectors.
        compound_prompts_text (nn.ParameterList): List of deeper-level prompt embeddings.
        compound_prompt_projections (nn.ModuleList): List of projection layers for deep prompts.
        n_cls (int): Number of classes.
        n_ctx (int): Number of context tokens.
        prompt_prefix (str): Prefix for text prompts.
        clip_model (nn.Module): CLIP model for embedding text.
        dtype (torch.dtype): Data type for computations.
    """

    def __init__(self, cfg, clip_model):
        """
        Initializes the multi-modal prompt learner.

        Args:
            cfg (dict): Configuration dictionary containing training parameters.
            clip_model (nn.Module): Pretrained CLIP model.
        """
        super().__init__()
        n_ctx = cfg.TRAINER.MAPLE.N_CTX
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224

        assert (
            cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1
        ), "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH

        assert (
            cfg_imsize == clip_imsize
        ), f"cfg_imsize ({cfg_imsize}) must match clip_imsize ({clip_imsize})"

        if ctx_init and n_ctx <= 4:
            # Use predefined words for context initialization
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # Random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")

        self.proj = nn.Linear(ctx_dim, 768)
        self.ctx = nn.Parameter(ctx_vectors)

        # Initialize deep prompt parameters
        self.compound_prompts_text = nn.ParameterList(
            [
                nn.Parameter(torch.empty(n_ctx, 512))
                for _ in range(self.compound_prompts_depth - 1)
            ]
        )
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)

        # Projection layers for deep prompts
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(
            single_layer, self.compound_prompts_depth - 1
        )

        self.n_cls = 1
        self.n_ctx = n_ctx
        self.prompt_prefix = prompt_prefix
        self.clip_model = clip_model
        self.dtype = dtype

    def construct_prompts(self, clip_model, ctx, prompt_prefix, text, dtype, n_ctx):
        """
        Constructs prompts by embedding textual inputs.

        Args:
            clip_model (nn.Module): The CLIP model for embedding.
            ctx (torch.Tensor): Context embeddings.
            prompt_prefix (str): Prefix text for prompts.
            text (list of str): Input text samples.
            dtype (torch.dtype): Data type for embeddings.
            n_ctx (int): Number of context tokens.

        Returns:
            tuple: (Prompt embeddings, Tokenized prompts)
        """
        ctx = ctx.to(device)

        with torch.no_grad():
            prompts = [f"{prompt_prefix} {item}" for item in text]
            tokenized_prompts = clip.tokenize(prompts, truncate=True).to(device)
            embeddings = (
                clip_model.token_embedding(tokenized_prompts).type(dtype).to(device)
            )

            prefix = embeddings[:, :1, :]  # SOS token
            suffix = embeddings[:, 1 + n_ctx :, :]  # CLS, EOS tokens

            prompt_list = []
            tokenized_prompt_list = []

            for i in range(len(text)):
                prompt_embedding = torch.cat(
                    (prefix[i : i + 1], ctx, suffix[i : i + 1]), dim=1
                )
                prompt_list.append(prompt_embedding)
                tokenized_prompt_list.append(tokenized_prompts[i : i + 1])

        return torch.cat(prompt_list, dim=0), torch.cat(tokenized_prompt_list, dim=0)

    def forward(self, text):
        """
        Forward pass for the prompt learner.

        Args:
            text (list of str): Input text samples.

        Returns:
            tuple: Processed text prompts and visual prompts.
        """
        clip_model = self.clip_model
        ctx = (
            self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            if self.ctx.dim() == 2
            else self.ctx
        )
        prompts, tokenized_prompt = self.construct_prompts(
            clip_model, ctx, self.prompt_prefix, text, self.dtype, self.n_ctx
        )

        visual_deep_prompts = [
            layer(p)
            for layer, p in zip(
                self.compound_prompt_projections, self.compound_prompts_text
            )
        ]

        return (
            prompts,
            tokenized_prompt,
            self.proj(self.ctx),
            self.compound_prompts_text,
            visual_deep_prompts,
        )


class CustomCLIP(nn.Module):
    """
    A custom CLIP model with multi-modal capabilities.

    Attributes:
        prompt_learner (MultiModalPromptLearner): Learns multi-modal prompts.
        multi_modal_transformer (nn.Module): Multi-modal transformer for fusion.
        image_encoder (nn.Module): Image encoder from CLIP.
        text_encoder (TextEncoder): Custom text encoder.
        dtype (torch.dtype): Data type for computations.
    """

    def __init__(self, cfg, clip_model, trans_args):
        """
        Initializes the CustomCLIP model.

        Args:
            cfg (dict): Configuration dictionary.
            clip_model (nn.Module): Pretrained CLIP model.
            trans_args (argparse.Namespace): Transformer arguments.
        """
        super().__init__()
        trans_model = nomenclature.MODELS[trans_args.model]
        self.prompt_learner = MultiModalPromptLearner(cfg, clip_model)
        self.multi_modal_transformer = trans_model(trans_args)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype

    def forward(self, batch):
        """Processes batch through the model."""
        texts = batch["texts"]
        images = batch["images"]

        # Handle multiple texts and images for a single entry
        text_features_list = []
        image_features_list = []

        # Iterate over texts and corresponding images
        for text, image in zip(texts, images):
            prompts, tokenized_prompt, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner(text)
            text_features = self.text_encoder(prompts, tokenized_prompt, deep_compound_prompts_text)

            # Convert the list to a tensor
            image = torch.stack(image).to(device)

            image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)
            
            image_features_list.append(image_features)
            text_features_list.append(text_features)

        # Aggregate features
        image_features = torch.stack(image_features_list)
        text_features = torch.stack(text_features_list)

        new_batch = {
            "image_mask" : batch["image_mask"].to(device),
            "text_mask" : batch["text_mask"].to(device),
            "text_embeddings" : text_features.to(device),
            "image_embeddings" : image_features.to(device),
            "time" : batch["time"].to(device),
        }
        output = self.multi_modal_transformer(new_batch)
        return output


def _get_clones(module, N):
    """Creates N copies of a module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
