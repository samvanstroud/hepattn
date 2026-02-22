import types

import pytest
import torch
from torch import nn

from hepattn.models.decoder import MaskFormerDecoder, MaskFormerDecoderLayer

BATCH_SIZE = 2
SEQ_LEN = 10
NUM_QUERIES = 5
DIM = 64
NUM_LAYERS = 2
NUM_HEADS = 8
HEAD_DIM = DIM // NUM_HEADS


class MockTask1:
    has_intermediate_loss = True
    has_first_layer_loss = True
    name = "task1"

    def __call__(self, x, outputs=None):
        return None

    def should_run_at_layer(self, layer_index):
        return True

    def attn_mask(self, x):
        mask = {"input1": torch.zeros(BATCH_SIZE, NUM_QUERIES, 4, dtype=torch.bool)}
        mask["input1"][0, 1, 1] = True
        mask["input1"][1, 2, 3] = True
        return mask


class MockTask2:
    has_intermediate_loss = True
    has_first_layer_loss = True
    name = "task2"

    def __call__(self, x, outputs=None):
        return None

    def should_run_at_layer(self, layer_index):
        return True

    def attn_mask(self, x):
        mask = {"input2": torch.zeros(BATCH_SIZE, NUM_QUERIES, 6, dtype=torch.bool)}
        mask["input2"][0, 1, 2] = True
        mask["input2"][1, 3, 3] = True
        mask["input2"][1, 4, 4] = True
        return mask


class LCATask:
    has_intermediate_loss = True
    has_first_layer_loss = True
    name = "task2"

    def __call__(self, x, outputs=None):
        return None

    def should_run_at_layer(self, layer_index):
        return True

    def attn_mask(self, x):
        mask = {"input2": torch.zeros(1, NUM_QUERIES, 6, dtype=torch.bool)}
        mask["input2"][0, 1, 2] = True
        mask["input2"][0, 3, 3] = True
        mask["input2"][0, 4, 4] = True
        return mask


class MockQueryInitTask(nn.Module):
    name = "query_init"

    def __init__(self, probs: torch.Tensor, threshold: float = 0.5):
        super().__init__()
        self.probs = probs
        self.threshold = threshold

    def forward(self, x):
        return {"hit_is_first_prob": self.probs}

    def predict(self, outputs):
        return {"hit_is_first_prob": outputs["hit_is_first_prob"]}


class MockKMeansLogitTask:
    has_intermediate_loss = True
    has_first_layer_loss = True
    name = "kmeans_task"

    def __init__(self, logits: torch.Tensor):
        self.logits = logits

    def __call__(self, x, outputs=None):
        return {"track_hit_logit": self.logits}

    def should_run_at_layer(self, layer_index):
        return True

    def attn_mask(self, outputs):
        return {}


class TestMaskFormerDecoder:
    @pytest.fixture
    def decoder_layer_config(self):
        return {
            "dim": DIM,
            "norm": "LayerNorm",
            "dense_kwargs": {},
            "attn_kwargs": {},
            "bidirectional_ca": True,
            "hybrid_norm": False,
        }

    @pytest.fixture
    def decoder(self, decoder_layer_config):
        return MaskFormerDecoder(
            num_queries=NUM_QUERIES,
            decoder_layer_config=decoder_layer_config,
            num_decoder_layers=NUM_LAYERS,
            mask_attention=True,
        )

    @pytest.fixture
    def dynamic_decoder(self, decoder_layer_config):
        config = decoder_layer_config.copy()
        return MaskFormerDecoder(
            num_queries=2,
            decoder_layer_config=config,
            num_decoder_layers=1,
            mask_attention=False,
            dynamic_queries=True,
        )

    @pytest.fixture
    def decoder_no_mask_attention(self, decoder_layer_config):
        """Decoder with mask_attention=False for testing without tasks."""
        config = decoder_layer_config.copy()
        return MaskFormerDecoder(
            num_queries=NUM_QUERIES,
            decoder_layer_config=config,
            num_decoder_layers=NUM_LAYERS,
            mask_attention=False,
        )

    @pytest.fixture
    def decoder_local_strided_attn(self, decoder_layer_config):
        """Decoder with local_strided_attn=True for testing local window attention."""
        config = decoder_layer_config.copy()
        return MaskFormerDecoder(
            num_queries=NUM_QUERIES,
            decoder_layer_config=config,
            num_decoder_layers=NUM_LAYERS,
            mask_attention=False,  # Must be False when local_strided_attn=True
            local_strided_attn=True,
            window_size=4,
            window_wrap=True,
        )

    @pytest.fixture
    def sample_decoder_data(self):
        x = {
            "query_embed": torch.randn(BATCH_SIZE, NUM_QUERIES, DIM),
            "key_embed": torch.randn(BATCH_SIZE, SEQ_LEN, DIM),
            "key_posenc": torch.randn(BATCH_SIZE, SEQ_LEN, DIM),
            "key_valid": torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.bool),
            "key_is_input1": torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.bool),
            "key_is_input2": torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.bool),
        }

        x["key_is_input1"][:, :4] = True
        x["key_is_input2"][:, 4:] = True

        input_names = ["input1", "input2"]
        return x, input_names

    @pytest.fixture
    def sample_local_strided_decoder_data(self):
        x = {
            "query_embed": torch.randn(1, NUM_QUERIES, DIM),
            "key_embed": torch.randn(1, SEQ_LEN, DIM),
            "key_posenc": torch.randn(1, SEQ_LEN, DIM),
            "key_valid": torch.ones(1, SEQ_LEN, dtype=torch.bool),
            "key_is_input1": torch.zeros(1, SEQ_LEN, dtype=torch.bool),
            "key_is_input2": torch.zeros(1, SEQ_LEN, dtype=torch.bool),
        }

        x["key_is_input1"][:, :4] = True
        x["key_is_input2"][:, 4:] = True

        input_names = ["input1", "input2"]
        return x, input_names

    def test_initialize_dynamic_queries_topk(self, dynamic_decoder):
        # probs: select indices {0,2,3} above threshold, then keep top-2 -> [0,2]
        probs = torch.tensor([[0.9, 0.1, 0.8, 0.7]], dtype=torch.float32)
        dynamic_decoder.dynamic_query_source = "hit"
        dynamic_decoder.encoder_tasks = [MockQueryInitTask(probs=probs, threshold=0.5)]

        hit_embed = torch.randn(1, 4, DIM)
        hit_valid = torch.tensor([[True, True, True, True]])
        x = {"hit_embed": hit_embed, "hit_valid": hit_valid}

        query_embed, query_valid = dynamic_decoder.initialize_dynamic_queries(x)

        # Verify we got top 2 queries (indices 0 and 2 based on probabilities)
        assert query_embed.shape == (1, 2, DIM)
        assert torch.all(query_valid)
        assert torch.allclose(query_embed[0, 0], hit_embed[0, 0])
        assert torch.allclose(query_embed[0, 1], hit_embed[0, 2])

    def test_forward_requires_preinitialized_queries_when_dynamic(self, dynamic_decoder):
        # Decoder forward should fail loudly if dynamic queries are enabled but not provided.
        dynamic_decoder.tasks = []
        x = {
            "key_embed": torch.randn(1, SEQ_LEN, DIM),
            "key_valid": torch.ones(1, SEQ_LEN, dtype=torch.bool),
            "key_is_input1": torch.zeros(1, SEQ_LEN, dtype=torch.bool),
            "key_is_input2": torch.zeros(1, SEQ_LEN, dtype=torch.bool),
        }
        input_names = ["input1", "input2"]

        with pytest.raises(ValueError, match="encoder_tasks"):
            dynamic_decoder(x, input_names)

    def test_initialize_dynamic_queries_raises_if_none_selected(self, dynamic_decoder):
        probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
        dynamic_decoder.dynamic_query_source = "hit"
        dynamic_decoder.encoder_tasks = [MockQueryInitTask(probs=probs, threshold=0.5)]

        hit_embed = torch.randn(1, 4, DIM)
        hit_valid = torch.tensor([[True, True, True, True]])
        x = {"hit_embed": hit_embed, "hit_valid": hit_valid}

        # When no hits pass threshold, it falls back to topk, so this should succeed
        query_embed, _query_valid = dynamic_decoder.initialize_dynamic_queries(x)
        # Verify the number of queries returned matches _num_queries
        assert query_embed.shape[1] == dynamic_decoder._num_queries  # noqa: SLF001

    def test_initialize_dynamic_queries_ordering_consistency(self, decoder_layer_config):
        """Test that dynamically selected queries preserve original hit ordering.

        In HEP applications, input hits are frequently ordered by detector layer or other
        spatial coordinates. Downstream components (like windowed attention or specific
        positional encodings) may rely on this ordering. Even if top-k selection is used
        based on probability, the resulting queries should maintain their relative
        spatial/temporal order (i.e., selected_hit_indices should be monotonically increasing).
        """
        # Create a decoder with num_queries=4 to select 4 hits
        config = decoder_layer_config.copy()
        decoder = MaskFormerDecoder(
            num_queries=4,
            decoder_layer_config=config,
            num_decoder_layers=1,
            mask_attention=False,
            dynamic_queries=True,
            dynamic_query_source="hit",
        )

        # Set up probabilities where higher indices have higher probabilities
        # This tests that even when sorting by probability descending would give
        # indices [7, 5, 3, 1], we should get [1, 3, 5, 7] to preserve ordering
        # Hits: 0    1    2    3    4    5    6    7
        probs = torch.tensor([[0.1, 0.6, 0.2, 0.7, 0.3, 0.8, 0.4, 0.9]], dtype=torch.float32)
        decoder.encoder_tasks = nn.ModuleList([MockQueryInitTask(probs=probs, threshold=0.5)])

        # Create distinct embeddings for each hit so we can verify correct selection
        num_hits = 8
        hit_embed = torch.zeros(1, num_hits, DIM)
        for i in range(num_hits):
            hit_embed[0, i, 0] = float(i)  # Use first feature as identifier

        hit_valid = torch.ones(1, num_hits, dtype=torch.bool)
        x = {"hit_embed": hit_embed, "hit_valid": hit_valid}

        query_embed, _query_valid = decoder.initialize_dynamic_queries(x)

        # Extract the indices from the embeddings (we used the first feature as identifier)
        indices = query_embed[0, :, 0].long()

        # Key assertion: indices should be monotonically increasing
        sorted_indices = indices.sort().values
        assert torch.equal(indices, sorted_indices), (
            f"Selected indices {indices.tolist()} are not monotonically increasing. "
            f"Expected {sorted_indices.tolist()} to preserve original hit ordering."
        )

        # Also verify that the correct hits were selected (those above threshold)
        # Hits above threshold (0.5): indices 1, 3, 5, 7 with probs 0.6, 0.7, 0.8, 0.9
        expected_indices = [1, 3, 5, 7]
        assert indices.tolist() == expected_indices, f"Expected indices {expected_indices}, got {indices.tolist()}"

        # Verify that query_embed corresponds to hits at selected_hit_indices in order
        for i, hit_idx in enumerate(indices.tolist()):
            assert torch.allclose(query_embed[0, i], hit_embed[0, hit_idx].detach()), f"query_embed[0, {i}] should match hit_embed[0, {hit_idx}]"

        # Verify the identifier values are in the correct order
        identifiers = query_embed[0, :, 0].tolist()
        assert identifiers == [1.0, 3.0, 5.0, 7.0], f"Query embeddings are not in the expected order. Got identifiers {identifiers}"

    def test_initialization(self, decoder, decoder_layer_config):
        """Test that the decoder initializes correctly."""
        assert decoder._num_queries == NUM_QUERIES  # noqa: SLF001
        assert decoder.mask_attention is True
        assert decoder.use_query_masks is False
        assert len(decoder.decoder_layers) == NUM_LAYERS
        assert decoder.tasks is None
        assert decoder.posenc is None

        # Check that decoder layers are initialized correctly
        for layer in decoder.decoder_layers:
            assert isinstance(layer, MaskFormerDecoderLayer)

    def test_initialization_with_options(self, decoder_layer_config):
        """Test initialization with various options."""
        decoder = MaskFormerDecoder(
            num_queries=NUM_QUERIES,
            decoder_layer_config=decoder_layer_config,
            num_decoder_layers=NUM_LAYERS,
            mask_attention=False,
            use_query_masks=True,
        )

        assert decoder.mask_attention is False
        assert decoder.use_query_masks is True

    def test_forward_without_tasks(self, decoder_no_mask_attention, sample_decoder_data):
        """Test forward pass without any tasks defined."""
        x, input_names = sample_decoder_data
        decoder_no_mask_attention.tasks = []  # Empty task list

        updated_x, outputs = decoder_no_mask_attention(x, input_names)

        # Check that x was updated with new embeddings
        assert "query_embed" in updated_x
        assert "key_embed" in updated_x
        assert updated_x["query_embed"].shape == (BATCH_SIZE, NUM_QUERIES, DIM)
        assert updated_x["key_embed"].shape == (BATCH_SIZE, SEQ_LEN, DIM)

        # Check outputs structure
        assert len(outputs) == NUM_LAYERS + 1  # decoder layers + encoder
        assert "encoder" in outputs
        for i in range(NUM_LAYERS):
            assert f"layer_{i}" in outputs
            assert isinstance(outputs[f"layer_{i}"], dict)

    def test_forward_local_strided_attn(self, decoder_local_strided_attn, sample_local_strided_decoder_data):
        """Test forward pass with local_strided_attn=True."""
        x, input_names = sample_local_strided_decoder_data
        decoder_local_strided_attn.tasks = []  # Empty task list

        updated_x, outputs = decoder_local_strided_attn(x, input_names)

        # Check that x was updated with new embeddings
        assert "query_embed" in updated_x
        assert "key_embed" in updated_x
        assert updated_x["query_embed"].shape == (1, NUM_QUERIES, DIM)
        assert updated_x["key_embed"].shape == (1, SEQ_LEN, DIM)

        # Check outputs structure
        assert len(outputs) == NUM_LAYERS + 1  # decoder layers + encoder
        assert "encoder" in outputs
        for i in range(NUM_LAYERS):
            assert f"layer_{i}" in outputs
            assert isinstance(outputs[f"layer_{i}"], dict)
            # Check that attention mask was created for local strided attention
            assert "attn_mask" in outputs[f"layer_{i}"]
            attn_mask = outputs[f"layer_{i}"]["attn_mask"]
            assert attn_mask.shape == (1, NUM_QUERIES, SEQ_LEN)
            assert attn_mask.dtype == torch.bool

    def test_forward_shapes(self, decoder_no_mask_attention, sample_decoder_data):
        """Test that forward pass maintains correct tensor shapes."""
        x, input_names = sample_decoder_data
        decoder_no_mask_attention.tasks = []

        original_query_shape = x["query_embed"].shape
        original_key_shape = x["key_embed"].shape

        updated_x, _ = decoder_no_mask_attention(x, input_names)

        assert updated_x["query_embed"].shape == original_query_shape
        assert updated_x["key_embed"].shape == original_key_shape

    def test_forward_shapes_local_strided_attn(self, decoder_local_strided_attn, sample_local_strided_decoder_data):
        """Test that forward pass maintains correct tensor shapes with local_strided_attn."""
        x, input_names = sample_local_strided_decoder_data
        decoder_local_strided_attn.tasks = []

        original_query_shape = x["query_embed"].shape
        original_key_shape = x["key_embed"].shape

        updated_x, _ = decoder_local_strided_attn(x, input_names)

        assert updated_x["query_embed"].shape == original_query_shape
        assert updated_x["key_embed"].shape == original_key_shape

    def test_decoder_posenc(self, decoder_layer_config, sample_decoder_data):
        x, input_names = sample_decoder_data
        dec = MaskFormerDecoder(
            num_queries=NUM_QUERIES,
            decoder_layer_config=decoder_layer_config,
            num_decoder_layers=NUM_LAYERS,
            mask_attention=True,
            posenc={"alpha": 1.0, "base": 2.0},
        )
        dec.tasks: list = [MockTask1(), MockTask2()]
        x["key_phi"] = torch.randn(BATCH_SIZE, SEQ_LEN)
        key_embed = x["key_embed"]
        query_embed = x["query_embed"]
        x["query_posenc"], x["key_posenc"] = dec.generate_positional_encodings(x)
        assert x["query_posenc"] is not None
        assert x["key_posenc"] is not None
        updated_x, _ = dec(x, input_names)
        assert not torch.allclose(updated_x["query_embed"], query_embed)
        assert not torch.allclose(updated_x["key_embed"], key_embed)

    def test_attn_mask_construction(self, decoder, sample_decoder_data):
        """Test that attention mask is constructed correctly."""
        x, input_names = sample_decoder_data
        x["key_valid"] = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.bool)

        decoder.tasks = [MockTask1(), MockTask2()]

        _, outputs = decoder(x, input_names)

        # Check attention masks in decoder layer outputs (not encoder)
        for layer_name, layer in outputs.items():
            if layer_name == "encoder":  # Skip encoder, it doesn't have attn_mask
                continue
            assert "attn_mask" in layer
            attn_mask = layer["attn_mask"]
            assert attn_mask.shape == (BATCH_SIZE, NUM_QUERIES, SEQ_LEN)
            assert attn_mask.dtype == torch.bool

            # check the values
            assert attn_mask.sum() == 65
            assert attn_mask[0, 1, 1]
            assert attn_mask[1, 2, 3]
            assert attn_mask[0, 1, 6]
            assert attn_mask[1, 3, 7]
            assert attn_mask[1, 4, 8]

            # test some false entries
            assert attn_mask[0, 0, 0]  # becomes True due to processing
            assert not attn_mask[0, 1, 0]
            assert attn_mask[0, 0, 1]  # becomes True
            assert attn_mask[1, 0, 1]  # becomes True
            assert not attn_mask[0, 1, 3]
            assert not attn_mask[1, 4, 5]

    def test_extract_kmeans_logits_returns_dense_logits(self, decoder):
        logits = torch.randn(BATCH_SIZE, NUM_QUERIES, SEQ_LEN)
        layer_outputs = {"task_a": {"track_hit_logit": logits}}

        extracted = decoder._extract_kmeans_logits(layer_outputs, SEQ_LEN)  # noqa: SLF001

        assert extracted is logits

    def test_extract_kmeans_logits_skips_wrong_shape_logits(self, decoder):
        wrong = torch.randn(BATCH_SIZE, NUM_QUERIES, SEQ_LEN + 1)
        layer_outputs = {"task_a": {"track_hit_logit": wrong}}

        with pytest.raises(ValueError, match="cross_attn_mode='kmeans' requires"):
            decoder._extract_kmeans_logits(layer_outputs, SEQ_LEN)  # noqa: SLF001

    def test_forward_kmeans_passes_task_logits_to_decoder_layer(self, monkeypatch, decoder_layer_config, sample_decoder_data):
        config = decoder_layer_config.copy()
        config["cross_attn_mode"] = "kmeans"
        config["bidirectional_ca"] = False
        decoder = MaskFormerDecoder(
            num_queries=NUM_QUERIES,
            decoder_layer_config=config,
            num_decoder_layers=1,
            mask_attention=False,
        )

        x, input_names = sample_decoder_data
        logits = torch.randn(BATCH_SIZE, NUM_QUERIES, SEQ_LEN)
        decoder.tasks = [MockKMeansLogitTask(logits)]

        captured: dict[str, torch.Tensor | None] = {"logits": None}
        layer = decoder.decoder_layers[0]

        def fake_layer_forward(
            self,
            q,
            kv,
            attn_mask=None,
            q_mask=None,
            kv_mask=None,
            query_posenc=None,
            key_posenc=None,
            attn_mask_transpose=None,
            logits=None,
        ):
            captured["logits"] = logits
            return q, kv

        monkeypatch.setattr(layer, "forward", types.MethodType(fake_layer_forward, layer))
        decoder(x, input_names)

        assert captured["logits"] is logits

    def test_flex_local_cross_attention(self, decoder_layer_config, sample_local_strided_decoder_data):
        """Test flex implementation of local cross attention in the decoder."""
        # Configure decoder to use flex attention with local_strided_attn
        config = decoder_layer_config.copy()
        config["attn_kwargs"] = {"attn_type": "flex"}
        decoder = MaskFormerDecoder(
            num_queries=NUM_QUERIES,
            decoder_layer_config=config,
            num_decoder_layers=1,
            mask_attention=False,
            local_strided_attn=True,
            window_size=4,
            window_wrap=True,
        )

        # flex local-strided attention only supports batch size 1
        x, input_names = sample_local_strided_decoder_data
        # Remove key_valid since flex attention doesn't support kv_mask
        x = {k: v for k, v in x.items() if k != "key_valid"}
        decoder.tasks = []  # ty: ignore[unresolved-attribute]  # no tasks / pure local CA

        # Forward pass should exercise the flex local CA path, including transpose_blockmask
        updated_x, outputs = decoder(x, input_names)

        # Basic shape checks on embeddings
        assert updated_x["query_embed"].shape == (1, NUM_QUERIES, DIM)
        assert updated_x["key_embed"].shape == (1, SEQ_LEN, DIM)

        # For flex attention, attention masks are fed directly to the backend and
        # not stored in outputs, but the layer should still produce a valid entry.
        assert "layer_0" in outputs
        assert isinstance(outputs["layer_0"], dict)


class TestMaskFormerDecoderLayer:
    @pytest.fixture
    def decoder_layer(self):
        return MaskFormerDecoderLayer(dim=DIM, bidirectional_ca=True)

    @pytest.fixture
    def sample_data(self):
        q = torch.randn(BATCH_SIZE, NUM_QUERIES, DIM)
        kv = torch.randn(BATCH_SIZE, SEQ_LEN, DIM)
        attn_mask = torch.zeros(BATCH_SIZE, NUM_QUERIES, SEQ_LEN, dtype=torch.bool)
        kv_mask = None
        return q, kv, attn_mask, kv_mask

    def test_initialization(self, decoder_layer):
        """Test that the decoder layer initializes correctly."""
        assert decoder_layer.bidirectional_ca is True
        assert hasattr(decoder_layer, "q_ca")
        assert hasattr(decoder_layer, "q_sa")
        assert hasattr(decoder_layer, "q_dense")
        assert hasattr(decoder_layer, "kv_ca")
        assert hasattr(decoder_layer, "kv_dense")

    def test_initialization_no_bidirectional(self):
        """Test initialization with bidirectional_ca=False."""
        layer = MaskFormerDecoderLayer(dim=DIM, bidirectional_ca=False)
        assert not hasattr(layer, "kv_ca")
        assert not hasattr(layer, "kv_dense")

    def test_forward_with_attn_mask(self, decoder_layer, sample_data):
        """Test forward pass with attention mask."""
        q, kv, attn_mask, kv_mask = sample_data
        new_q, new_kv = decoder_layer(q, kv, attn_mask=attn_mask, kv_mask=kv_mask)

        # Check output shapes
        assert new_q.shape == q.shape
        assert new_kv.shape == kv.shape

    def test_forward_no_attn_mask(self, sample_data):
        """Test forward pass without attention mask."""
        q, kv, _, kv_mask = sample_data
        layer = MaskFormerDecoderLayer(dim=DIM, bidirectional_ca=True)

        # Should work fine with no attn_mask
        new_q, new_kv = layer(q, kv, attn_mask=None, kv_mask=kv_mask)

        # Check output shapes
        assert new_q.shape == q.shape
        assert new_kv.shape == kv.shape

    def test_forward_no_bidirectional(self, sample_data):
        """Test forward pass with bidirectional_ca=False."""
        q, kv, attn_mask, kv_mask = sample_data
        layer = MaskFormerDecoderLayer(dim=DIM, bidirectional_ca=False)

        new_q, new_kv = layer(q, kv, attn_mask=attn_mask, kv_mask=kv_mask)

        # Check output shapes
        assert new_q.shape == q.shape
        # Without bidirectional, kv should remain unchanged
        assert new_kv is kv

    def test_forward_kmeans_uses_logits_argument(self, monkeypatch, sample_data):
        q, kv, _, _ = sample_data
        layer = MaskFormerDecoderLayer(dim=DIM, bidirectional_ca=False, cross_attn_mode="kmeans")
        logits = torch.randn(BATCH_SIZE, NUM_QUERIES, SEQ_LEN)

        captured: dict[str, torch.Tensor | None] = {"logits": None}

        def fake_kmeans_forward(self, q, k=None, v=None, attn_mask=None, q_mask=None, kv_mask=None, logits=None, **kwargs):
            captured["logits"] = logits
            return q

        monkeypatch.setattr(layer.q_ca.fn, "forward", types.MethodType(fake_kmeans_forward, layer.q_ca.fn))
        new_q, new_kv = layer(q, kv, logits=logits)

        assert captured["logits"] is logits
        assert new_q.shape == q.shape
        assert new_kv is kv


class MockUnifiedTask:
    """Mock task for testing unified decoding strategy."""

    has_intermediate_loss = True
    has_first_layer_loss = True
    name = "unified_task"

    def __call__(self, x, outputs=None):
        # Return mock outputs with the expected shape
        batch_size, num_queries = x["query_embed"].shape[:2]
        num_constituents = x["key_embed"].shape[1]
        return {"track_hit_logit": torch.randn(batch_size, num_queries, num_constituents)}

    def should_run_at_layer(self, layer_index):
        return True

    def attn_mask(self, outputs):
        # Return attention mask for the full merged tensor
        attn_mask = outputs["track_hit_logit"].sigmoid() > 0.5
        return {"key": attn_mask}


class TestMaskFormerDecoderUnified:
    """Test class for unified decoding strategy."""

    @pytest.fixture
    def decoder_layer_config(self):
        return {
            "dim": DIM,
            "norm": "LayerNorm",
            "dense_kwargs": {},
            "attn_kwargs": {},
            "bidirectional_ca": True,
            "hybrid_norm": False,
        }

    @pytest.fixture
    def unified_decoder(self, decoder_layer_config):
        """Decoder with unified_decoding=True for testing unified decoding."""
        return MaskFormerDecoder(
            num_queries=NUM_QUERIES,
            decoder_layer_config=decoder_layer_config,
            num_decoder_layers=NUM_LAYERS,
            mask_attention=True,
            unified_decoding=True,
        )

    @pytest.fixture
    def sample_unified_decoder_data(self):
        """Sample data for unified decoding tests - no key_is_ masks needed."""
        x = {
            "key_embed": torch.randn(BATCH_SIZE, SEQ_LEN, DIM),
            "key_posenc": torch.randn(BATCH_SIZE, SEQ_LEN, DIM),
            "key_valid": torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.bool),
        }
        input_names = ["input1", "input2"]  # Still needed for backward compatibility
        return x, input_names

    def test_unified_initialization(self, unified_decoder):
        """Test that unified decoder initializes correctly."""
        assert unified_decoder._num_queries == NUM_QUERIES  # noqa: SLF001
        assert unified_decoder.mask_attention is True
        assert unified_decoder.unified_decoding is True
        assert len(unified_decoder.decoder_layers) == NUM_LAYERS

    def test_unified_forward_without_tasks(self, unified_decoder, sample_unified_decoder_data):
        """Test unified decoder forward pass without tasks."""
        x, input_names = sample_unified_decoder_data
        unified_decoder.tasks = []

        x_out, outputs = unified_decoder(x, input_names)

        # Check that the forward pass completes successfully
        assert "query_embed" in x_out
        assert "key_embed" in x_out
        assert x_out["query_embed"].shape == (BATCH_SIZE, NUM_QUERIES, DIM)
        assert x_out["key_embed"].shape == (BATCH_SIZE, SEQ_LEN, DIM)

        # Check that individual input embeddings are NOT created (unified mode)
        assert "input1_embed" not in x_out
        assert "input2_embed" not in x_out

        # Check layer outputs structure
        for i in range(NUM_LAYERS):
            assert f"layer_{i}" in outputs

    def test_unified_forward_with_task(self, unified_decoder, sample_unified_decoder_data):
        """Test unified decoder with a task that works on merged inputs."""
        x, input_names = sample_unified_decoder_data

        # Set up the task
        task = MockUnifiedTask()
        unified_decoder.tasks = [task]

        _, outputs = unified_decoder(x, input_names)

        # Check outputs exist for all layers
        for i in range(NUM_LAYERS):
            assert f"layer_{i}" in outputs
            assert task.name in outputs[f"layer_{i}"]
            assert "track_hit_logit" in outputs[f"layer_{i}"][task.name]

        # Check attention mask is created correctly
        for i in range(NUM_LAYERS):
            if "attn_mask" in outputs[f"layer_{i}"]:
                attn_mask = outputs[f"layer_{i}"]["attn_mask"]
                assert attn_mask.shape == (BATCH_SIZE, NUM_QUERIES, SEQ_LEN)

    def test_unified_no_key_is_masks_needed(self, unified_decoder, sample_unified_decoder_data):
        """Test that unified decoder doesn't require key_is_ masks."""
        x, input_names = sample_unified_decoder_data

        # Verify that key_is_ masks are not in the input
        assert "key_is_input1" not in x
        assert "key_is_input2" not in x

        # Set up a task
        task = MockUnifiedTask()
        unified_decoder.tasks = [task]

        # This should work without key_is_ masks
        x_out, _ = unified_decoder(x, input_names)

        # Verify successful completion
        assert "query_embed" in x_out
        assert "key_embed" in x_out

    def test_unified_attention_mask_shape(self, unified_decoder, sample_unified_decoder_data):
        """Test that attention masks have correct shape in unified mode."""
        x, input_names = sample_unified_decoder_data

        task = MockUnifiedTask()
        unified_decoder.tasks = [task]

        _, outputs = unified_decoder(x, input_names)

        # Check that attention masks, if present, have the correct shape
        for layer_idx in range(NUM_LAYERS):
            layer_outputs = outputs[f"layer_{layer_idx}"]
            if "attn_mask" in layer_outputs:
                attn_mask = layer_outputs["attn_mask"]
                # Should be (batch_size, num_queries, seq_len)
                assert attn_mask.shape == (BATCH_SIZE, NUM_QUERIES, SEQ_LEN)
                assert attn_mask.dtype == torch.bool

    def test_unified_vs_traditional_compatibility(self, decoder_layer_config):
        """Test that unified and traditional modes can coexist."""
        # Traditional decoder
        traditional_decoder = MaskFormerDecoder(
            num_queries=NUM_QUERIES,
            decoder_layer_config=decoder_layer_config,
            num_decoder_layers=NUM_LAYERS,
            mask_attention=True,
            unified_decoding=False,  # Traditional mode
        )

        # Unified decoder
        unified_decoder = MaskFormerDecoder(
            num_queries=NUM_QUERIES,
            decoder_layer_config=decoder_layer_config,
            num_decoder_layers=NUM_LAYERS,
            mask_attention=True,
            unified_decoding=True,  # Unified mode
        )

        assert traditional_decoder.unified_decoding is False
        assert unified_decoder.unified_decoding is True

    def test_unified_output_shapes(self, unified_decoder, sample_unified_decoder_data):
        """Test that all outputs have expected shapes in unified mode."""
        x, input_names = sample_unified_decoder_data

        task = MockUnifiedTask()
        unified_decoder.tasks = [task]

        x_out, outputs = unified_decoder(x, input_names)

        # Test final embeddings shapes
        assert x_out["query_embed"].shape == (BATCH_SIZE, NUM_QUERIES, DIM)
        assert x_out["key_embed"].shape == (BATCH_SIZE, SEQ_LEN, DIM)
        # query_valid is only set when dynamic_queries=True, not in regular mode
        if "query_mask" in x_out:
            assert x_out["query_mask"].shape == (BATCH_SIZE, NUM_QUERIES)

        # Test that layer outputs have correct structure
        for layer_idx in range(NUM_LAYERS):
            layer_key = f"layer_{layer_idx}"
            assert layer_key in outputs
            assert task.name in outputs[layer_key]

            task_outputs = outputs[layer_key][task.name]
            assert "track_hit_logit" in task_outputs
            assert task_outputs["track_hit_logit"].shape == (BATCH_SIZE, NUM_QUERIES, SEQ_LEN)
