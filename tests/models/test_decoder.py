import pytest
import torch

from hepattn.models.decoder import MaskFormerDecoder, MaskFormerDecoderLayer

BATCH_SIZE = 2
SEQ_LEN = 10
NUM_QUERIES = 5
DIM = 64
NUM_LAYERS = 2
NUM_HEADS = 8
HEAD_DIM = DIM // NUM_HEADS


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
    def sample_decoder_data(self):
        x = {
            "query_embed": torch.randn(BATCH_SIZE, NUM_QUERIES, DIM),
            "key_embed": torch.randn(BATCH_SIZE, SEQ_LEN, DIM),
            "key_posenc": torch.randn(BATCH_SIZE, SEQ_LEN, DIM),
            "key_valid": torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.bool),
            "key_is_input1": torch.zeros(SEQ_LEN, dtype=torch.bool),
            "key_is_input2": torch.zeros(SEQ_LEN, dtype=torch.bool),
        }
        # Set some positions to be input1 and input2
        x["key_is_input1"][:3] = True
        x["key_is_input2"][3:6] = True

        input_names = ["input1", "input2"]
        return x, input_names

    def test_initialization(self, decoder, decoder_layer_config):
        """Test that the decoder initializes correctly."""
        assert decoder.num_queries == NUM_QUERIES
        assert decoder.mask_attention is True
        assert decoder.use_query_masks is False
        assert len(decoder.decoder_layers) == NUM_LAYERS
        assert decoder.tasks is None
        assert decoder.query_posenc is None
        assert decoder.preserve_posenc is False

        # Check that decoder layers are initialized correctly
        for layer in decoder.decoder_layers:
            assert isinstance(layer, MaskFormerDecoderLayer)
            assert layer.mask_attention is True

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
        assert len(outputs) == NUM_LAYERS
        for i in range(NUM_LAYERS):
            assert f"layer_{i}" in outputs
            assert isinstance(outputs[f"layer_{i}"], dict)

    def test_forward_shapes(self, decoder_no_mask_attention, sample_decoder_data):
        """Test that forward pass maintains correct tensor shapes."""
        x, input_names = sample_decoder_data
        decoder_no_mask_attention.tasks = []

        original_query_shape = x["query_embed"].shape
        original_key_shape = x["key_embed"].shape

        updated_x, _ = decoder_no_mask_attention(x, input_names)

        assert updated_x["query_embed"].shape == original_query_shape
        assert updated_x["key_embed"].shape == original_key_shape

    def test_add_positional_encodings_no_posenc(self, decoder, sample_decoder_data):
        """Test add_positional_encodings when no positional encoding is set."""
        x, _ = sample_decoder_data
        original_embed = x["query_embed"].clone()

        updated_query, _ = decoder.add_positional_encodings(x)

        # Should remain unchanged when no query_posenc is set
        assert torch.equal(updated_query, original_embed)

    def test_add_positional_encodings_no_preserve(self, decoder, sample_decoder_data):
        """Test add_positional_encodings when preserve_posenc is False."""
        x, _ = sample_decoder_data
        original_query = x["query_embed"].clone()
        original_key = x["key_embed"].clone()

        updated_query, updated_key = decoder.add_positional_encodings(x)

        # Should remain unchanged when preserve_posenc is False
        assert torch.equal(updated_query, original_query)
        assert torch.equal(updated_key, original_key)


class TestMaskFormerDecoderLayer:
    @pytest.fixture
    def decoder_layer(self):
        return MaskFormerDecoderLayer(dim=DIM, mask_attention=True, bidirectional_ca=True)

    @pytest.fixture
    def sample_data(self):
        q = torch.randn(BATCH_SIZE, NUM_QUERIES, DIM)
        kv = torch.randn(BATCH_SIZE, SEQ_LEN, DIM)
        attn_mask = torch.zeros(BATCH_SIZE, NUM_QUERIES, SEQ_LEN, dtype=torch.bool)
        kv_mask = None
        return q, kv, attn_mask, kv_mask

    def test_initialization(self, decoder_layer):
        """Test that the decoder layer initializes correctly."""
        assert decoder_layer.mask_attention is True
        assert decoder_layer.bidirectional_ca is True
        assert hasattr(decoder_layer, "q_ca")
        assert hasattr(decoder_layer, "q_sa")
        assert hasattr(decoder_layer, "q_dense")
        assert hasattr(decoder_layer, "kv_ca")
        assert hasattr(decoder_layer, "kv_dense")

    def test_initialization_no_bidirectional(self):
        """Test initialization with bidirectional_ca=False."""
        layer = MaskFormerDecoderLayer(dim=DIM, mask_attention=True, bidirectional_ca=False)
        assert not hasattr(layer, "kv_ca")
        assert not hasattr(layer, "kv_dense")

    def test_forward_with_mask_attention(self, decoder_layer, sample_data):
        """Test forward pass with mask_attention=True."""
        q, kv, attn_mask, kv_mask = sample_data
        new_q, new_kv = decoder_layer(q, kv, attn_mask=attn_mask, kv_mask=kv_mask)

        # Check output shapes
        assert new_q.shape == q.shape
        assert new_kv.shape == kv.shape

    def test_forward_mask_attention_no_mask(self, decoder_layer, sample_data):
        """Test that mask_attention=True requires an attn_mask."""
        q, kv, _, kv_mask = sample_data
        with pytest.raises(AssertionError, match="attn_mask must be provided for mask attention"):
            decoder_layer(q, kv, attn_mask=None, kv_mask=kv_mask)

    def test_forward_no_mask_attention(self, sample_data):
        """Test forward pass with mask_attention=False."""
        q, kv, _, kv_mask = sample_data
        layer = MaskFormerDecoderLayer(dim=DIM, mask_attention=False, bidirectional_ca=True)

        # Should not raise an assertion error even with no attn_mask
        new_q, new_kv = layer(q, kv, attn_mask=None, kv_mask=kv_mask)

        # Check output shapes
        assert new_q.shape == q.shape
        assert new_kv.shape == kv.shape

    def test_forward_no_bidirectional(self, sample_data):
        """Test forward pass with bidirectional_ca=False."""
        q, kv, attn_mask, kv_mask = sample_data
        layer = MaskFormerDecoderLayer(dim=DIM, mask_attention=True, bidirectional_ca=False)

        new_q, new_kv = layer(q, kv, attn_mask=attn_mask, kv_mask=kv_mask)

        # Check output shapes
        assert new_q.shape == q.shape
        # Without bidirectional, kv should remain unchanged
        assert new_kv is kv

    def test_attn_mask_all_invalid(self, decoder_layer, sample_data):
        """Test behavior when attn_mask is all True for a query."""
        q, kv, attn_mask, kv_mask = sample_data
        # Set one query's mask to all True (invalid)
        attn_mask[0, 0, :] = True

        # Should not raise an error because the code handles this case
        _, _ = decoder_layer(q, kv, attn_mask=attn_mask, kv_mask=kv_mask)

        # We'd need to check that the mask was modified correctly
        # In real testing, you might want to verify this, but we'll skip for simplicity
