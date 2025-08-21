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


class MockTask1:
    has_intermediate_loss = True
    name = "task1"

    def __call__(self, x):
        return None

    def attn_mask(self, x):
        mask = {"input1": torch.zeros(BATCH_SIZE, NUM_QUERIES, 4, dtype=torch.bool)}
        mask["input1"][0, 1, 1] = True
        mask["input1"][1, 2, 3] = True
        return mask


class MockTask2:
    has_intermediate_loss = True
    name = "task2"

    def __call__(self, x):
        return None

    def attn_mask(self, x):
        mask = {"input2": torch.zeros(BATCH_SIZE, NUM_QUERIES, 6, dtype=torch.bool)}
        mask["input2"][0, 1, 2] = True
        mask["input2"][1, 3, 3] = True
        mask["input2"][1, 4, 4] = True
        return mask


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

    def test_initialization(self, decoder, decoder_layer_config):
        """Test that the decoder initializes correctly."""
        assert decoder.num_queries == NUM_QUERIES
        assert decoder.mask_attention is True
        assert decoder.use_query_masks is False
        assert len(decoder.decoder_layers) == NUM_LAYERS
        assert decoder.tasks is None
        assert decoder.posenc is None

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
        assert len(outputs) == NUM_LAYERS
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

    def test_decoder_posenc(self, decoder_layer_config):
        dec = MaskFormerDecoder(
            num_queries=NUM_QUERIES,
            decoder_layer_config=decoder_layer_config,
            num_decoder_layers=NUM_LAYERS,
            mask_attention=True,
            posenc={"alpha": 1.0, "base": 2.0},
        )
        query_embed = torch.randn(BATCH_SIZE, NUM_QUERIES, DIM)
        key_embed = torch.randn(BATCH_SIZE, SEQ_LEN, DIM)
        x = {"key_phi": torch.randn(BATCH_SIZE, SEQ_LEN), "query_embed": query_embed.clone(), "key_embed": key_embed.clone()}
        x["query_posenc"], x["key_posenc"] = dec.generate_positional_encodings(x)
        x["query_posenc"], x["key_posenc"] = dec.add_positional_encodings(x)
        assert not torch.allclose(x["query_embed"], query_embed)
        assert not torch.allclose(x["key_embed"], key_embed)

    def test_attn_mask_construction(self, decoder, sample_decoder_data):
        """Test that attention mask is constructed correctly."""
        x, input_names = sample_decoder_data
        x["key_valid"] = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.bool)

        decoder.tasks = [MockTask1(), MockTask2()]

        _, outputs = decoder(x, input_names)

        for layer in outputs.values():
            assert "attn_mask" in layer
            attn_mask = layer["attn_mask"]
            assert attn_mask.shape == (BATCH_SIZE, NUM_QUERIES, SEQ_LEN)
            assert attn_mask.dtype == torch.bool

            # check the values
            assert attn_mask.sum() == 5
            assert attn_mask[0, 1, 1]
            assert attn_mask[1, 2, 3]
            assert attn_mask[0, 1, 6]
            assert attn_mask[1, 3, 7]
            assert attn_mask[1, 4, 8]

            # test some false entries
            assert not attn_mask[0, 0, 0]
            assert not attn_mask[0, 1, 0]
            assert not attn_mask[0, 0, 1]
            assert not attn_mask[1, 0, 1]
            assert not attn_mask[0, 1, 3]
            assert not attn_mask[1, 4, 5]


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
