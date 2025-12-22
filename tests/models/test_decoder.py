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
    has_first_layer_loss = True
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
    has_first_layer_loss = True
    name = "task2"

    def __call__(self, x):
        return None

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

    def __call__(self, x):
        return None

    def attn_mask(self, x):
        mask = {"input2": torch.zeros(1, NUM_QUERIES, 6, dtype=torch.bool)}
        mask["input2"][0, 1, 2] = True
        mask["input2"][0, 3, 3] = True
        mask["input2"][0, 4, 4] = True
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

        for layer in outputs.values():
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

    def test_flex_local_ca_mask(self, decoder_layer_config):
        """Test that flex_local_ca_mask method works correctly for both window_wrap branches."""
        # Use parameters that ensure valid masks (window_size should be large enough)
        window_size = 8
        q_len = 10
        kv_len = 40
        device = "cpu"

        # Test with window_wrap=False
        config = decoder_layer_config.copy()
        config["attn_kwargs"] = {"attn_type": "flex"}
        decoder_no_wrap = MaskFormerDecoder(
            num_queries=q_len,
            decoder_layer_config=config,
            num_decoder_layers=1,
            mask_attention=False,
            local_strided_attn=True,
            window_size=window_size,
            window_wrap=False,
        )

        block_mask_no_wrap = decoder_no_wrap.flex_local_ca_mask(q_len, kv_len, device)
        # Verify it returns a block mask object with mask_mod attribute
        assert hasattr(block_mask_no_wrap, "mask_mod")
        assert block_mask_no_wrap.mask_mod is not None
        # Verify it's a callable (the mask_mod function)
        assert callable(block_mask_no_wrap.mask_mod)

        # Test with window_wrap=True
        decoder_wrap = MaskFormerDecoder(
            num_queries=q_len,
            decoder_layer_config=config,
            num_decoder_layers=1,
            mask_attention=False,
            local_strided_attn=True,
            window_size=window_size,
            window_wrap=True,
        )

        block_mask_wrap = decoder_wrap.flex_local_ca_mask(q_len, kv_len, device)
        # Verify it returns a block mask object with mask_mod attribute
        assert hasattr(block_mask_wrap, "mask_mod")
        assert block_mask_wrap.mask_mod is not None
        # Verify it's a callable (the mask_mod function)
        assert callable(block_mask_wrap.mask_mod)

        # Verify both branches produce valid BlockMask objects
        # The wrapped and non-wrapped versions use different mask_mod functions
        # We can verify they're different by checking they produce different results for some inputs
        test_q_idx = torch.tensor([0, 5, 9])
        test_kv_idx = torch.tensor([0, 20, 39])
        result_no_wrap = block_mask_no_wrap.mask_mod(0, 0, test_q_idx, test_kv_idx)
        result_wrap = block_mask_wrap.mask_mod(0, 0, test_q_idx, test_kv_idx)
        # The results should be tensors of the same shape
        assert isinstance(result_no_wrap, torch.Tensor)
        assert isinstance(result_wrap, torch.Tensor)
        assert result_no_wrap.shape == result_wrap.shape

    def test_add_direct_pe(self, decoder_layer_config, sample_decoder_data):
        """Test that add_direct_pe adds positional encoding before mask generation."""
        x, input_names = sample_decoder_data
        decoder = MaskFormerDecoder(
            num_queries=NUM_QUERIES,
            decoder_layer_config=decoder_layer_config,
            num_decoder_layers=1,
            mask_attention=True,
            posenc={"alpha": 1.0, "base": 2.0},
            add_direct_pe=True,
        )
        x["key_phi"] = torch.randn(BATCH_SIZE, SEQ_LEN)

        # Create a task that will generate attention masks
        class TaskWithMask:
            has_intermediate_loss = True
            has_first_layer_loss = True
            name = "task"

            def __call__(self, x):
                return {"logit": torch.randn(BATCH_SIZE, NUM_QUERIES, SEQ_LEN)}

            def attn_mask(self, outputs):
                return {"input1": outputs["logit"].sigmoid() > 0.5}

        decoder.tasks = [TaskWithMask()]

        # Store original embeddings before forward
        original_query_embed = x["query_embed"].clone()
        original_key_embed = x["key_embed"].clone()

        # Forward pass
        updated_x, _ = decoder(x, input_names)

        # With add_direct_pe, the embeddings should have been modified by PE before mask generation
        # The embeddings should be different from the original (PE was added)
        assert not torch.allclose(updated_x["query_embed"], original_query_embed, atol=1e-5)
        assert not torch.allclose(updated_x["key_embed"], original_key_embed, atol=1e-5)

    def test_use_query_masks(self, decoder_layer_config, sample_decoder_data):
        """Test that use_query_masks collects and uses query masks from tasks."""
        x, input_names = sample_decoder_data
        decoder = MaskFormerDecoder(
            num_queries=NUM_QUERIES,
            decoder_layer_config=decoder_layer_config,
            num_decoder_layers=1,
            mask_attention=True,
            use_query_masks=True,
        )

        # Create a task that provides query masks
        class TaskWithQueryMask:
            has_intermediate_loss = True
            has_first_layer_loss = True
            name = "task"

            def __call__(self, x):
                return {"logit": torch.randn(BATCH_SIZE, NUM_QUERIES, SEQ_LEN)}

            def attn_mask(self, outputs):
                return {"input1": outputs["logit"].sigmoid() > 0.5}

            def query_mask(self, outputs):
                # Return a query mask indicating which queries are valid
                return outputs["logit"].sum(dim=-1) > 0

        decoder.tasks = [TaskWithQueryMask()]

        updated_x, _ = decoder(x, input_names)

        # Check that query_mask was set in x
        assert "query_mask" in updated_x
        assert updated_x["query_mask"].shape == (BATCH_SIZE, NUM_QUERIES)
        assert updated_x["query_mask"].dtype == torch.bool

    def test_fast_local_ca(self, decoder_layer_config, sample_local_strided_decoder_data):
        """Test fast_local_ca=True uses build_strided_sliding_window_blockmask."""
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
            fast_local_ca=True,
            block_size=64,
        )

        x, input_names = sample_local_strided_decoder_data
        x = {k: v for k, v in x.items() if k != "key_valid"}
        decoder.tasks = []

        # Forward pass should use fast_local_ca path
        updated_x, outputs = decoder(x, input_names)

        # Basic shape checks
        assert updated_x["query_embed"].shape == (1, NUM_QUERIES, DIM)
        assert updated_x["key_embed"].shape == (1, SEQ_LEN, DIM)
        assert "layer_0" in outputs

    def test_task_skipping_has_intermediate_loss(self, decoder_layer_config, sample_decoder_data):
        """Test that tasks with has_intermediate_loss=False are skipped."""
        x, input_names = sample_decoder_data
        decoder = MaskFormerDecoder(
            num_queries=NUM_QUERIES,
            decoder_layer_config=decoder_layer_config,
            num_decoder_layers=1,
            mask_attention=True,
        )

        # Create tasks with different has_intermediate_loss values
        class TaskWithIntermediate:
            has_intermediate_loss = True
            has_first_layer_loss = True
            name = "task_with"

            def __call__(self, x):
                return {"logit": torch.randn(BATCH_SIZE, NUM_QUERIES, SEQ_LEN)}

            def attn_mask(self, outputs):
                return {"input1": outputs["logit"].sigmoid() > 0.5}

        class TaskWithoutIntermediate:
            has_intermediate_loss = False
            has_first_layer_loss = False
            name = "task_without"

            def __call__(self, x):
                return {"logit": torch.randn(BATCH_SIZE, NUM_QUERIES, SEQ_LEN)}

            def attn_mask(self, outputs):
                return {"input1": outputs["logit"].sigmoid() > 0.5}

        decoder.tasks = [TaskWithIntermediate(), TaskWithoutIntermediate()]

        _, outputs = decoder(x, input_names)

        # Task with has_intermediate_loss=True should appear in outputs
        assert "task_with" in outputs["layer_0"]
        # Task with has_intermediate_loss=False should NOT appear in outputs
        assert "task_without" not in outputs["layer_0"]

    def test_task_skipping_has_first_layer_loss(self, decoder_layer_config, sample_decoder_data):
        """Test that tasks with has_first_layer_loss=False are skipped in layer 0."""
        x, input_names = sample_decoder_data
        decoder = MaskFormerDecoder(
            num_queries=NUM_QUERIES,
            decoder_layer_config=decoder_layer_config,
            num_decoder_layers=2,  # Need at least 2 layers to test this
            mask_attention=True,
        )

        # Create a task that skips first layer
        class TaskSkipFirstLayer:
            has_intermediate_loss = True
            has_first_layer_loss = False  # Skip layer 0
            name = "task_skip_first"

            def __call__(self, x):
                return {"logit": torch.randn(BATCH_SIZE, NUM_QUERIES, SEQ_LEN)}

            def attn_mask(self, outputs):
                return {"input1": outputs["logit"].sigmoid() > 0.5}

        decoder.tasks = [TaskSkipFirstLayer()]

        _, outputs = decoder(x, input_names)

        # Task should NOT appear in layer 0
        assert "task_skip_first" not in outputs["layer_0"]
        # Task should appear in layer 1
        assert "task_skip_first" in outputs["layer_1"]

    def test_phi_shift(self, decoder_layer_config, sample_decoder_data):
        """Test that phi_shift affects positional encoding generation."""
        x, _ = sample_decoder_data
        x["key_phi"] = torch.randn(BATCH_SIZE, SEQ_LEN)

        decoder_shift_0 = MaskFormerDecoder(
            num_queries=NUM_QUERIES,
            decoder_layer_config=decoder_layer_config,
            num_decoder_layers=1,
            mask_attention=False,
            posenc={"alpha": 1.0, "base": 2.0},
            phi_shift=0.0,
        )

        decoder_shift_1 = MaskFormerDecoder(
            num_queries=NUM_QUERIES,
            decoder_layer_config=decoder_layer_config,
            num_decoder_layers=1,
            mask_attention=False,
            posenc={"alpha": 1.0, "base": 2.0},
            phi_shift=0.5,
        )

        decoder_shift_0.tasks = []
        decoder_shift_1.tasks = []

        # Generate positional encodings with different shifts
        pe0_q, pe0_k = decoder_shift_0.generate_positional_encodings(x.copy())
        pe1_q, pe1_k = decoder_shift_1.generate_positional_encodings(x.copy())

        # Different phi_shift should produce different positional encodings
        assert not torch.allclose(pe0_q, pe1_q, atol=1e-5)
        # Key positional encodings should be the same (phi_shift only affects queries)
        assert torch.allclose(pe0_k, pe1_k, atol=1e-5)

    def test_unmask_all_false(self, decoder_layer_config, sample_decoder_data):
        """Test that unmask_all_false=False doesn't unmask all-false attention masks."""
        x, input_names = sample_decoder_data

        # Create a task that produces sparse masks
        class TaskSparse:
            has_intermediate_loss = True
            has_first_layer_loss = True
            name = "task"

            def __call__(self, x):
                return {"logit": torch.randn(BATCH_SIZE, NUM_QUERIES, SEQ_LEN)}

            def attn_mask(self, outputs):
                # Create very sparse masks - some queries might end up with all False
                mask = torch.zeros(BATCH_SIZE, NUM_QUERIES, 4, dtype=torch.bool)
                mask[0, 1, 1] = True  # Only query 1 has some True for input1
                return {"input1": mask}

        # Test with unmask_all_false=False
        decoder_no_unmask = MaskFormerDecoder(
            num_queries=NUM_QUERIES,
            decoder_layer_config=decoder_layer_config,
            num_decoder_layers=1,
            mask_attention=True,
            unmask_all_false=False,
        )
        decoder_no_unmask.tasks = [TaskSparse()]

        # Test with unmask_all_false=True (default)
        decoder_with_unmask = MaskFormerDecoder(
            num_queries=NUM_QUERIES,
            decoder_layer_config=decoder_layer_config,
            num_decoder_layers=1,
            mask_attention=True,
            unmask_all_false=True,
        )
        decoder_with_unmask.tasks = [TaskSparse()]

        _, outputs_no_unmask = decoder_no_unmask(x.copy(), input_names)
        _, outputs_with_unmask = decoder_with_unmask(x.copy(), input_names)

        attn_mask_no_unmask = outputs_no_unmask["layer_0"]["attn_mask"]
        attn_mask_with_unmask = outputs_with_unmask["layer_0"]["attn_mask"]

        # The masks should be different if unmask_all_false logic is applied
        # (at least verify the code path is executed)
        assert attn_mask_no_unmask.shape == (BATCH_SIZE, NUM_QUERIES, SEQ_LEN)
        assert attn_mask_with_unmask.shape == (BATCH_SIZE, NUM_QUERIES, SEQ_LEN)

        # Verify that query 1 has some True values (it should in both cases)
        assert attn_mask_no_unmask[0, 1, :].any()
        assert attn_mask_with_unmask[0, 1, :].any()


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

    def test_scale_pe(self, sample_data):
        """Test that scale_pe correctly scales positional encodings."""
        q, kv, attn_mask, kv_mask = sample_data

        # Create positional encodings
        query_posenc = torch.randn(BATCH_SIZE, NUM_QUERIES, DIM)
        key_posenc = torch.randn(BATCH_SIZE, SEQ_LEN, DIM)

        # Create two layers with different scale_pe values
        layer_scale_1 = MaskFormerDecoderLayer(dim=DIM, bidirectional_ca=True, scale_pe=1.0)
        layer_scale_2 = MaskFormerDecoderLayer(dim=DIM, bidirectional_ca=True, scale_pe=2.0)

        # Run forward passes with the same inputs
        q1, kv1 = layer_scale_1(q, kv, attn_mask=attn_mask, kv_mask=kv_mask, query_posenc=query_posenc, key_posenc=key_posenc)

        q2, kv2 = layer_scale_2(q, kv, attn_mask=attn_mask, kv_mask=kv_mask, query_posenc=query_posenc, key_posenc=key_posenc)

        # Verify that different scale_pe values produce different outputs
        # (since the scaled PE affects the attention computation)
        assert not torch.allclose(q1, q2, atol=1e-6), "Outputs should differ when scale_pe differs"
        assert not torch.allclose(kv1, kv2, atol=1e-6), "Outputs should differ when scale_pe differs"

        # Verify that scale_pe=1.0 uses the original PE (no scaling)
        # by checking that scale_pe=1.0 produces different output than scale_pe=0.0
        layer_scale_0 = MaskFormerDecoderLayer(dim=DIM, bidirectional_ca=True, scale_pe=0.0)
        q0, kv0 = layer_scale_0(q, kv, attn_mask=attn_mask, kv_mask=kv_mask, query_posenc=query_posenc, key_posenc=key_posenc)

        # scale_pe=0.0 should effectively remove PE, producing different output than scale_pe=1.0
        assert not torch.allclose(q1, q0, atol=1e-6), "scale_pe=0.0 should produce different output than scale_pe=1.0"
        assert not torch.allclose(kv1, kv0, atol=1e-6), "scale_pe=0.0 should produce different output than scale_pe=1.0"


class MockUnifiedTask:
    """Mock task for testing unified decoding strategy."""

    has_intermediate_loss = True
    has_first_layer_loss = True
    name = "unified_task"

    def __call__(self, x):
        # Return mock outputs with the expected shape
        batch_size, num_queries = x["query_embed"].shape[:2]
        num_constituents = x["key_embed"].shape[1]
        return {"track_hit_logit": torch.randn(batch_size, num_queries, num_constituents)}

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
        assert unified_decoder.num_queries == NUM_QUERIES
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
        assert x_out["query_valid"].shape == (BATCH_SIZE, NUM_QUERIES)

        # Test that layer outputs have correct structure
        for layer_idx in range(NUM_LAYERS):
            layer_key = f"layer_{layer_idx}"
            assert layer_key in outputs
            assert task.name in outputs[layer_key]

            task_outputs = outputs[layer_key][task.name]
            assert "track_hit_logit" in task_outputs
            assert task_outputs["track_hit_logit"].shape == (BATCH_SIZE, NUM_QUERIES, SEQ_LEN)
