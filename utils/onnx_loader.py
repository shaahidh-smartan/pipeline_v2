"""
Base ONNX Model Utility

Provides common functionality for ONNX-based models including session management,
thread safety, GPU acceleration, and inference patterns.
"""

import os
import threading
import onnxruntime


class BaseONNXModel:
    """
    Base utility class for ONNX models with common functionality.
    
    Provides shared functionality for model loading, session configuration,
    thread safety, and GPU acceleration setup. Designed to be inherited by
    specific model implementations.
    
    Features:
        - Thread-safe model loading and inference
        - Automatic GPU/CPU provider configuration
        - CUDA synchronization for consistent results
        - Configurable session options for different use cases
        - Validation and error handling
    """
    
    def __init__(self, model_file=None, session=None):
        """
        Initialize base ONNX model with thread safety and GPU support.
        
        Args:
            model_file (str, optional): Path to ONNX model file
            session (onnxruntime.InferenceSession, optional): Pre-initialized session
            
        Raises:
            AssertionError: If neither model_file nor session is provided
            AssertionError: If model_file doesn't exist
        """
        self.model_file = model_file
        self.session = session
        
        # Thread safety lock for inference calls
        self._inference_lock = threading.Lock()
        
        # Initialize session if not provided
        if self.session is None:
            self.validate_model_file()
            self.session = self.create_session()
        
        # Initialize model-specific information
        self.initialize_model_info()
    
    def validate_model_file(self):
        """
        Validate that model file exists and is accessible.
        
        Raises:
            AssertionError: If model_file is None or doesn't exist
        """
        assert self.model_file is not None, "Either model_file or session must be provided"
        assert os.path.exists(self.model_file), f"Model file not found: {self.model_file}"
    
    def create_session(self):
        """
        Create ONNX runtime session with optimized thread safety configurations.
        
        Configures session for:
        - Thread safety in multi-threaded environments
        - GPU acceleration with fallback to CPU
        - Memory management for concurrent access
        - Deterministic execution patterns
        
        Returns:
            onnxruntime.InferenceSession: Configured session ready for inference
        """
        # Configure session options for thread safety
        session_options = onnxruntime.SessionOptions()
        session_options.enable_mem_pattern = False    # Disable memory pattern for thread safety
        session_options.enable_mem_reuse = False      # Disable memory reuse to prevent race conditions
        session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL  # Force sequential execution
        
        # Configure providers for GPU acceleration with thread safety
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        provider_options = {
            'CUDAExecutionProvider': {
                'arena_extend_strategy': 'kSameAsRequested',  # Prevent memory pool conflicts
                'cudnn_conv_algo_search': 'HEURISTIC',        # Faster, more deterministic
                'do_copy_in_default_stream': '0'              # Use separate streams
            }
        }
        
        return onnxruntime.InferenceSession(
            self.model_file, 
            sess_options=session_options,
            providers=providers,
            provider_options=[provider_options.get(p, {}) for p in providers]
        )
    
    def initialize_model_info(self):
        """
        Initialize model input/output information from ONNX session.
        
        Sets up basic model information that can be used by subclasses.
        Override this method in subclasses for model-specific initialization.
        """
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
    
    def prepare(self, ctx_id=0, **kwargs):
        """
        Configure model for optimal GPU or CPU inference.
        
        Sets up execution providers and handles model-specific parameters.
        This method should be called before inference to ensure optimal performance.
        
        Args:
            ctx_id (int): Device context ID
                         >=0: Try to use GPU with given ID
                         <0:  Use CPU only
            **kwargs: Additional model-specific configuration parameters
        """
        # Configure execution providers based on device preference
        if ctx_id >= 0:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        # Update session providers
        self.session.set_providers(providers)
        
        # Handle additional parameters in subclasses
        self.handle_prepare_kwargs(**kwargs)
    
    def handle_prepare_kwargs(self, **kwargs):
        """
        Handle additional parameters for prepare method.
        
        Override this method in subclasses to handle model-specific
        configuration parameters passed to prepare().
        
        Args:
            **kwargs: Model-specific configuration parameters
        """
        pass
    
    def safe_inference(self, input_dict):
        """
        Perform thread-safe inference with CUDA synchronization.
        
        Ensures consistent results in multi-threaded environments by:
        - Using a thread lock to serialize inference calls
        - Synchronizing CUDA operations before and after inference
        - Handling torch dependency gracefully
        
        Args:
            input_dict (dict): Input dictionary mapping input names to numpy arrays
            
        Returns:
            list: Model outputs as returned by ONNX runtime
        """
        with self._inference_lock:
            # Import torch here to avoid dependency issues
            try:
                import torch
                cuda_available = torch.cuda.is_available()
            except ImportError:
                cuda_available = False
            
            # Ensure CUDA synchronization before inference
            if cuda_available:
                torch.cuda.synchronize()
            
            # Run inference
            outputs = self.session.run(self.output_names, input_dict)
            
            # Ensure CUDA synchronization after inference
            if cuda_available:
                torch.cuda.synchronize()
            
            return outputs
    
    def get_model_info(self):
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information including input/output shapes and types
        """
        inputs_info = []
        for input_info in self.session.get_inputs():
            inputs_info.append({
                'name': input_info.name,
                'shape': input_info.shape,
                'type': input_info.type
            })
        
        outputs_info = []
        for output_info in self.session.get_outputs():
            outputs_info.append({
                'name': output_info.name,
                'shape': output_info.shape,
                'type': output_info.type
            })
        
        return {
            'model_file': self.model_file,
            'inputs': inputs_info,
            'outputs': outputs_info,
            'providers': self.session.get_providers()
        }