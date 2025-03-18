"""
CSM-1B TTS API main application.
Provides an OpenAI-compatible API for the CSM-1B text-to-speech model.
"""
import os
import time
import tempfile
import logging
from logging.handlers import RotatingFileHandler
import traceback
import asyncio
import torch
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from app.api.routes import router as api_router
# Setup logging
os.makedirs("logs", exist_ok=True)
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(log_format))
# File handler
file_handler = RotatingFileHandler(
    "logs/csm_tts_api.log", 
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setFormatter(logging.Formatter(log_format))
# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[console_handler, file_handler]
)
logger = logging.getLogger(__name__)
logger.info("Starting CSM-1B TTS API")
# Initialize FastAPI app
app = FastAPI(
    title="CSM-1B TTS API",
    description="OpenAI-compatible TTS API using the CSM-1B model from Sesame",
    version="1.0.0"
)
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Create static and other required directories
os.makedirs("static", exist_ok=True)
os.makedirs("cloned_voices", exist_ok=True)
# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")
# Include routers
app.include_router(api_router, prefix="/api/v1")
# Add OpenAI compatible route
app.include_router(api_router, prefix="/v1")
# Add voice cloning routes
from app.api.voice_cloning_routes import router as voice_cloning_router
app.include_router(voice_cloning_router, prefix="/api/v1")
app.include_router(voice_cloning_router, prefix="/v1")
# Middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to track request processing time."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.debug(f"Request to {request.url.path} processed in {process_time:.3f} seconds")
    return response
@app.on_event("startup")
async def startup_event():
    """Application startup event that loads the model and initializes voices."""
    logger.info("Starting application initialization")
    
    app.state.startup_time = time.time()
    app.state.generator = None  # Will be populated later if model loads
    app.state.logger = logger  # Make logger available to routes
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("tokenizers", exist_ok=True)
    os.makedirs("voice_memories", exist_ok=True)
    os.makedirs("voice_references", exist_ok=True)
    os.makedirs("cloned_voices", exist_ok=True)
    
    # Set tokenizer cache
    try:
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.getcwd(), "tokenizers")
        logger.info(f"Set tokenizer cache to: {os.environ['TRANSFORMERS_CACHE']}")
    except Exception as e:
        logger.error(f"Error setting tokenizer cache: {e}")
    
    # Install additional dependencies if needed
    try:
        import scipy
        import soundfile
        logger.info("Audio processing dependencies available")
    except ImportError as e:
        logger.warning(f"Audio processing dependency missing: {e}. Some audio enhancements may not work.")
        logger.warning("Consider installing: pip install scipy soundfile")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "unknown"
        logger.info(f"CUDA is available: {device_count} device(s). Using {device_name}")
        
        # Report CUDA memory
        if hasattr(torch.cuda, 'get_device_properties'):
            total_memory = torch.cuda.get_device_properties(0).total_memory
            logger.info(f"Total CUDA memory: {total_memory / (1024**3):.2f} GB")
    else:
        logger.warning("CUDA is not available. Using CPU (this will be slow)")
    
    # Determine device and device mapping
    device = "cuda" if cuda_available else "cpu"
    device_map = os.environ.get("CSM_DEVICE_MAP", None)  # Options: "auto", "balanced", "sequential"
    
    if device_map and cuda_available:
        if torch.cuda.device_count() > 1:
            logger.info(f"Using device mapping strategy: {device_map} across {torch.cuda.device_count()} GPUs")
        else:
            logger.info("Device mapping requested but only one GPU available, ignoring device_map")
            device_map = None
    else:
        device_map = None
    
    logger.info(f"Using device: {device}")
    app.state.device = device
    app.state.device_map = device_map
    
    # Check if model file exists
    model_path = os.path.join("models", "ckpt.pt")
    
    if not os.path.exists(model_path):
        # Try to download at runtime if not present
        logger.info("Model not found. Attempting to download...")
        try:
            from huggingface_hub import hf_hub_download, login
            
            # Check for token in environment
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                logger.info("Logging in to Hugging Face using provided token")
                login(token=hf_token)
                
            logger.info("Downloading CSM-1B model from Hugging Face...")
            download_start = time.time()
            model_path = hf_hub_download(
                repo_id="sesame/csm-1b", 
                filename="ckpt.pt", 
                local_dir="models"
            )
            download_time = time.time() - download_start
            logger.info(f"Model downloaded to {model_path} in {download_time:.2f} seconds")
        except Exception as e:
            error_stack = traceback.format_exc()
            logger.error(f"Error downloading model: {str(e)}\n{error_stack}")
            logger.error("Please build the image with HF_TOKEN to download the model")
            logger.error("Starting without model - API will return 503 Service Unavailable")
            return
    else:
        logger.info(f"Found existing model at {model_path}")
        logger.info(f"Model size: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")
    
    # Load the model
    try:
        logger.info("Loading CSM-1B model...")
        load_start = time.time()
        
        from app.generator import load_csm_1b
        app.state.generator = load_csm_1b(model_path, device, device_map)
        
        load_time = time.time() - load_start
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        
        # Store sample rate in app state
        app.state.sample_rate = app.state.generator.sample_rate
        logger.info(f"Model sample rate: {app.state.sample_rate} Hz")
        
        # Initialize voice enhancement system (this will create proper voice profiles)
        logger.info("Initializing voice enhancement system...")
        try:
            from app.voice_enhancement import initialize_voice_profiles, save_voice_profiles
            initialize_voice_profiles()
            logger.info("Voice profiles initialized successfully")
        except Exception as e:
            error_stack = traceback.format_exc()
            logger.error(f"Error initializing voice profiles: {str(e)}\n{error_stack}")
            logger.warning("Voice enhancement features will be limited")
        
        # Initialize voice cloning system
        try:
            logger.info("Initializing voice cloning system...")
            from app.voice_cloning import VoiceCloner, CLONED_VOICES_DIR
            
            # Update the cloned voices directory to use the persistent volume
            CLONED_VOICES_DIR = "/app/cloned_voices"
            os.makedirs(CLONED_VOICES_DIR, exist_ok=True)
            
            app.state.voice_cloner = VoiceCloner(app.state.generator, device=device)
            
            # Make sure existing voices are loaded
            app.state.voice_cloner._load_existing_voices()
            logger.info(f"Voice cloning system initialized with {len(app.state.voice_cloner.list_voices())} existing voices from {CLONED_VOICES_DIR}")
        except Exception as e:
            error_stack = traceback.format_exc()
            logger.error(f"Error initializing voice cloning: {e}\n{error_stack}")
            logger.warning("Voice cloning features will not be available")
        
        # Create prompt templates for consistent generation
        logger.info("Setting up prompt engineering templates...")
        try:
            from app.prompt_engineering import initialize_templates
            initialize_templates()
            logger.info("Prompt templates initialized")
        except Exception as e:
            error_stack = traceback.format_exc()
            logger.error(f"Error initializing prompt templates: {e}\n{error_stack}")
            logger.warning("Voice consistency features will be limited")
        
        # Generate voice reference samples (runs in background to avoid blocking startup)
        async def generate_samples_async():
            try:
                logger.info("Starting voice reference generation (background task)...")
                from app.voice_enhancement import create_voice_segments
                create_voice_segments(app.state)
                logger.info("Voice reference generation completed")
            except Exception as e:
                error_stack = traceback.format_exc()
                logger.error(f"Error in voice reference generation: {str(e)}\n{error_stack}")
        
        # Start as a background task
        asyncio.create_task(generate_samples_async())
            
        # Initialize voice cache
        app.state.voice_cache = {}
        for voice in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]:
            app.state.voice_cache[voice] = []
            
        # Log model information
        app.state.model_info = {
            "name": "CSM-1B",
            "device": device,
            "device_map": device_map,
            "sample_rate": app.state.sample_rate,
            "voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            "enhancements_enabled": True
        }
        
        # Log GPU utilization after model loading
        if cuda_available and torch.cuda.device_count() > 1 and device_map:
            logger.info("Multi-GPU setup active with the following memory usage:")
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                logger.info(f"GPU {i}: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")
        
        logger.info(f"CSM-1B TTS API is ready on {device} with sample rate {app.state.sample_rate}")
        
    except Exception as e:
        error_stack = traceback.format_exc()
        logger.error(f"Error loading model: {str(e)}\n{error_stack}")
        app.state.generator = None
        
    # Calculate total startup time
    startup_time = time.time() - app.state.startup_time
    logger.info(f"Application startup completed in {startup_time:.2f} seconds")
    
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event that cleans up resources."""
    logger.info("Application shutdown initiated")
    
    # Clean up model resources
    if hasattr(app.state, "generator") and app.state.generator is not None:
        try:
            # Clean up CUDA memory if available
            if torch.cuda.is_available():
                logger.info("Clearing CUDA cache")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logger.error(f"Error during CUDA cleanup: {e}")
    
    # Save voice profiles if they've been updated
    try:
        from app.voice_enhancement import save_voice_profiles
        logger.info("Saving voice profiles...")
        save_voice_profiles()
        logger.info("Voice profiles saved successfully")
    except Exception as e:
        logger.error(f"Error saving voice profiles: {e}")
    
    logger.info("Application shutdown complete")
# Health check endpoint
@app.get("/health", include_in_schema=False)
async def health_check():
    """Health check endpoint that returns the status of the API."""
    model_status = "healthy" if hasattr(app.state, "generator") and app.state.generator is not None else "unhealthy"
    uptime = time.time() - getattr(app.state, "startup_time", time.time())
    
    # Get enhanced voices info
    voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    try:
        from app.voice_enhancement import VOICE_PROFILES
        voices = list(VOICE_PROFILES.keys())
    except Exception:
        pass
    
    # Get cloned voices count
    cloned_voices_count = 0
    if hasattr(app.state, "voice_cloner"):
        cloned_voices_count = len(app.state.voice_cloner.list_voices())
    
    # Get CUDA memory stats if available
    cuda_stats = None
    if torch.cuda.is_available():
        cuda_stats = {
            "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "reserved_gb": torch.cuda.memory_reserved() / (1024**3)
        }
    
    return {
        "status": model_status,
        "uptime": f"{uptime:.2f} seconds",
        "device": getattr(app.state, "device", "unknown"),
        "model": "CSM-1B",
        "voices": voices,
        "cloned_voices_count": cloned_voices_count,
        "sample_rate": getattr(app.state, "sample_rate", 0),
        "enhancements": "enabled" if hasattr(app.state, "model_info") and 
                        app.state.model_info.get("enhancements_enabled", False) else "disabled",
        "cuda": cuda_stats,
        "version": "1.0.0"
    }
# Version endpoint
@app.get("/version", include_in_schema=False)
async def version():
    """Version endpoint that returns API version information."""
    return {
        "api_version": "1.0.0",
        "model_version": "CSM-1B",
        "compatible_with": "OpenAI TTS v1",
        "enhancements": "voice consistency and audio quality v1.0",
        "voice_cloning": "enabled" if hasattr(app.state, "voice_cloner") else "disabled"
    }
# Voice cloning UI endpoint
@app.get("/voice-cloning", include_in_schema=False)
async def voice_cloning_ui():
    """Voice cloning UI endpoint."""
    return FileResponse("static/voice-cloning.html")
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint that redirects to docs."""
    logger.debug("Root endpoint accessed, redirecting to docs")
    return RedirectResponse(url="/docs")
if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Development mode flag
    dev_mode = os.environ.get("DEV_MODE", "false").lower() == "true"
    
    # Log level (default to INFO, but can be overridden)
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.getLogger().setLevel(log_level)
    
    # Check for audio enhancement and voice cloning flags
    enable_enhancements = os.environ.get("ENABLE_ENHANCEMENTS", "true").lower() == "true"
    enable_voice_cloning = os.environ.get("ENABLE_VOICE_CLONING", "true").lower() == "true"
    
    if not enable_enhancements:
        logger.warning("Voice enhancements disabled by environment variable")
    if not enable_voice_cloning:
        logger.warning("Voice cloning disabled by environment variable")
    
    logger.info(f"Voice enhancements: {'enabled' if enable_enhancements else 'disabled'}")
    logger.info(f"Voice cloning: {'enabled' if enable_voice_cloning else 'disabled'}")
    logger.info(f"Log level: {log_level}")
    
    if dev_mode:
        logger.info(f"Running in development mode with auto-reload enabled on port {port}")
        uvicorn.run(
            "app.main:app", 
            host="0.0.0.0", 
            port=port, 
            reload=True, 
            log_level=log_level.lower()
        )
    else:
        logger.info(f"Running in production mode on port {port}")
        uvicorn.run(
            "app.main:app", 
            host="0.0.0.0", 
            port=port, 
            reload=False, 
            log_level=log_level.lower()
        )