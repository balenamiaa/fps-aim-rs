use thiserror::Error;
use win_desktop_duplication::errors::DDApiError;

#[derive(Error, Debug)]
pub enum AimAssistError {
    #[error("Failed to initialize screen capturer")]
    ScreenCapturerInitError,

    #[error("Failed to capture frame")]
    ScreenCaptureError(#[from] crate::errors::ScreenCaptureError),

    #[error("Failed to run inference on screen capture")]
    InferenceError(#[from] crate::errors::InferenceError),

    #[error("Failed to move mouse")]
    MouseMovementError,

    #[error("Failed to create session")]
    SessionCreationError,
    
    #[error("Failed to set thread priority")]
    ThreadPriorityError(windows::core::Error),

    #[error("Failed to serialize/deserialize configuration")]
    ConfigurationJsonError(serde_json::Error),
}

#[derive(Error, Debug)]
pub enum ScreenCaptureError {
    #[error("Failed to initialize application's runtime for screen capturing")]
    RuntimeInitializationError,

    #[error("Failed to get GPU adapter")]
    FailedToGetAdapter,

    #[error("Failed to get display")]
    FailedToGetDisplay,

    #[error("Failed to initialize display duplication api")]
    ApiInitializationError,

    #[error("Failed to create capture region buffer")]
    CaptureRegionBufferCreationError,

    #[error("Access to frame denied")]
    DXGIError(windows::core::Error),

    #[error("Failed to create UAV for screen capture output buffer")]
    UAVCreationError,

    #[error("Failed to compile compute shader code")]
    ShaderCompilationError,

    #[error("Failed to create compute shader")]
    ShaderCreationError,

    #[error("Failed to create shader resource view for screen capture output buffer")]
    ShaderResorceCreationError(windows::core::Error),

    #[error("Failed to create output buffer.")]
    OutputBufferCreationError,

    #[error("Failed to create DXGIFactory")]
    DXGIFactoryCreationError,

    #[error("Failed to create shader input texture")]
    InputTextureCreationError,

    #[error("Failed to reacquire duplication")]
    ReacquireDuplicationError,

    #[error("Lost access to duplication output")]
    DXGIAccessLost(windows::core::Error),

    #[error("Failed to register output buffer as CUDA resource")]
    CudaRegistrationError(i32),

    #[error("Duplication output is not available")]
    DuplicationNotAvailable,
}

#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Failed to load model")]
    ModelLoadingError,

    #[error("Failed to run inference")]
    InferenceError(ort::Error),

    #[error("Failed to parse output")]
    OutputParsingError,
}
