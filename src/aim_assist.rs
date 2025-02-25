use std::{
    fmt::{self, Display, Formatter, Write},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc, OnceLock,
    },
    thread,
    time::Instant,
};

use log::warn;
use ort::{CPUExecutionProvider, CUDAExecutionProvider, Session, TensorRTExecutionProvider};
use serde::{Deserialize, Serialize};

use crate::interception::{self, is_button_held_mouse4, is_button_held_mouse5, InterceptionContext};
use crate::screen_capture::{ScreenCaptureOutput, ScreenCapturer};
use crate::{errors::AimAssistError, interception::VirtualKey};
use crate::{
    inference::{DetectionResult, InferenceEngine},
    interception::{INTERCEPTION_KEY_DOWN, INTERCEPTION_KEY_UP, INTERCEPTION_SCANCODE_SPACE},
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AimAssistAccelerator {
    CUDA,
    TensorRT,
    None,
}

#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum AimAssistTarget {
    Head,
    Torso,
}

impl Display for AimAssistTarget {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            AimAssistTarget::Head => write!(f, "Head"),
            AimAssistTarget::Torso => write!(f, "Torso"),
        }
    }
}

pub trait ActivationCondition {
    fn should_aim_assist(&self) -> bool;
    fn update(&mut self);
}

pub trait SmoothingFunction {
    fn calculate(&self, distance: f64, dt: f64) -> f64;
    fn update(&mut self);
}

pub trait TargetPredictor {
    fn predict(&mut self, target_x: f32, target_y: f32, dt: f64) -> (f32, f32);
    fn reset(&mut self);
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ToggleActivationCondition {
    toggle_mouse_button: u16,
    #[serde(skip)]
    was_previously_down: bool,
    is_toggled: bool,
}

impl ToggleActivationCondition {
    pub fn new(toggle_mouse_button: u16) -> Self {
        Self {
            toggle_mouse_button,
            was_previously_down: false,
            is_toggled: false,
        }
    }
}

impl ActivationCondition for ToggleActivationCondition {
    fn should_aim_assist(&self) -> bool {
        self.is_toggled
    }

    fn update(&mut self) {
        let is_down = match self.toggle_mouse_button {
            0x05 => is_button_held_mouse4(),
            0x06 => is_button_held_mouse5(),
            _ => unreachable!(),
        };

        if is_down && !self.was_previously_down {
            self.is_toggled = !self.is_toggled;
        }

        self.was_previously_down = is_down;
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NullSmoothing {}

impl SmoothingFunction for NullSmoothing {
    fn calculate(&self, _distance: f64, _dt: f64) -> f64 {
        1.0
    }

    fn update(&mut self) {}
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearInterpolationSmoothing {
    pub distances: Vec<f64>,
    pub multipliers: Vec<f64>,
}

impl LinearInterpolationSmoothing {
    fn new(distances: Vec<f64>, multipliers: Vec<f64>) -> Self {
        assert!(distances.len() == multipliers.len() && distances.len() >= 2, "distances and multipliers must have the same length and be at least of length 2");

        Self { distances, multipliers }
    }

    fn lerpf(x: f64, x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
        y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    }
}

impl SmoothingFunction for LinearInterpolationSmoothing {
    fn calculate(&self, distance: f64, _dt: f64) -> f64 {
        if self.distances.is_empty() {
            return 1.0;
        }

        if distance <= self.distances[0] {
            return self.multipliers[0];
        } else if distance >= self.distances[self.distances.len() - 1] {
            return self.multipliers[self.multipliers.len() - 1];
        }

        let i = self.distances.iter().position(|&d| d > distance).unwrap();

        Self::lerpf(distance, self.distances[i - 1], self.distances[i], self.multipliers[i - 1], self.multipliers[i])
    }

    fn update(&mut self) {}
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveSmoothingModifier<T: SmoothingFunction + Clone> {
    pub base_smoothing: T,
    pub progress_factor: f64,
    #[serde(skip)]
    time_mouse1_down: Option<Instant>,
}

impl<T: SmoothingFunction + Clone> ProgressiveSmoothingModifier<T> {
    pub fn new(base_smoothing: T, progress_factor: f64) -> Self {
        Self {
            base_smoothing,
            progress_factor,
            time_mouse1_down: None,
        }
    }
}

impl<T: SmoothingFunction + Clone> SmoothingFunction for ProgressiveSmoothingModifier<T> {
    fn calculate(&self, distance: f64, dt: f64) -> f64 {
        let dt_mouse1_down = self.time_mouse1_down.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);
        self.base_smoothing.calculate(distance, dt) / (1.0 + self.progress_factor * dt_mouse1_down)
    }

    fn update(&mut self) {
        self.base_smoothing.update();
        let is_down = interception::is_vk_down(0x1);

        if is_down && !self.time_mouse1_down.is_some() {
            self.time_mouse1_down = Some(Instant::now());
        } else if !is_down && self.time_mouse1_down.is_some() {
            self.time_mouse1_down = None;
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LinearPredictor {
    #[serde(skip)]
    previous_x: Option<f32>,
    #[serde(skip)]
    previous_y: Option<f32>,
    #[serde(skip)]
    velocity: Option<f64>,
}

impl LinearPredictor {
    pub fn new() -> Self {
        Self { previous_x: None, previous_y: None, velocity: None }
    }
}

impl TargetPredictor for LinearPredictor {
    fn predict(&mut self, target_x: f32, target_y: f32, dt: f64) -> (f32, f32) {
        match (self.previous_x, self.previous_y) {
            (Some(prev_x), Some(prev_y)) => {
                let delta_x = target_x - prev_x;
                let delta_y = target_y - prev_y;
                self.previous_x = Some(target_x);
                self.previous_y = Some(target_y);

                let velocity_x = delta_x as f64 / dt;
                let velocity_y = delta_y as f64 / dt;
                let acceleration_x = self.velocity.map(|v| (velocity_x - v) / dt).unwrap_or(0.0);
                let acceleration_y = self.velocity.map(|v| (velocity_y - v) / dt).unwrap_or(0.0);
                let target_x = target_x + (velocity_x * dt + 0.5 * acceleration_x * dt.powi(2)) as f32;
                let target_y = target_y + (velocity_y * dt + 0.5 * acceleration_y * dt.powi(2)) as f32;
                (target_x, target_y)
            }
            _ => {
                self.previous_x = Some(target_x);
                self.previous_y = Some(target_y);
                (target_x, target_y)
            }
        }
    }

    fn reset(&mut self) {
        self.previous_x = None;
        self.previous_y = None;
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NullPredictor {}

impl TargetPredictor for NullPredictor {
    fn predict(&mut self, target_x: f32, target_y: f32, _dt: f64) -> (f32, f32) {
        (target_x, target_y)
    }

    fn reset(&mut self) {}
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SmoothingFunctions {
    Null(NullSmoothing),
    LinearInterpolation(LinearInterpolationSmoothing),
    ProgressiveLinearInterpolation(ProgressiveSmoothingModifier<LinearInterpolationSmoothing>),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationConditions {
    Toggle(ToggleActivationCondition),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TargetPredictors {
    Linear(LinearPredictor),
    Null(NullPredictor),
}

impl SmoothingFunction for SmoothingFunctions {
    fn calculate(&self, distance: f64, dt: f64) -> f64 {
        match self {
            SmoothingFunctions::Null(smoothing) => smoothing.calculate(distance, dt),
            SmoothingFunctions::LinearInterpolation(smoothing) => smoothing.calculate(distance, dt),
            SmoothingFunctions::ProgressiveLinearInterpolation(smoothing) => smoothing.calculate(distance, dt),
        }
    }

    fn update(&mut self) {
        match self {
            SmoothingFunctions::Null(smoothing) => smoothing.update(),
            SmoothingFunctions::LinearInterpolation(smoothing) => smoothing.update(),
            SmoothingFunctions::ProgressiveLinearInterpolation(smoothing) => smoothing.update(),
        }
    }
}

impl ActivationCondition for ActivationConditions {
    fn should_aim_assist(&self) -> bool {
        match self {
            ActivationConditions::Toggle(condition) => condition.should_aim_assist(),
        }
    }

    fn update(&mut self) {
        match self {
            ActivationConditions::Toggle(condition) => condition.update(),
        }
    }
}

impl TargetPredictor for TargetPredictors {
    fn predict(&mut self, target_x: f32, target_y: f32, dt: f64) -> (f32, f32) {
        match self {
            TargetPredictors::Linear(predictor) => predictor.predict(target_x, target_y, dt),
            TargetPredictors::Null(predictor) => predictor.predict(target_x, target_y, dt),
        }
    }

    fn reset(&mut self) {
        match self {
            TargetPredictors::Linear(predictor) => predictor.reset(),
            TargetPredictors::Null(predictor) => predictor.reset(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AimAssistConfiguration {
    pub accelerator: AimAssistAccelerator,
    pub target_predictor: TargetPredictors,
    pub smoothing_function: SmoothingFunctions,
    pub activation_condition: ActivationConditions,
    pub target: AimAssistTarget,
    pub confidence_threshold: f32,
    pub capture_width: u32,
    pub capture_height: u32,
    pub sensitivity: (f32, f32),
    pub dpi: f32,
    pub bhop_enabled: bool,
    pub bhop_held_vk: VirtualKey,
    pub bhop_frequency_micros: u64,
    pub mouse_move_frequency_micros: u64,
}

impl AimAssistConfiguration {
    const FILE_NAME: &'static str = "config.json";

    pub fn cs2_configuration() -> Result<Self, AimAssistError> {
        let file = std::fs::File::open(Self::FILE_NAME);
        if let Ok(file) = file {
            let reader = std::io::BufReader::new(file);
            let config: AimAssistConfiguration = serde_json::from_reader(reader).map_err(|e| AimAssistError::ConfigurationJsonError(e))?;
            return Ok(config);
        } else {
            let config = Self {
                accelerator: AimAssistAccelerator::TensorRT,
                target_predictor: TargetPredictors::Linear(LinearPredictor::new()),
                smoothing_function: SmoothingFunctions::ProgressiveLinearInterpolation(ProgressiveSmoothingModifier::new(LinearInterpolationSmoothing::new(vec![0.0, 0.2, 0.8, 1.0, 1.6], vec![0.0, 0.01, 0.002, 0.001, 0.0]), 5.0)),
                activation_condition: ActivationConditions::Toggle(ToggleActivationCondition::new(VirtualKey::X1Button as u16)),
                target: AimAssistTarget::Head,
                confidence_threshold: 0.6,
                capture_width: 320,
                capture_height: 320,
                sensitivity: (1.0, 1.0),
                dpi: 1280.0,
                bhop_enabled: true,
                bhop_held_vk: VirtualKey::X2Button,
                bhop_frequency_micros: 1500,
                mouse_move_frequency_micros: 1000,
            };

            let file = std::fs::File::create(Self::FILE_NAME).unwrap();
            serde_json::to_writer_pretty(file, &config).map_err(|e| AimAssistError::ConfigurationJsonError(e))?;

            Ok(config)
        }
    }

    pub fn save(&self) -> Result<(), AimAssistError> {
        let file = std::fs::File::create(Self::FILE_NAME).unwrap();
        serde_json::to_writer_pretty(file, self).map_err(|e| AimAssistError::ConfigurationJsonError(e))?;
        Ok(())
    }

    pub fn load(&mut self) -> Result<(), AimAssistError> {
        let file = std::fs::File::open(Self::FILE_NAME).unwrap();
        let reader = std::io::BufReader::new(file);
        *self = serde_json::from_reader(reader).map_err(|e| AimAssistError::ConfigurationJsonError(e))?;
        Ok(())
    }

    pub fn x_correction_multiplier(&self) -> f32 {
        22.0 * (640.0 / self.dpi) / self.sensitivity.0
    }

    pub fn y_correction_multiplier(&self) -> f32 {
        22.0 * (640.0 / self.dpi) / self.sensitivity.1
    }
}

struct InferenceOutput {
    detections: Vec<DetectionResult>,
    inference_time: f64,
    screencap_time: f64,
    frame_start_time: Instant,
}

pub struct AimAssistModule {
    config: &'static mut AimAssistConfiguration,
    running: Arc<AtomicBool>,
    _inference_thread: thread::JoinHandle<Result<(), AimAssistError>>,
    interception_ctx: InterceptionContext,
    screen_width: u32,
    screen_height: u32,
    detections_rx: Receiver<InferenceOutput>,
}

struct LoopMetrics {
    total_inference_time: f64,
    total_screencap_time: f64,
    loop_count: u32,
}

impl LoopMetrics {
    fn new() -> Self {
        Self {
            total_inference_time: 0.0,
            total_screencap_time: 0.0,
            loop_count: 0,
        }
    }

    fn update(&mut self, inference_time: f64, screencap_time: f64) {
        self.total_inference_time += inference_time;
        self.total_screencap_time += screencap_time;
        self.loop_count += 1;
    }

    fn print_and_reset(&mut self) {
        if self.loop_count > 0 {
            crate::info_text().write_fmt(format_args!("Average inference time: {:.2}ms\n", self.total_inference_time / self.loop_count as f64)).unwrap();
            crate::info_text().write_fmt(format_args!("Average screencap time: {:.2}ms\n", self.total_screencap_time / self.loop_count as f64)).unwrap();

            self.total_inference_time = 0.0;
            self.total_screencap_time = 0.0;
            self.loop_count = 0;
        }
    }
}

fn inference_loop(mut inference_engine: InferenceEngine, confidence_threshold: &f32, mut screen_capturer: ScreenCapturer, detections_tx: Sender<InferenceOutput>, running: Arc<AtomicBool>) -> Result<(), AimAssistError> {
    // unsafe {
    //     let mut task_index = 0;
    //     let task_name = CString::new("Games").unwrap();
    //     let handle = AvSetMmThreadCharacteristicsA(PCSTR(task_name.as_ptr() as *const _), &mut task_index).map_err(|e| AimAssistError::ThreadPriorityError(e))?;
    //     AvSetMmThreadPriority(handle, AVRT_PRIORITY_HIGH).ok().map_err(|e| AimAssistError::ThreadPriorityError(e))?;
    // }
    while running.load(Ordering::Relaxed) {
        let frame_start_time = Instant::now();

        let frame = match screen_capturer.capture_frame() {
            Ok(ScreenCaptureOutput::Available(frame)) | Ok(ScreenCaptureOutput::NotAvailable(frame)) => frame,
            Err(e) => {
                warn!("Failed to capture frame: {:?}", e);
                continue;
            }
        };

        let screencap_time = frame_start_time.elapsed().as_nanos() as f64 / 1_000_000.0;
        inference_engine.infer_screen_capture(frame, *confidence_threshold)?;
        let inference_time = frame_start_time.elapsed().as_nanos() as f64 / 1_000_000.0 - screencap_time;
        let detections = inference_engine.get_best_detections();

        if let Err(_) = detections_tx.send(InferenceOutput {
            detections,
            inference_time,
            screencap_time,
            frame_start_time,
        }) {
            break;
        };
    }

    Ok(())
}

impl AimAssistModule {
    pub fn new(engine: InferenceEngine, config: &'static mut AimAssistConfiguration) -> Result<Self, AimAssistError> {
        let screen_capturer = ScreenCapturer::create(0, 0, config.capture_width, config.capture_height)?;
        let mouse_mover = InterceptionContext::new();

        let screen_width = screen_capturer.screen_width;
        let screen_height = screen_capturer.screen_height;
        let (detections_tx, detections_rx) = mpsc::channel();
        let running = Arc::new(AtomicBool::new(true));

        let running_clone = running.clone();
        let p_confidence_threshold = unsafe { &(*(config as *mut AimAssistConfiguration)).confidence_threshold };
        let inference_thread = thread::spawn(move || inference_loop(engine, p_confidence_threshold, screen_capturer, detections_tx, running_clone));

        Ok(Self {
            config,
            running,
            _inference_thread: inference_thread,
            interception_ctx: mouse_mover,
            screen_width,
            screen_height,
            detections_rx,
        })
    }

    pub fn run(&mut self) -> Result<(), AimAssistError> {
        let screen_center_x = self.screen_width as f32 / 2.0;
        let screen_center_y = self.screen_height as f32 / 2.0;

        let _ = self.detections_rx.recv();
        crate::info_text().write_str("Aim assist is running.\n").unwrap();

        let mut loop_metrics = LoopMetrics::new();
        let mut accumulated_dx = 0.0f32;
        let mut accumulated_dy = 0.0f32;
        let mut cached_pos = None;
        let mut last_mouse_move_time = None;
        let mut last_bhop_time = None;
        while self.running.load(Ordering::SeqCst) {
            let start_time = Instant::now();
            self.config.activation_condition.update();
            self.config.smoothing_function.update();

            cached_pos = match self.detections_rx.try_recv() {
                Ok(InferenceOutput {
                    detections,
                    inference_time,
                    screencap_time,
                    frame_start_time,
                }) => {
                    loop_metrics.update(inference_time, screencap_time);

                    let class_to_target = match self.config.target {
                        AimAssistTarget::Head => 0,
                        AimAssistTarget::Torso => 1,
                    };

                    if let Some(detection) = detections.iter().find(|d| d.class_id as usize == class_to_target) {
                        let target_x = (detection.x_min + detection.x_max) / 2.0;
                        let target_y = detection.y_min + (detection.y_max - detection.y_min) * 0.35;

                        let (predicted_x, predicted_y) = self.config.target_predictor.predict(target_x, target_y, frame_start_time.elapsed().as_secs_f64());

                        Some((detection.offset_absolute_from_center(predicted_x, predicted_y, screen_center_x, screen_center_y), detection.distance_calculator()))
                    } else {
                        None
                    }
                }
                Err(mpsc::TryRecvError::Empty) => cached_pos,
                Err(_) => {
                    println!("Inference thread disconnected");
                    break;
                }
            };

            if self.config.activation_condition.should_aim_assist()
                && let Some(((x, y), distance_calculator)) = cached_pos
                && last_mouse_move_time.map(|t: Instant| t.elapsed().as_micros() > self.config.mouse_move_frequency_micros as u128).unwrap_or(true)
            {
                let distance = distance_calculator.get_distance_units((x, y), (screen_center_x, screen_center_y));
                let dx = x - screen_center_x;
                let dy = y - screen_center_y;

                let smoothing_factor = self.config.smoothing_function.calculate(distance as f64, start_time.elapsed().as_secs_f64()) as f32;
                let dx_smoothed = self.config.x_correction_multiplier() * dx * smoothing_factor + accumulated_dx;
                let dy_smoothed = self.config.y_correction_multiplier() * dy * smoothing_factor + accumulated_dy;

                let dx_rounded = dx_smoothed.round() as i32;
                let dy_rounded = dy_smoothed.round() as i32;

                if dx_rounded != 0 || dy_rounded != 0 {
                    self.interception_ctx.move_relative(dx_rounded, dy_rounded)?;
                    last_mouse_move_time = Some(Instant::now());
                }

                accumulated_dx = dx_smoothed - dx_rounded as f32;
                accumulated_dy = dy_smoothed - dy_rounded as f32;
            }

            if self.config.bhop_enabled && interception::is_vk_down(self.config.bhop_held_vk as i32) && last_bhop_time.map(|t: Instant| t.elapsed().as_micros() > self.config.bhop_frequency_micros as u128).unwrap_or(true) {
                self.interception_ctx.mouse_wheel_up(120)?;

                last_bhop_time = Some(Instant::now());
            }

            if loop_metrics.loop_count % 1000 == 0 {
                loop_metrics.print_and_reset();
            }
        }

        Ok(())
    }
}

pub fn create_aimassist_session(model_path: &str, accelerator: AimAssistAccelerator) -> Result<Session, AimAssistError> {
    let mut execution_providers = vec![];
    match accelerator {
        AimAssistAccelerator::TensorRT => {
            execution_providers.push(TensorRTExecutionProvider::default().build());
        }
        AimAssistAccelerator::CUDA => {
            execution_providers.push(CUDAExecutionProvider::default().build());
        }
        AimAssistAccelerator::None => {
            execution_providers.push(CPUExecutionProvider::default().build());
        }
    }

    Session::builder()
        .map_err(|_| AimAssistError::SessionCreationError)?
        .with_execution_providers(execution_providers)
        .map_err(|_| AimAssistError::SessionCreationError)?
        .with_optimization_level(ort::GraphOptimizationLevel::Level3)
        .unwrap()
        .commit_from_file(model_path)
        .map_err(|_| AimAssistError::SessionCreationError)
}
