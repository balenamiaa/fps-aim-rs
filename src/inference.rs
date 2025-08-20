use crate::errors::InferenceError;
use crate::screen_capture::ScreenCaptureOutputAvailable;
use ort::session::{Session, SessionInputValue, SessionInputs, SessionOutputs};
use std::cell::RefCell;

#[derive(Debug, Clone, Copy)]
pub struct DetectionResult {
    pub class_id: i64,
    pub confidence: f32,
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
    pub image_width: usize,
    pub image_height: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct DetectionDistanceCalculator {
    pub width: f32,
    pub height: f32,
}

impl DetectionDistanceCalculator {
    pub fn get_distance_units(&self, p1: (f32, f32), p2: (f32, f32)) -> f32 {
        let box_magnitude = (self.width.powi(2) + self.height.powi(2)).sqrt();

        let dx = p2.0 - p1.0;
        let dy = p2.1 - p1.1;

        let delta_magnitude = (dx.powi(2) + dy.powi(2)).sqrt();

        delta_magnitude / box_magnitude
    }
}

impl DetectionResult {
    pub fn offset_relative_from_center(&self, x_norm: f32, y_norm: f32, center_x: f32, center_y: f32) -> (f32, f32) {
        let top = center_y - self.image_height as f32 / 2.0;
        let left = center_x - self.image_width as f32 / 2.0;

        (left + x_norm * self.image_width as f32, top + y_norm * self.image_height as f32)
    }

    pub fn offset_absolute_from_center(&self, x: f32, y: f32, center_x: f32, center_y: f32) -> (f32, f32) {
        let top = center_y - self.image_height as f32 / 2.0;
        let left = center_x - self.image_width as f32 / 2.0;

        (left + x, top + y)
    }

    pub fn distance_calculator(&self) -> DetectionDistanceCalculator {
        DetectionDistanceCalculator {
            width: self.x_max - self.x_min,
            height: self.y_max - self.y_min,
        }
    }
}

pub struct InferenceEngine {
    session: Session,
    config: InferenceConfig,
    detections_buffer: Box<[Option<DetectionResult>]>,
    _intermediate_buffer: RefCell<Box<[f32]>>,
    model_fixed_num_detections: usize,
    num_classes: usize,
}

unsafe impl<'a> Send for InferenceEngine {}

#[derive(Debug, Clone, Copy)]
pub struct InferenceConfig {
    pub num_detections: usize,
    pub image_width: usize,
    pub image_height: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            num_detections: InferenceEngine::DEFAULT_MAX_DETECTIONS,
            image_width: InferenceEngine::DEFAULT_IMAGE_WIDTH,
            image_height: InferenceEngine::DEFAULT_IMAGE_HEIGHT,
        }
    }
}

impl InferenceEngine {
    const DEFAULT_MAX_DETECTIONS: usize = 100;
    const DEFAULT_IMAGE_WIDTH: usize = 640;
    const DEFAULT_IMAGE_HEIGHT: usize = 640;

    pub fn new(session: Session, config: Option<InferenceConfig>) -> Result<Self, InferenceError> {
        let config = config.unwrap_or_default();
        let max_detections = config.num_detections;
        let image_width = config.image_width;
        let image_height = config.image_height;

        let inputs = &session.inputs;
        let outputs = &session.outputs;

        debug_assert!(inputs.len() == 1);
        debug_assert!(inputs[0].name == "images");

        let output_dims = outputs[0].output_type.tensor_shape().unwrap().clone();
        let max_class_index = output_dims[1] as usize;
        let num_classes = max_class_index - 4;
        let model_fixed_num_detections = output_dims[2] as usize;

        let intermediate_buffer = vec![0.0; image_width * image_height * 3].into_boxed_slice();
        let detections_buffer = vec![None; max_detections].into_boxed_slice();

        Ok(Self {
            session,
            config,
            detections_buffer,
            _intermediate_buffer: RefCell::new(intermediate_buffer),
            model_fixed_num_detections,
            num_classes,
        })
    }

    pub fn infer_screen_capture<'a>(&'a mut self, input: ScreenCaptureOutputAvailable, confidence_threshold: f32) -> Result<&'a mut [Option<DetectionResult>], InferenceError> {
        self.detections_buffer.iter_mut().for_each(|d| *d = None);

        let mut input = unsafe { input.data_as_ort_tensor() };
        let input_tensor = input.take_tensor().unwrap();
        let session_input_value = SessionInputValue::Owned(input_tensor.into_dyn());
        let session_inputs: SessionInputs<1> = SessionInputs::ValueMap(vec![("images".into(), session_input_value)]);
        let output = self.session.run(session_inputs).map_err(InferenceError::InferenceError)?;
        Self::parse_output(self.model_fixed_num_detections, self.config.image_width, self.config.image_height, &mut self.detections_buffer, output, confidence_threshold)
    }

    fn parse_output<'a>(model_fixed_num_detections: usize, image_width: usize, image_height: usize, detections_buffer: &'a mut Box<[Option<DetectionResult>]>, output: SessionOutputs, confidence_threshold: f32) -> Result<&'a mut [Option<DetectionResult>], InferenceError> {
        let (_, dense_tensor_as_slice) = output["output0"].try_extract_tensor::<half::f16>().map_err(InferenceError::InferenceError)?;

        let valid_indices: Vec<(usize, usize)> = (4 * model_fixed_num_detections..6 * model_fixed_num_detections)
            .filter_map(|index| {
                let class_id = index / model_fixed_num_detections - 4;
                let detection_index = index % model_fixed_num_detections;
                let confidence: f32 = dense_tensor_as_slice[index].into();
                if confidence > confidence_threshold { Some((detection_index, class_id)) } else { None }
            })
            .collect();

        let mut count = 0;
        {
            for (j, class_id) in valid_indices {
                let x_center: f32 = dense_tensor_as_slice[0 * model_fixed_num_detections + j].into();
                let y_center: f32 = dense_tensor_as_slice[1 * model_fixed_num_detections + j].into();
                let width: f32 = dense_tensor_as_slice[2 * model_fixed_num_detections + j].into();
                let height: f32 = dense_tensor_as_slice[3 * model_fixed_num_detections + j].into();
                let confidence: f32 = dense_tensor_as_slice[4 * model_fixed_num_detections + class_id * model_fixed_num_detections + j].into();

                detections_buffer[count] = Some(DetectionResult {
                    class_id: class_id as i64,
                    confidence,
                    x_min: x_center - width / 2.0,
                    y_min: y_center - height / 2.0,
                    x_max: x_center + width / 2.0,
                    y_max: y_center + height / 2.0,
                    image_width,
                    image_height,
                });
                count += 1;
            }
        }

        Ok(&mut detections_buffer[..count])
    }

    pub fn get_best_detections(&self) -> Vec<DetectionResult> {
        let mut best_detections = Vec::new();
        let mut best_scores = vec![0.0; self.num_classes];
        let mut best_results = vec![None; self.num_classes];

        for detection in self.detections_buffer.iter().filter_map(|&d| d) {
            let class_id = detection.class_id as usize;
            if detection.confidence > best_scores[class_id] {
                best_scores[class_id] = detection.confidence;
                best_results[class_id] = Some(detection);
            }
        }

        for result in best_results.into_iter().filter_map(|d| d) {
            best_detections.push(result);
        }

        best_detections
    }

    fn _save_ndarray_as_image(tensor: &ndarray::Array4<f32>, path: &str) {
        let (batch_size, channels, height, width) = (tensor.shape()[0], tensor.shape()[1], tensor.shape()[2], tensor.shape()[3]);
        assert_eq!(batch_size, 1);
        assert_eq!(channels, 3);

        let mut img: image::RgbImage = image::ImageBuffer::new(width as u32, height as u32);

        for y in 0..height {
            for x in 0..width {
                let r = tensor[[0, 0, y, x]];
                let g = tensor[[0, 1, y, x]];
                let b = tensor[[0, 2, y, x]];
                img.put_pixel(x as u32, y as u32, image::Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]));
            }
        }

        img.save(std::path::Path::new(path)).unwrap();
    }
}
