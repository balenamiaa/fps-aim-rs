// #![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
#![feature(once_cell_get_mut)]
#![feature(let_chains)]

use std::fmt::Write;
use std::sync::OnceLock;
use std::{error::Error, thread::spawn};

use aim_assist::{create_aimassist_session, AimAssistConfiguration, AimAssistModule, AimAssistTarget, SmoothingFunctions};
use eframe::egui;
use inference::{InferenceConfig, InferenceEngine};
mod aim_assist;
mod errors;
mod inference;
mod interception;
mod screen_capture;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    env_logger::init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    let dylib_path = std::env::current_dir()?.join("onnxruntime-win-x64-gpu-1.18.0\\lib\\onnxruntime.dll").to_str().unwrap().to_string();
    ort::init_from(dylib_path).commit()?;

    let config_mut_ref: *mut AimAssistConfiguration = Box::leak(Box::new(AimAssistConfiguration::cs2_configuration()?)) as *mut _;

    let aa_config = unsafe { &mut *config_mut_ref };
    let _ = spawn(move || {
        if let Err(e) = aim_assist_fn(aa_config) {
            eprintln!("Error: {}", e);
        }
    });

    info_text().write_str("Starting aim assist...\n").unwrap();
    let ef_config = unsafe { &mut *config_mut_ref };
    eframe::run_native("CS2 Cheetos", options, Box::new(move |_cc| Box::new(GUIContext::new(ef_config)))).map_err(|e| e.to_string())?;

    // aim_assist_thread.join().unwrap()?;
    Ok(())
}

fn aim_assist_fn(config: &'static mut AimAssistConfiguration) -> Result<(), Box<dyn Error + Send + Sync>> {
    let engine = InferenceEngine::new(
        create_aimassist_session("v1_f16_nano.onnx", config.accelerator)?,
        Some(InferenceConfig {
            image_width: 320,
            image_height: 320,
            ..Default::default()
        }),
    )?;
    let mut aim_assist_module = AimAssistModule::new(engine, config)?;
    aim_assist_module.run()?;

    Ok(())
}

struct GUIContext {
    config: &'static mut AimAssistConfiguration,
}

impl GUIContext {
    pub fn new(config: &'static mut AimAssistConfiguration) -> Self {
        Self { config }
    }
}

impl eframe::App for GUIContext {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Load Configuration").clicked() {
                    self.config.load().unwrap();
                }
                if ui.button("Save Configuration").clicked() {
                    self.config.save().unwrap();
                }
            });

            ui.horizontal(|ui| {
                ui.label("Confidence Threshold");
                ui.add(egui::Slider::new(&mut self.config.confidence_threshold, 0.0..=1.0).text("Confidence Threshold"));
            });

            ui.horizontal(|ui| {
                ui.label("Target");
                egui::ComboBox::from_id_source("target").selected_text(self.config.target.to_string()).show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.config.target, AimAssistTarget::Head, "Head");
                    ui.selectable_value(&mut self.config.target, AimAssistTarget::Torso, "Torso");
                });
            });

            ui.horizontal(|ui| {
                ui.label("Bunnyhop Enabled");
                ui.checkbox(&mut self.config.bhop_enabled, "");
            });

            ui.horizontal(|ui| {
                ui.label("Mouse Movement Frequency (micros)");
                ui.add(egui::Slider::new(&mut self.config.mouse_move_frequency_micros, 500..=3_000).text("Mouse Movement Frequency"));
            });

            ui.horizontal(|ui| {
                ui.label("Bunny Hop Frequency (micros)");
                ui.add(egui::Slider::new(&mut self.config.bhop_frequency_micros, 1500..=5_000).text("Bunny Hop Frequency"));
            });

            ui.horizontal(|ui| {
                ui.label("Sensitivity X");
                ui.add(egui::Slider::new(&mut self.config.sensitivity.0, 0.0..=10.0).text("Sensitivity"));
            });

            ui.horizontal(|ui| {
                ui.label("Sensitivity Y");
                ui.add(egui::Slider::new(&mut self.config.sensitivity.1, 0.0..=10.0).text("Sensitivity"));
            });

            ui.horizontal(|ui| {
                ui.label("DPI");
                ui.add(egui::Slider::new(&mut self.config.dpi, 0.0..=10000.0).text("DPI"));
            });

            match &mut self.config.smoothing_function {
                SmoothingFunctions::ProgressiveLinearInterpolation(smoothing) => {
                    ui.horizontal(|ui| {
                        ui.label("Progress Factor");
                        ui.add(egui::Slider::new(&mut smoothing.progress_factor, 0.0..=15.0).text("Progress Factor"));
                    });

                    let distances = &mut smoothing.base_smoothing.distances;
                    let multipliers = &mut smoothing.base_smoothing.multipliers;
                    ui.label("Distances - Multipliers");

                    for (i, (distance, multiplier)) in distances.iter_mut().zip(multipliers.iter_mut()).enumerate() {
                        ui.horizontal(|ui| {
                            ui.add(egui::Slider::new(distance, 0.0..=5.0).step_by(0.01).text(format!("Distance {}", i)));
                            ui.add(egui::Slider::new(multiplier, 0.0..=1.0).step_by(0.0001).text(format!("Multiplier {}", i)));
                        });
                    }
                }
                _ => {}
            }

            ui.separator();
            ui.with_layout(egui::Layout::left_to_right(egui::Align::Min).with_cross_justify(true), |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let job = egui::text::LayoutJob::single_section(info_text().clone(), egui::TextFormat::default());
                    ui.add(egui::Label::new(job));
                })
            });
        });
    }
}

pub fn info_text() -> &'static mut String {
    static mut INFO_TEXT: OnceLock<String> = OnceLock::new();
    unsafe { INFO_TEXT.get_mut_or_init(|| "".to_string()) }
}
