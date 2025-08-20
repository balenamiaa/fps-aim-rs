// #![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
#![feature(once_cell_get_mut)]

use std::error::Error;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::{fs, path::PathBuf};

use crate::interception::is_button_held_mouse5;
use crate::screen_capture::ScreenCaptureOutput;
use crate::screen_capture::ScreenCapturer;

mod errors;
mod inference;
mod interception;
mod screen_capture;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    env_logger::init();

    // Ensure output directory exists
    let output_dir = PathBuf::from("collected_images");
    if !output_dir.exists() {
        fs::create_dir_all(&output_dir)?;
    }

    let mut screen_capturer = ScreenCapturer::create(0, 0, 640, 640)?;
    let mut last_capture_time = Instant::now() - Duration::from_millis(250);

    loop {
        if is_button_held_mouse5() && last_capture_time.elapsed() >= Duration::from_millis(250) {
            match screen_capturer.capture_frame() {
                Ok(ScreenCaptureOutput::Available(mut available)) => {
                    let view = available.data_as_ndarray(); // shape: (1, 3, width, height), f16 in [0,1]
                    let width = view.shape()[2] as u32;
                    let height = view.shape()[3] as u32;

                    let mut img: image::RgbImage = image::ImageBuffer::new(width, height);
                    for y in 0..height {
                        for x in 0..width {
                            let r: f32 = view[[0, 0, y as usize, x as usize]].to_f32();
                            let g: f32 = view[[0, 1, y as usize, x as usize]].to_f32();
                            let b: f32 = view[[0, 2, y as usize, x as usize]].to_f32();
                            let ru = (r.clamp(0.0, 1.0) * 255.0).round() as u8;
                            let gu = (g.clamp(0.0, 1.0) * 255.0).round() as u8;
                            let bu = (b.clamp(0.0, 1.0) * 255.0).round() as u8;
                            img.put_pixel(x, y, image::Rgb([ru, gu, bu]));
                        }
                    }

                    let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
                    let mut path = output_dir.clone();
                    path.push(format!("captured_frame_{}.png", ts));
                    if let Err(e) = img.save(&path) {
                        log::warn!("Failed to save image {:?}: {}", path, e);
                    } else {
                        log::info!("Saved {:?}", path);
                    }

                    last_capture_time = Instant::now();
                }
                Ok(ScreenCaptureOutput::NotAvailable(_)) => {
                    log::info!("No new frame ready; try again next tick");
                }
                Err(e) => {
                    log::warn!("Capture error: {}", e);
                }
            }
        }

        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}
