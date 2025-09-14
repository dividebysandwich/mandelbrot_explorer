#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use mandelbrot_explorer::MandelbrotApp;

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([300.0, 200.0]),
        ..Default::default()
    };
    
    eframe::run_native(
        "Mandelbrot Explorer",
        options,
        Box::new(|cc| Ok(Box::new(MandelbrotApp::new(cc)))),
    )
}

