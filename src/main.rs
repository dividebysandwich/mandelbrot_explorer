#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::egui;
use egui::epaint::Hsva;
use egui::{Color32, ColorImage, Pos2, Rect, TextureHandle, Vec2};
use num_complex::Complex64;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

const MAX_ITERATIONS: u32 = 1000;
const ESCAPE_RADIUS_SQ: f64 = 100.0;

// The main application state.
struct MandelbrotApp {
    texture: Option<TextureHandle>,
    // The parameters of the view represented by the current texture.
    texture_center: Complex64,
    texture_zoom: f64,

    // The view we are currently displaying (interpolated).
    display_center: Complex64,
    display_zoom: f64,
    // The target view the animation is moving towards.
    target_center: Complex64,
    target_zoom: f64,

    // Holds the result from the background thread.
    new_image_and_params: Arc<Mutex<Option<(ColorImage, Complex64, f64)>>>,
    // Flag to prevent starting multiple calculations.
    is_calculating: bool,

    viewport_size: Vec2,
}

// Parameters passed to the background rendering thread.
#[derive(Clone)]
struct CalculationParams {
    center: Complex64,
    zoom: f64,
    size: [usize; 2],
    reference_c: Complex64,
    reference_orbit: Vec<Complex64>,
}

impl Default for MandelbrotApp {
    fn default() -> Self {
        let initial_zoom = 4.0;
        let initial_center = Complex64::new(-0.75, 0.0);

        Self {
            texture: None,
            texture_center: initial_center,
            texture_zoom: initial_zoom,
            display_center: initial_center,
            display_zoom: initial_zoom,
            target_center: initial_center,
            target_zoom: initial_zoom,
            new_image_and_params: Arc::new(Mutex::new(None)),
            is_calculating: true, // Start calculating the initial view right away
            viewport_size: Vec2::ZERO,
        }
    }
}

impl MandelbrotApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Default::default()
    }

    // Kicks off a background thread to render the Mandelbrot set.
    fn start_calculation(&mut self) {
        if self.is_calculating {
            return;
        }
        self.is_calculating = true;

        let reference_c = self.target_center;
        let mut reference_orbit = Vec::new();

        // Pre-compute the reference orbit.
        let mut z = Complex64::new(0.0, 0.0);
        for _ in 0..MAX_ITERATIONS {
            reference_orbit.push(z);
            z = z * z + reference_c;
            // Stop if the reference point escapes, creating a stable `MaxRefIteration`.
            if z.norm_sqr() > ESCAPE_RADIUS_SQ * 100.0 {
                break;
            }
        }

        let params = CalculationParams {
            center: self.target_center,
            zoom: self.target_zoom,
            size: [
                self.viewport_size.x as usize,
                self.viewport_size.y as usize,
            ],
            reference_c,
            reference_orbit,
        };

        let new_image_mutex = self.new_image_and_params.clone();
        rayon::spawn(move || {
            let image = render_mandelbrot_to_new_image(params.clone());
            *new_image_mutex.lock().unwrap() = Some((image, params.center, params.zoom));
        });
    }
}

impl eframe::App for MandelbrotApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Check for and integrate newly calculated image
        if let Ok(mut guard) = self.new_image_and_params.try_lock() {
            if let Some((new_image, center, zoom)) = guard.take() {
                self.is_calculating = false; // A calculation has finished.

                // Check if the received image is for the view we are currently interested in.
                // This prevents an old, slow calculation from overwriting a newer view.
                let is_relevant = (center.re - self.target_center.re).abs() < 1e-9 &&
                                  (center.im - self.target_center.im).abs() < 1e-9 &&
                                  (zoom - self.target_zoom).abs() / self.target_zoom < 1e-6;

                if is_relevant {
                    self.texture = Some(ctx.load_texture("mandelbrot", new_image, Default::default()));
                    self.texture_center = center;
                    self.texture_zoom = zoom;
                    // Snap the display to the new texture's parameters to stop the animation
                    self.display_center = center;
                    self.display_zoom = zoom;
                }
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            let panel_rect = ui.available_rect_before_wrap();

            // Resize Logic
            if self.viewport_size != panel_rect.size() && panel_rect.size().x > 0.0 && panel_rect.size().y > 0.0 {
                self.viewport_size = panel_rect.size();
                self.target_center = self.display_center;
                self.target_zoom = self.display_zoom;
                self.is_calculating = false;
                // A new calculation will be triggered by the check at the end of this function.
            }

            // Animate Display Parameters
            self.display_center.re = lerp(self.display_center.re, self.target_center.re, 0.2);
            self.display_center.im = lerp(self.display_center.im, self.target_center.im, 0.2);
            self.display_zoom = lerp(self.display_zoom, self.target_zoom, 0.2);

            if let Some(texture) = &self.texture {
                // Calculate UV Mapping for Animated Zoom
                let aspect_ratio = self.viewport_size.x / self.viewport_size.y;
                let tex_view_height = self.texture_zoom / aspect_ratio as f64;
                let disp_view_height = self.display_zoom / aspect_ratio as f64;

                let u_min = (self.display_center.re - 0.5 * self.display_zoom - (self.texture_center.re - 0.5 * self.texture_zoom)) / self.texture_zoom;
                let u_max = (self.display_center.re + 0.5 * self.display_zoom - (self.texture_center.re - 0.5 * self.texture_zoom)) / self.texture_zoom;
                let v_min = (self.display_center.im - 0.5 * disp_view_height - (self.texture_center.im - 0.5 * tex_view_height)) / tex_view_height;
                let v_max = (self.display_center.im + 0.5 * disp_view_height - (self.texture_center.im - 0.5 * tex_view_height)) / tex_view_height;

                let uv = Rect::from_min_max(Pos2::new(u_min as f32, v_min as f32), Pos2::new(u_max as f32, v_max as f32));
                
                let image_widget = egui::Image::new(&*texture)
                    .uv(uv)
                    .fit_to_exact_size(panel_rect.size());
                let response = ui.add(image_widget);
                let response = ui.interact(response.rect, ui.id().with("mandelbrot_interactive_area"), egui::Sense::click_and_drag());

                if response.dragged() {
                    let delta = response.drag_delta();
                    let complex_delta_re = (delta.x as f64 / self.viewport_size.x as f64) * self.display_zoom;
                    let complex_delta_im = (delta.y as f64 / self.viewport_size.y as f64) * disp_view_height;
                    self.target_center.re -= complex_delta_re;
                    self.target_center.im -= complex_delta_im;
                    self.display_center = self.target_center; // Snap display for immediate feedback
                    self.start_calculation();
                }

                if response.hovered() {
                    let scroll = ui.input(|i| i.raw_scroll_delta);
                    if scroll.y != 0.0 {
                        let zoom_factor = (scroll.y as f64 * 0.001).exp();
                        if let Some(hover_pos) = response.hover_pos() {
                            // The calculation must be based on the stable target, not the animating display values.
                            // This makes the zoom deterministic and centers it on the cursor.
                            let complex_pos = screen_to_complex(hover_pos, self.target_center, self.target_zoom, panel_rect);
                            self.target_center = complex_pos + (self.target_center - complex_pos) / zoom_factor;
                        }
                        self.target_zoom /= zoom_factor;
                        self.start_calculation();
                    }
                }

            } else {
                // Show a loading indicator if there's no texture yet
                ui.centered_and_justified(|ui| ui.spinner());
                self.start_calculation();
            }
            
            // After interaction, check if a new calculation is needed
            // This is a fallback for cases where an old calculation result was discarded.
            let target_is_stale = (self.target_center.re - self.texture_center.re).abs() > 1e-9 ||
                                  (self.target_center.im - self.texture_center.im).abs() > 1e-9 ||
                                  (self.target_zoom - self.texture_zoom).abs() / self.target_zoom > 1e-6;

            if !self.is_calculating && target_is_stale {
                self.start_calculation();
            }

            ctx.request_repaint();
        });

        // Side panel with basic controls and info
        egui::SidePanel::left("info_panel").show(ctx, |ui| {
            ui.heading("Mandelbrot Explorer");
            ui.separator();
            ui.label("Controls:");
            ui.label(" - Zoom: Mouse Wheel");
            ui.label(" - Pan: Click & Drag");
            ui.separator();
            ui.label("Current View:");
            ui.monospace(format!("Center Re: {:.12}", self.target_center.re));
            ui.monospace(format!("Center Im: {:.12}", self.target_center.im));
            ui.monospace(format!("Zoom: {:.2e}", self.target_zoom));
            ui.separator();
            ui.label(format!("Max Iterations: {}", MAX_ITERATIONS));
             if ui.button("Reset View").clicked() {
                let initial_center = Complex64::new(-0.75, 0.0);
                let initial_zoom = 4.0;
                self.target_center = initial_center;
                self.target_zoom = initial_zoom;
                self.display_center = initial_center;
                self.display_zoom = initial_zoom;
                self.is_calculating = false; // Allow new calc to start
                self.start_calculation();
            }
        });
    }
}

// Renders the Mandelbrot set to a new ColorImage. Can be run in a background thread.
fn render_mandelbrot_to_new_image(params: CalculationParams) -> ColorImage {
    let mut image = ColorImage::new(params.size, vec![Color32::TRANSPARENT; params.size[0] * params.size[1]]);
    let aspect_ratio = params.size[0] as f64 / params.size[1] as f64;
    let view_height = params.zoom / aspect_ratio;
    let use_perturbation = params.zoom < 1e-5;

    image.pixels.par_iter_mut().enumerate().for_each(|(i, pixel)| {
        let x = i % params.size[0];
        let y = i / params.size[0];

        let n = if use_perturbation {
            let offset_re = (x as f64 / params.size[0] as f64 - 0.5) * params.zoom;
            let offset_im = (y as f64 / params.size[1] as f64 - 0.5) * view_height;
            let delta = Complex64::new(offset_re, offset_im);
            let c = params.reference_c + delta;
            calculate_iterations_perturbed(&delta, &params.reference_orbit, c)
        } else {
            let re = params.center.re + (x as f64 / params.size[0] as f64 - 0.5) * params.zoom;
            let im = params.center.im + (y as f64 / params.size[1] as f64 - 0.5) * view_height;
            let c = Complex64::new(re, im);
            calculate_iterations_standard(c)
        };
        *pixel = smooth_color(n, MAX_ITERATIONS);
    });
    image
}


fn lerp(a: f64, b: f64, t: f64) -> f64 { a + (b - a) * t }

fn screen_to_complex(pos: Pos2, center: Complex64, zoom: f64, screen_rect: Rect) -> Complex64 {
    let screen_size = screen_rect.size();
    let aspect_ratio = screen_size.x / screen_size.y;
    let re = center.re + ((pos.x - screen_rect.min.x) as f64 / screen_size.x as f64 - 0.5) * zoom;
    let im = center.im + ((pos.y - screen_rect.min.y) as f64 / screen_size.y as f64 - 0.5) * (zoom / aspect_ratio as f64);
    Complex64::new(re, im)
}

fn calculate_iterations_standard(c: Complex64) -> f64 {
    let mut z = Complex64::new(0.0, 0.0);
    for i in 0..MAX_ITERATIONS {
        if z.norm_sqr() > ESCAPE_RADIUS_SQ {
            return i as f64 - (z.norm().log2().log2()) + 4.0;
        }
        z = z * z + c;
    }
    MAX_ITERATIONS as f64
}

fn calculate_iterations_perturbed(
    delta_c: &Complex64,
    ref_orbit: &[Complex64],
    c: Complex64,
) -> f64 {
    let mut delta_z = Complex64::new(0.0, 0.0);
    let mut z = Complex64::new(0.0, 0.0);
    let mut start_iter = 0;

    // Part 1: Perturbation Loop, as long as we have reference data
    for i in 0..ref_orbit.len() {
        let z_ref = ref_orbit[i];
        z = z_ref + delta_z;

        if z.norm_sqr() > ESCAPE_RADIUS_SQ {
            return i as f64 - (z.norm().log2().log2()) + 4.0;
        }

        // Rebase condition: If the delta grows too large, the approximation becomes
        // unstable. We must switch to the standard algorithm.
        if z.norm() < delta_z.norm() {
            start_iter = i + 1;
            break; 
        }

        delta_z = 2.0 * z_ref * delta_z + delta_z.powi(2) + delta_c;
        start_iter = i + 1;
    }

    // Part 2: Standard Iteration Fallback
    // This continues the calculation from where the perturbation loop left off.
    for i in start_iter..(MAX_ITERATIONS as usize) {
        if z.norm_sqr() > ESCAPE_RADIUS_SQ {
            return i as f64 - (z.norm().log2().log2()) + 4.0;
        }
        z = z * z + c;
    }

    MAX_ITERATIONS as f64
}

fn smooth_color(n: f64, max_iter: u32) -> Color32 {
    if (n - max_iter as f64).abs() < 1e-6 {
        return Color32::BLACK;
    }
    let hue = (n / 30.0).fract() as f32;
    let saturation = 0.8;
    let value = (n / max_iter as f64).sqrt() as f32;
    Hsva::new(hue, saturation, value.clamp(0.1, 1.0), 1.0).into()
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
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

