use eframe::egui;
use egui::epaint::Hsva;
use egui::{Color32, ColorImage, Pos2, Rect, TextureHandle, Vec2};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::{Arc, Mutex};
use rayon::prelude::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

const MAX_ITERATIONS: u32 = 1000;
// Note: ESCAPE_RADIUS_SQ is now a `Double` constant.
const ESCAPE_RADIUS_SQ: Double = Double { hi: 100.0, lo: 0.0 };

// High-Precision Number Implementation (Double-Double Arithmetic)
// A high-precision number represented by the sum of two f64s.
#[derive(Clone, Copy, Debug, Default)]
struct Double {
    hi: f64, // The most significant part of the number
    lo: f64, // The error term from the last operation
}

// A high-precision complex number.
#[derive(Clone, Copy, Debug, Default)]
struct ComplexDouble {
    re: Double,
    im: Double,
}

impl Double {
    // Exact square of a Double
    fn sqr(self) -> Self {
        let p1 = self.hi * self.hi;
        let p2 = self.hi * 2.0 * self.lo;
        let p3 = self.lo * self.lo;
        let (s1, e1) = two_sum(p1, p2);
        let (s2, e2) = two_sum(s1, p3);
        let e = e1 + e2;
        let (hi, lo) = quick_two_sum(s2, e);
        Double { hi, lo }
    }
}

impl From<f64> for Double {
    fn from(n: f64) -> Self {
        Self { hi: n, lo: 0.0 }
    }
}

// Error-free sum of two f64s
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    (s, e)
}

// A slightly faster version of two_sum when |a| >= |b|
fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let e = b - (s - a);
    (s, e)
}

// Error-free product of two f64s
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let (a_hi, a_lo) = split(a);
    let (b_hi, b_lo) = split(b);
    let e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    (p, e)
}

// Splits a f64 into two 26-bit halves
fn split(a: f64) -> (f64, f64) {
    let t = (27.0_f64.exp2() + 1.0) * a;
    let a_hi = t - (t - a);
    let a_lo = a - a_hi;
    (a_hi, a_lo)
}

impl Neg for Double {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

impl Add for Double {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let (s, e) = two_sum(self.hi, rhs.hi);
        let (hi, lo) = quick_two_sum(s, e + self.lo + rhs.lo);
        Double { hi, lo }
    }
}

impl Sub for Double {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl Mul for Double {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let (p, e) = two_prod(self.hi, rhs.hi);
        let (hi, lo) = quick_two_sum(p, e + self.hi * rhs.lo + self.lo * rhs.hi);
        Double { hi, lo }
    }
}

impl Div for Double {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let q1 = self.hi / rhs.hi;
        let r = self - (rhs * q1.into());
        let q2 = r.hi / rhs.hi;
        let (hi, lo) = quick_two_sum(q1, q2);
        Double { hi, lo }
    }
}

impl ComplexDouble {
    fn new(re: Double, im: Double) -> Self { Self { re, im } }
    
    // Square of a ComplexDouble
    fn sqr(self) -> Self {
        // (re + im*i)^2 = re^2 - im^2 + 2*re*im*i
        let re_sq = self.re.sqr();
        let im_sq = self.im.sqr();
        let re_im = self.re * self.im;
        Self {
            re: re_sq - im_sq,
            im: re_im + re_im,
        }
    }
    
    fn norm_sqr(self) -> Double {
        self.re.sqr() + self.im.sqr()
    }
}

impl Add for ComplexDouble {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self { re: self.re + rhs.re, im: self.im + rhs.im }
    }
}


// The main application state
pub struct MandelbrotApp {
    texture: Option<TextureHandle>,
    // The parameters of the view represented by the current texture.
    texture_center: ComplexDouble,
    texture_zoom: f64,
    // The view we are currently displaying (interpolated).
    display_center: ComplexDouble,
    display_zoom: f64,
    // The target view the animation is moving towards.
    target_center: ComplexDouble,
    target_zoom: f64,
    // Holds the result from the background thread.
    new_image_and_params: Arc<Mutex<Option<(ColorImage, ComplexDouble, f64)>>>,
    // Flag to prevent starting multiple calculations.
    is_calculating: bool,
    viewport_size: Vec2,
}

// Parameters passed to the background rendering thread.
#[derive(Clone)]
struct CalculationParams {
    center: ComplexDouble,
    zoom: f64,
    size: [usize; 2],
}

impl Default for MandelbrotApp {
    fn default() -> Self {
        let initial_zoom = 4.0;
        let initial_center = ComplexDouble::new((-0.75).into(), 0.0.into());

        Self {
            texture: None,
            texture_center: initial_center,
            texture_zoom: initial_zoom,
            display_center: initial_center,
            display_zoom: initial_zoom,
            target_center: initial_center,
            target_zoom: initial_zoom,
            new_image_and_params: Arc::new(Mutex::new(None)),
            is_calculating: true,
            viewport_size: Vec2::ZERO,
        }
    }
}

impl MandelbrotApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Default::default()
    }

    // Kicks off a background thread to render the Mandelbrot set.
    fn start_calculation(&mut self) {
        if self.is_calculating {
            return;
        }
        self.is_calculating = true;

        let params = CalculationParams {
            center: self.target_center,
            zoom: self.target_zoom,
            size: [
                self.viewport_size.x as usize,
                self.viewport_size.y as usize,
            ],
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
                let is_relevant = (center.re.hi - self.target_center.re.hi).abs() < 1e-15 &&
                                  (center.im.hi - self.target_center.im.hi).abs() < 1e-15 &&
                                  (zoom - self.target_zoom).abs() / self.target_zoom < 1e-9;

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
            self.display_center.re.hi = lerp(self.display_center.re.hi, self.target_center.re.hi, 0.2);
            self.display_center.im.hi = lerp(self.display_center.im.hi, self.target_center.im.hi, 0.2);
            self.display_zoom = lerp(self.display_zoom, self.target_zoom, 0.2);

            if let Some(texture) = &self.texture {
                // Calculate UV Mapping for Animated Zoom
                let aspect_ratio = self.viewport_size.x / self.viewport_size.y;
                let u_min = (self.display_center.re.hi - 0.5 * self.display_zoom - (self.texture_center.re.hi - 0.5 * self.texture_zoom)) / self.texture_zoom;
                let u_max = (self.display_center.re.hi + 0.5 * self.display_zoom - (self.texture_center.re.hi - 0.5 * self.texture_zoom)) / self.texture_zoom;
                let aspect_ratio_f64 = aspect_ratio as f64;
                let v_min = (self.display_center.im.hi - 0.5 * self.display_zoom / aspect_ratio_f64 - (self.texture_center.im.hi - 0.5 * self.texture_zoom / aspect_ratio_f64)) / (self.texture_zoom / aspect_ratio_f64);
                let v_max = (self.display_center.im.hi + 0.5 * self.display_zoom / aspect_ratio_f64 - (self.texture_center.im.hi - 0.5 * self.texture_zoom / aspect_ratio_f64)) / (self.texture_zoom / aspect_ratio_f64);

                let uv = Rect::from_min_max(Pos2::new(u_min as f32, v_min as f32), Pos2::new(u_max as f32, v_max as f32));
                
                let image_widget = egui::Image::new(&*texture).uv(uv).fit_to_exact_size(panel_rect.size());
                let response = ui.add(image_widget);
                let response = ui.interact(response.rect, ui.id().with("mandelbrot_interactive_area"), egui::Sense::click_and_drag());

                if response.dragged() {
                    let delta = response.drag_delta();
                    let complex_delta_re = (delta.x as f64 / self.viewport_size.x as f64) * self.display_zoom;
                    let complex_delta_im = (delta.y as f64 / self.viewport_size.y as f64) * (self.display_zoom / aspect_ratio as f64);
                    self.target_center.re = self.target_center.re - complex_delta_re.into();
                    self.target_center.im = self.target_center.im - complex_delta_im.into();
                    self.display_center = self.target_center;
                    self.start_calculation();
                }

                if response.hovered() {
                    let scroll = ui.input(|i| i.raw_scroll_delta);
                    if scroll.y != 0.0 {
                        let zoom_factor = (scroll.y as f64 * 0.001).exp();
                        if let Some(hover_pos) = response.hover_pos() {
                            let complex_pos = screen_to_complex_double(hover_pos, self.target_center, self.target_zoom, panel_rect);
                            let center_diff_re = self.target_center.re - complex_pos.re;
                            let center_diff_im = self.target_center.im - complex_pos.im;
                            self.target_center.re = complex_pos.re + center_diff_re * (1.0 / zoom_factor).into();
                            self.target_center.im = complex_pos.im + center_diff_im * (1.0 / zoom_factor).into();
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
            let target_is_stale = (self.target_center.re.hi - self.texture_center.re.hi).abs() > 1e-15 ||
                                  (self.target_center.im.hi - self.texture_center.im.hi).abs() > 1e-15 ||
                                  (self.target_zoom - self.texture_zoom).abs() / self.target_zoom > 1e-9;

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
            ui.monospace(format!("Center Re: {:.16}", self.target_center.re.hi));
            ui.monospace(format!("Center Im: {:.16}", self.target_center.im.hi));
            ui.monospace(format!("Zoom: {:.2e}", self.target_zoom));
            ui.separator();
            ui.label(format!("Max Iterations: {}", MAX_ITERATIONS));
             if ui.button("Reset View").clicked() {
                let initial_center = ComplexDouble::new((-0.75).into(), 0.0.into());
                let initial_zoom = 4.0;
                self.target_center = initial_center;
                self.target_zoom = initial_zoom;
                self.display_center = initial_center;
                self.display_zoom = initial_zoom;
                self.is_calculating = false;
                self.start_calculation();
            }
        });
    }
}

/// Renders the Mandelbrot set to a new ColorImage using high-precision math.
fn render_mandelbrot_to_new_image(params: CalculationParams) -> ColorImage {
    let mut image = ColorImage::new(
        params.size,
        vec![Color32::BLACK; params.size[0] * params.size[1]],
    );
    let aspect_ratio = params.size[0] as f64 / params.size[1] as f64;
    let zoom_dd: Double = params.zoom.into();
    let view_height_dd: Double = zoom_dd * (1.0 / aspect_ratio).into();

    image.pixels.par_iter_mut().enumerate().for_each(|(i, pixel)| {
        let x = i % params.size[0];
        let y = i / params.size[0];

        let re_offset = (Double::from(x as f64) / Double::from(params.size[0] as f64) - Double::from(0.5)) * zoom_dd;
        let im_offset = (Double::from(y as f64) / Double::from(params.size[1] as f64) - Double::from(0.5)) * view_height_dd;
        
        let c = ComplexDouble::new(params.center.re + re_offset, params.center.im + im_offset);
        let n = calculate_iterations_high_precision(c);
        
        *pixel = smooth_color(n, MAX_ITERATIONS);
    });
    image
}



fn lerp(a: f64, b: f64, t: f64) -> f64 { a + (b - a) * t }

fn screen_to_complex_double(pos: Pos2, center: ComplexDouble, zoom: f64, screen_rect: Rect) -> ComplexDouble {
    let screen_size = screen_rect.size();
    let aspect_ratio = screen_size.x / screen_size.y;
    let zoom_dd: Double = zoom.into();

    let re_offset = (Double::from((pos.x - screen_rect.min.x) as f64) / Double::from(screen_size.x as f64) - Double::from(0.5)) * zoom_dd;
    let im_offset = (Double::from((pos.y - screen_rect.min.y) as f64) / Double::from(screen_size.y as f64) - Double::from(0.5)) * zoom_dd * Double::from(1.0 / aspect_ratio as f64);
    
    ComplexDouble::new(center.re + re_offset, center.im + im_offset)
}

fn calculate_iterations_high_precision(c: ComplexDouble) -> f64 {
    let mut z = ComplexDouble::default();
    for i in 0..MAX_ITERATIONS {
        let norm_sq = z.norm_sqr();
        if norm_sq.hi > ESCAPE_RADIUS_SQ.hi {
            // Use the high part for the smooth coloring calculation
            return i as f64 - (norm_sq.hi.sqrt().log2().log2()) + 4.0;
        }
        z = z.sqr() + c;
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

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WebHandle {
    runner: eframe::WebRunner,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WebHandle {
    /// Installs a panic hook, then returns.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        // Redirect [`log`] message to `console.log` and friends:
        eframe::WebLogger::init(log::LevelFilter::Debug).ok();

        Self {
            runner: eframe::WebRunner::new(),
        }
    }

    /// Call this once from JavaScript to start your app.
    #[wasm_bindgen]
    pub async fn start(
        &self,
        canvas: web_sys::HtmlCanvasElement,
    ) -> Result<(), wasm_bindgen::JsValue> {
        self.runner
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(|cc| Ok(Box::new(MandelbrotApp::new(cc)))),
            )
            .await
    }
}
