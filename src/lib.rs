use eframe::egui;
use egui::epaint::Hsva;
use egui::{Color32, ColorImage, Pos2, Rect, TextureHandle, Vec2};
use std::sync::{Arc, Mutex};
use rayon::prelude::*;
use astro_float::{BigFloat, RoundingMode, Sign};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

// Rounding mode used for every arbitrary-precision operation.
const RM: RoundingMode = RoundingMode::ToEven;
// Pixel escape radius, squared (|z|^2 > 100  =>  |z| > 10).
const ESCAPE_RADIUS_SQ: f64 = 100.0;
// Iteration budget at the starting zoom, and the hard cap for very deep zooms.
const BASE_ITERATIONS: f64 = 1000.0;
const MAX_ITERATIONS_CAP: u32 = 50_000;

// A complex number stored at arbitrary precision. Only the view center / the
// reference orbit are kept at this precision; everything per-pixel is plain f64.
#[derive(Clone)]
struct HpComplex {
    re: BigFloat,
    im: BigFloat,
}

impl HpComplex {
    fn from_f64(re: f64, im: f64, prec: usize) -> Self {
        Self {
            re: BigFloat::from_f64(re, prec),
            im: BigFloat::from_f64(im, prec),
        }
    }
}

// Number of mantissa bits needed to resolve a view of the given width (`zoom`).
// Grows by ~1 bit per halving of the zoom, with margin, and rounds up to a word.
fn precision_for_zoom(zoom: f64) -> usize {
    let depth_bits = (-zoom.log2()).max(0.0) as usize;
    let bits = depth_bits + 64; // headroom above the visible depth
    ((bits + 63) / 64 * 64).max(64)
}

// Iteration count scaled to the zoom depth (deeper views need far more detail).
fn max_iter_for_zoom(zoom: f64) -> u32 {
    let depth_decades = (-(zoom / 4.0).log10()).max(0.0);
    let iters = BASE_ITERATIONS + 400.0 * depth_decades;
    (iters.min(MAX_ITERATIONS_CAP as f64)) as u32
}

// Convert a BigFloat to the nearest f64 (round toward zero).
//
// astro-float has no public `to_f64`, but the mantissa is stored little-endian
// and normalised so the most-significant word has its top bit set. The value is
// therefore `sign * msw * 2^(exponent - 64)`, where `msw` is the last word.
fn bf_to_f64(x: &BigFloat) -> f64 {
    let digits = match x.mantissa_digits() {
        Some(d) if !d.is_empty() => d,
        _ => return 0.0,
    };
    let msw = digits[digits.len() - 1];
    if msw == 0 {
        return 0.0;
    }
    let exp = match x.exponent() {
        Some(e) => e as f64,
        None => return 0.0,
    };
    let sign = if x.sign() == Some(Sign::Neg) { -1.0 } else { 1.0 };
    sign * (msw as f64) * (exp - 64.0).exp2()
}

// (a - b) reduced to f64. The operands are near-equal high-precision values, so
// the difference itself is small and representable in f64.
fn hp_diff_f64(a: &BigFloat, b: &BigFloat, prec: usize) -> f64 {
    bf_to_f64(&a.sub(b, prec, RM))
}

// a + d, where d is a small f64 offset added at high precision.
fn hp_add_f64(a: &BigFloat, d: f64, prec: usize) -> BigFloat {
    a.add(&BigFloat::from_f64(d, prec), prec, RM)
}

// Compute the reference orbit Z_0, Z_1, ... for the view center at high
// precision, storing each point rounded to an f64 complex for the
// perturbation pass. Stops early if the reference escapes.
fn compute_reference_orbit(center: &HpComplex, max_iter: u32, prec: usize) -> Vec<(f64, f64)> {
    let mut zr = BigFloat::from_f64(0.0, prec);
    let mut zi = BigFloat::from_f64(0.0, prec);
    let mut orbit = Vec::with_capacity(max_iter as usize + 1);
    orbit.push((0.0, 0.0)); // Z_0 = 0

    for _ in 0..max_iter {
        // z = z^2 + c
        let zr2 = zr.mul(&zr, prec, RM);
        let zi2 = zi.mul(&zi, prec, RM);
        let zrzi = zr.mul(&zi, prec, RM);
        let new_zr = zr2.sub(&zi2, prec, RM).add(&center.re, prec, RM);
        let new_zi = zrzi.add(&zrzi, prec, RM).add(&center.im, prec, RM); // 2*zr*zi + ci
        zr = new_zr;
        zi = new_zi;

        let fr = bf_to_f64(&zr);
        let fi = bf_to_f64(&zi);
        orbit.push((fr, fi));
        if fr * fr + fi * fi > ESCAPE_RADIUS_SQ {
            break;
        }
    }
    orbit
}

// Iterate a single pixel as a delta from the reference orbit (perturbation
// theory). All arithmetic here is f64, which is what makes deep zooms fast.
//
//   Δz_{n+1} = 2·Z_n·Δz_n + Δz_n^2 + Δc
//
// Zhuoran's "rebasing" keeps the delta well-conditioned and avoids glitches:
// whenever the full value z = Z + Δz becomes smaller than the delta itself (or
// we reach the end of the stored reference), we restart the reference index and
// fold the full value back into the delta. Returns a smooth (fractional)
// iteration count.
fn perturbation_iterations(orbit: &[(f64, f64)], dcr: f64, dci: f64, max_iter: u32) -> f64 {
    let ref_len = orbit.len();
    if ref_len < 2 {
        return max_iter as f64;
    }

    let mut dzr = 0.0f64;
    let mut dzi = 0.0f64;
    let mut ref_i = 0usize;

    for n in 1..=max_iter {
        let (xr, xi) = orbit[ref_i];
        // Δz = 2·Z·Δz + Δz^2 + Δc
        let two_x_dz_r = 2.0 * (xr * dzr - xi * dzi);
        let two_x_dz_i = 2.0 * (xr * dzi + xi * dzr);
        let dz2_r = dzr * dzr - dzi * dzi;
        let dz2_i = 2.0 * dzr * dzi;
        dzr = two_x_dz_r + dz2_r + dcr;
        dzi = two_x_dz_i + dz2_i + dci;
        ref_i += 1;

        // Full value z = Z_{ref_i} + Δz.
        let (xr_next, xi_next) = orbit[ref_i];
        let zr = xr_next + dzr;
        let zi = xi_next + dzi;
        let z_norm = zr * zr + zi * zi;

        if z_norm > ESCAPE_RADIUS_SQ {
            return n as f64 - z_norm.sqrt().log2().log2() + 4.0;
        }

        // Rebase when the delta dominates the full value, or the reference runs out.
        let dz_norm = dzr * dzr + dzi * dzi;
        if z_norm < dz_norm || ref_i >= ref_len - 1 {
            dzr = zr;
            dzi = zi;
            ref_i = 0;
        }
    }
    max_iter as f64
}


// The main application state
pub struct MandelbrotApp {
    texture: Option<TextureHandle>,
    // The parameters of the view represented by the current texture.
    texture_center: HpComplex,
    texture_zoom: f64,
    // The view we are currently displaying (interpolated).
    display_center: HpComplex,
    display_zoom: f64,
    // The target view the animation is moving towards.
    target_center: HpComplex,
    target_zoom: f64,
    // Holds the result from the background thread.
    new_image_and_params: Arc<Mutex<Option<(ColorImage, HpComplex, f64)>>>,
    // Flag to prevent starting multiple calculations.
    is_calculating: bool,
    viewport_size: Vec2,
}

// Parameters passed to the background rendering thread.
#[derive(Clone)]
struct CalculationParams {
    center: HpComplex,
    zoom: f64,
    size: [usize; 2],
}

impl Default for MandelbrotApp {
    fn default() -> Self {
        let initial_zoom = 4.0;
        let prec = precision_for_zoom(initial_zoom);
        let initial_center = HpComplex::from_f64(-0.75, 0.0, prec);

        Self {
            texture: None,
            texture_center: initial_center.clone(),
            texture_zoom: initial_zoom,
            display_center: initial_center.clone(),
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
            center: self.target_center.clone(),
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
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        // Precision sufficient for the deepest view currently in flight.
        let prec = precision_for_zoom(self.target_zoom.min(self.display_zoom));

        // Check for and integrate newly calculated image
        if let Ok(mut guard) = self.new_image_and_params.try_lock() {
            if let Some((new_image, center, zoom)) = guard.take() {
                self.is_calculating = false; // A calculation has finished.

                // Check if the received image is for the view we are currently interested in.
                // This prevents an old, slow calculation from overwriting a newer view.
                let ddr = hp_diff_f64(&center.re, &self.target_center.re, prec);
                let ddi = hp_diff_f64(&center.im, &self.target_center.im, prec);
                let is_relevant = ddr.abs() < zoom * 1e-6
                    && ddi.abs() < zoom * 1e-6
                    && (zoom - self.target_zoom).abs() / self.target_zoom < 1e-9;

                if is_relevant {
                    self.texture = Some(ctx.load_texture("mandelbrot", new_image, Default::default()));
                    self.texture_center = center.clone();
                    self.texture_zoom = zoom;
                    // Snap the display to the new texture's parameters to stop the animation
                    self.display_center = center;
                    self.display_zoom = zoom;
                }
            }
        }

        // Side panel with basic controls and info
        egui::Panel::left("info_panel").show_inside(ui, |ui| {
            ui.heading("Mandelbrot Explorer");
            ui.separator();
            ui.label("Controls:");
            ui.label(" - Zoom: Mouse Wheel");
            ui.label(" - Pan: Click & Drag");
            ui.separator();
            ui.label("Current View:");
            ui.monospace(format!("Center Re: {:.16}", bf_to_f64(&self.target_center.re)));
            ui.monospace(format!("Center Im: {:.16}", bf_to_f64(&self.target_center.im)));
            ui.monospace(format!("Zoom: {:.2e}", self.target_zoom));
            ui.separator();
            ui.label(format!("Max Iterations: {}", max_iter_for_zoom(self.target_zoom)));
            ui.label(format!("Precision: {} bits", precision_for_zoom(self.target_zoom)));
             if ui.button("Reset View").clicked() {
                let initial_zoom = 4.0;
                let initial_center = HpComplex::from_f64(-0.75, 0.0, precision_for_zoom(initial_zoom));
                self.target_center = initial_center.clone();
                self.target_zoom = initial_zoom;
                self.display_center = initial_center;
                self.display_zoom = initial_zoom;
                self.is_calculating = false;
                self.start_calculation();
            }
        });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            let panel_rect = ui.available_rect_before_wrap();

            // Resize Logic
            if self.viewport_size != panel_rect.size() && panel_rect.size().x > 0.0 && panel_rect.size().y > 0.0 {
                self.viewport_size = panel_rect.size();
                self.target_center = self.display_center.clone();
                self.target_zoom = self.display_zoom;
                self.is_calculating = false;
                // A new calculation will be triggered by the check at the end of this function.
            }

            // Animate Display Parameters: display_center += (target_center - display_center) * 0.2.
            // The difference between the two centers is tiny, so it is taken at high precision
            // and then folded back in as a small f64 offset.
            let dxr = hp_diff_f64(&self.target_center.re, &self.display_center.re, prec) * 0.2;
            let dxi = hp_diff_f64(&self.target_center.im, &self.display_center.im, prec) * 0.2;
            self.display_center.re = hp_add_f64(&self.display_center.re, dxr, prec);
            self.display_center.im = hp_add_f64(&self.display_center.im, dxi, prec);
            self.display_zoom = lerp(self.display_zoom, self.target_zoom, 0.2);

            let aspect_ratio = self.viewport_size.x / self.viewport_size.y;
            let aspect_ratio_f64 = aspect_ratio as f64;

            if let Some(texture) = &self.texture {
                // Calculate UV Mapping for Animated Zoom. Offsets of the display center
                // relative to the texture center are small, so they reduce to f64.
                let dcx = hp_diff_f64(&self.display_center.re, &self.texture_center.re, prec);
                let dcy = hp_diff_f64(&self.display_center.im, &self.texture_center.im, prec);
                let u_min = (dcx - 0.5 * self.display_zoom + 0.5 * self.texture_zoom) / self.texture_zoom;
                let u_max = (dcx + 0.5 * self.display_zoom + 0.5 * self.texture_zoom) / self.texture_zoom;
                let tex_h = self.texture_zoom / aspect_ratio_f64;
                let disp_h = self.display_zoom / aspect_ratio_f64;
                let v_min = (dcy - 0.5 * disp_h + 0.5 * tex_h) / tex_h;
                let v_max = (dcy + 0.5 * disp_h + 0.5 * tex_h) / tex_h;

                let uv = Rect::from_min_max(Pos2::new(u_min as f32, v_min as f32), Pos2::new(u_max as f32, v_max as f32));

                let image_widget = egui::Image::new(&*texture).uv(uv).fit_to_exact_size(panel_rect.size());
                let response = ui.add(image_widget);
                let response = ui.interact(response.rect, ui.id().with("mandelbrot_interactive_area"), egui::Sense::click_and_drag());

                if response.dragged() {
                    let delta = response.drag_delta();
                    let complex_delta_re = (delta.x as f64 / self.viewport_size.x as f64) * self.display_zoom;
                    let complex_delta_im = (delta.y as f64 / self.viewport_size.y as f64) * (self.display_zoom / aspect_ratio_f64);
                    self.target_center.re = hp_add_f64(&self.target_center.re, -complex_delta_re, prec);
                    self.target_center.im = hp_add_f64(&self.target_center.im, -complex_delta_im, prec);
                    self.display_center = self.target_center.clone();
                    self.start_calculation();
                }

                if response.hovered() {
                    let scroll = ui.input(|i| i.smooth_scroll_delta);
                    if scroll.y != 0.0 {
                        let zoom_factor = (scroll.y as f64 * 0.01).exp();
                        if let Some(hover_pos) = response.hover_pos() {
                            // Keep the point under the cursor fixed while zooming. The cursor's
                            // complex offset from the center is small (~view size), so the whole
                            // adjustment reduces to adding `offset * (1 - 1/zoom_factor)`.
                            let off_re = ((hover_pos.x - panel_rect.min.x) as f64 / self.viewport_size.x as f64 - 0.5) * self.target_zoom;
                            let off_im = ((hover_pos.y - panel_rect.min.y) as f64 / self.viewport_size.y as f64 - 0.5) * (self.target_zoom / aspect_ratio_f64);
                            let k = 1.0 - 1.0 / zoom_factor;
                            self.target_center.re = hp_add_f64(&self.target_center.re, off_re * k, prec);
                            self.target_center.im = hp_add_f64(&self.target_center.im, off_im * k, prec);
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
            let sdr = hp_diff_f64(&self.target_center.re, &self.texture_center.re, prec);
            let sdi = hp_diff_f64(&self.target_center.im, &self.texture_center.im, prec);
            let target_is_stale = sdr.abs() > self.target_zoom * 1e-6
                || sdi.abs() > self.target_zoom * 1e-6
                || (self.target_zoom - self.texture_zoom).abs() / self.target_zoom > 1e-9;

            if !self.is_calculating && target_is_stale {
                self.start_calculation();
            }

            ctx.request_repaint();
        });
    }
}

/// Renders the Mandelbrot set to a new ColorImage using perturbation theory: a
/// single high-precision reference orbit, then one cheap f64 delta-iteration per
/// pixel. This is what makes both fast recalculation and very deep zooms possible.
fn render_mandelbrot_to_new_image(params: CalculationParams) -> ColorImage {
    let mut image = ColorImage::new(
        params.size,
        vec![Color32::BLACK; params.size[0] * params.size[1]],
    );
    let aspect_ratio = params.size[0] as f64 / params.size[1] as f64;
    let view_height = params.zoom / aspect_ratio;

    let prec = precision_for_zoom(params.zoom);
    let max_iter = max_iter_for_zoom(params.zoom);

    // The one and only high-precision computation: the reference orbit at the
    // view center. Δc for each pixel is then its offset from that center.
    let orbit = compute_reference_orbit(&params.center, max_iter, prec);

    image.pixels.par_iter_mut().enumerate().for_each(|(i, pixel)| {
        let x = i % params.size[0];
        let y = i / params.size[0];

        let dcr = (x as f64 / params.size[0] as f64 - 0.5) * params.zoom;
        let dci = (y as f64 / params.size[1] as f64 - 0.5) * view_height;

        let n = perturbation_iterations(&orbit, dcr, dci, max_iter);
        *pixel = smooth_color(n, max_iter);
    });
    image
}

fn lerp(a: f64, b: f64, t: f64) -> f64 { a + (b - a) * t }

fn smooth_color(n: f64, max_iter: u32) -> Color32 {
    if (n - max_iter as f64).abs() < 1e-6 {
        return Color32::BLACK;
    }
    let hue = (n / 30.0).fract() as f32;
    let saturation = 1.0;
    let value = 0.6 + 0.4 * (n / max_iter as f64) as f32;
    Hsva::new(hue, saturation, value.clamp(0.2, 1.0), 1.0).into()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, rel: f64) -> bool {
        if a == b { return true; }
        (a - b).abs() <= rel * a.abs().max(b.abs()).max(f64::MIN_POSITIVE)
    }

    #[test]
    fn bf_to_f64_roundtrips() {
        for &v in &[0.0, 1.0, -1.0, 0.5, -0.75, 3.14159265358979, 1e-30, -2.5e10, 100.0, 0.1] {
            let bf = BigFloat::from_f64(v, 256);
            let back = bf_to_f64(&bf);
            assert!(approx(v, back, 1e-12), "roundtrip {v} -> {back}");
        }
    }

    #[test]
    fn bf_arithmetic_matches_f64() {
        let p = 256;
        let a = BigFloat::from_f64(1.5, p);
        let b = BigFloat::from_f64(-0.25, p);
        assert!(approx(bf_to_f64(&a.add(&b, p, RM)), 1.25, 1e-12));
        assert!(approx(bf_to_f64(&a.sub(&b, p, RM)), 1.75, 1e-12));
        assert!(approx(bf_to_f64(&a.mul(&b, p, RM)), -0.375, 1e-12));
    }

    #[test]
    fn interior_point_does_not_escape() {
        // (-0.75, 0) is inside the set: should reach max_iter.
        let p = precision_for_zoom(4.0);
        let center = HpComplex::from_f64(-0.75, 0.0, p);
        let orbit = compute_reference_orbit(&center, 1000, p);
        let n = perturbation_iterations(&orbit, 0.0, 0.0, 1000);
        assert_eq!(n, 1000.0, "center should not escape");
    }

    #[test]
    fn exterior_point_escapes_quickly() {
        // A reference inside the set, with a pixel delta landing on c=(1.0, 0.0),
        // which escapes almost immediately.
        let p = precision_for_zoom(4.0);
        let center = HpComplex::from_f64(0.0, 0.0, p);
        let orbit = compute_reference_orbit(&center, 1000, p);
        let n = perturbation_iterations(&orbit, 1.0, 0.0, 1000);
        assert!(n < 10.0, "c=(1,0) should escape fast, got {n}");
    }

    #[test]
    fn perturbation_matches_naive_f64() {
        // Compare the perturbation result against a direct f64 Mandelbrot iteration
        // at a shallow zoom where plain f64 is accurate.
        let p = precision_for_zoom(4.0);
        let cx = -0.745;
        let cy = 0.113;
        let center = HpComplex::from_f64(cx, cy, p);
        let max_iter = 1000;
        let orbit = compute_reference_orbit(&center, max_iter, p);

        for &(dx, dy) in &[(0.0, 0.0), (1e-3, -2e-3), (5e-4, 5e-4), (-1e-3, 1e-3)] {
            let n_pert = perturbation_iterations(&orbit, dx, dy, max_iter);

            // naive
            let (cr, ci) = (cx + dx, cy + dy);
            let (mut zr, mut zi) = (0.0f64, 0.0f64);
            let mut n_naive = max_iter as f64;
            for k in 1..=max_iter {
                let zr2 = zr * zr - zi * zi + cr;
                let zi2 = 2.0 * zr * zi + ci;
                zr = zr2; zi = zi2;
                let nrm = zr * zr + zi * zi;
                if nrm > ESCAPE_RADIUS_SQ {
                    n_naive = k as f64 - nrm.sqrt().log2().log2() + 4.0;
                    break;
                }
            }
            assert!(approx(n_pert, n_naive, 1e-6) || (n_pert - n_naive).abs() < 1.0,
                "delta ({dx},{dy}): pert={n_pert} naive={n_naive}");
        }
    }
}
