---
name: Bau Medical Systems
colors:
  surface: '#0f1417'
  surface-dim: '#0f1417'
  surface-bright: '#353a3d'
  surface-container-lowest: '#0a0f12'
  surface-container-low: '#171c1f'
  surface-container: '#1b2023'
  surface-container-high: '#262b2e'
  surface-container-highest: '#313539'
  on-surface: '#dfe3e7'
  on-surface-variant: '#b9cacb'
  inverse-surface: '#dfe3e7'
  inverse-on-surface: '#2c3134'
  outline: '#849495'
  outline-variant: '#3b494b'
  surface-tint: '#00dbe9'
  primary: '#dbfcff'
  on-primary: '#00363a'
  primary-container: '#00f0ff'
  on-primary-container: '#006970'
  inverse-primary: '#006970'
  secondary: '#b9c7e4'
  on-secondary: '#233148'
  secondary-container: '#3c4962'
  on-secondary-container: '#abb9d6'
  tertiary: '#f4f5ff'
  on-tertiary: '#20304f'
  tertiary-container: '#cad9ff'
  on-tertiary-container: '#4f5e80'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#7df4ff'
  primary-fixed-dim: '#00dbe9'
  on-primary-fixed: '#002022'
  on-primary-fixed-variant: '#004f54'
  secondary-fixed: '#d6e3ff'
  secondary-fixed-dim: '#b9c7e4'
  on-secondary-fixed: '#0d1c32'
  on-secondary-fixed-variant: '#39475f'
  tertiary-fixed: '#d8e2ff'
  tertiary-fixed-dim: '#b6c6ed'
  on-tertiary-fixed: '#091b39'
  on-tertiary-fixed-variant: '#374767'
  background: '#0f1417'
  on-background: '#dfe3e7'
  surface-variant: '#313539'
typography:
  headline-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: '1.2'
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.3'
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.6'
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: '1.5'
  label-caps:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: '1.0'
    letterSpacing: 0.08em
  data-readout:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '400'
    lineHeight: '1.2'
  headline-lg-mobile:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.2'
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  unit: 4px
  gutter: 16px
  margin-desktop: 24px
  margin-mobile: 16px
  panel-padding: 12px
---

## Brand & Style

The design system is engineered for precision, clarity, and clinical reliability. It targets radiologists, surgeons, and medical technicians who require high-density information without cognitive fatigue. The brand personality is "Technological Authority"—it feels like a high-end medical instrument: silent, powerful, and impeccably accurate.

The design style utilizes **Clinical Minimalism** blended with **X-Ray Depth**. It leverages subtle transparency, light-refraction effects, and layered surfaces to mimic the experience of looking through anatomical volumes. The UI should evoke a sense of digital sterility and cutting-edge diagnostic power.

## Colors

The palette is optimized for long-duration viewing in dark radiology reading rooms.

- **Primary (Electric Cyan):** Used exclusively for interactive states, 3D mesh highlights, and active diagnostic tools. It represents the "energy" of the scan.
- **Secondary (Deep Clinical Blue):** The foundation of the workspace. It provides a high-contrast backdrop for grayscale medical images (X-rays/CTs).
- **Tertiary (Surface Blue):** Used for paneling and container backgrounds to differentiate the workspace from the utility chrome.
- **Neutral (Sterile White/Gray):** Used for critical data readouts and labels to ensure maximum legibility against dark backgrounds.

Color application should be sparse. High-saturation colors are reserved for anomalies, alerts, or active selection states to maintain focus on the medical imagery.

## Typography

This design system utilizes **Inter** for its neutral, systematic clarity and excellent legibility at small sizes. For technical data, coordinates, and metadata (e.g., DICOM tags), **JetBrains Mono** is used to provide a "technical instrument" feel and ensure character distinction (e.g., 0 vs O).

All headlines should use tighter letter spacing to maintain a compact, professional look. Labels for metadata should always be rendered in the `label-caps` or `data-readout` styles to distinguish system information from user-generated content.

## Layout & Spacing

The layout follows a **Fixed-Sidebar Fluid-Canvas** model. The central viewing area (the "Lightbox") maximizes available screen real estate for 3D volumes. 

- **The Grid:** A 12-column system is used for dashboard views, but the primary workspace relies on a "Modular Docking" system where panels can be collapsed to favor the imaging area.
- **Rhythm:** A strict 4px soft-grid ensures alignment. All components (buttons, inputs, icons) must align to this 4px baseline.
- **Margins:** Desktop views use 24px outer margins. On mobile, this reduces to 16px. Gutters between utility panels are kept tight (16px) to maximize data density.

## Elevation & Depth

Depth in this design system is achieved through **Tonal Layering** and **Luminance**, rather than traditional shadows.

1.  **Level 0 (Base):** Deepest clinical blue (`#0A192F`). The "dark room."
2.  **Level 1 (Panels):** Tertiary blue (`#112240`) with a 1px inner stroke of 10% white to define edges.
3.  **Level 2 (Overlays/Modals):** Glassmorphism effect. Use a backdrop blur (20px) with a semi-transparent background (60% opacity) and a crisp 1px primary-color border at 30% opacity.

This creates the "X-ray layer" effect, where floating menus feel like they are hovering over the anatomical data without obscuring it completely.

## Shapes

The shape language is **Soft-Geometric**. A subtle 4px radius (`roundedness: 1`) is applied to buttons, inputs, and containers. This provides a modern, professional feel that is less aggressive than sharp corners but more clinical than highly rounded "consumer" apps. 

Large image viewports and 3D canvases should remain sharp (0px) to maximize the perception of technical precision and to utilize every pixel of the sensor data.

## Components

- **Buttons:** Primary buttons use a solid Electric Cyan fill with dark text. Secondary buttons are "Ghost" style: 1px border with cyan text and no fill until hover.
- **Inputs:** Dark backgrounds with a 1px border. On focus, the border glows with the primary cyan color and a subtle outer neon spread (2px).
- **Status Chips:** Small, rectangular indicators. Use "Positive" (Green), "Warning" (Amber), and "Critical" (Red) only for diagnostic findings; system status should remain Cyan.
- **The Lightbox:** The primary component for 3D viewing. It should feature a 1px Cyan crosshair and unobtrusive corner text overlays for orientation (L/R, S/I, A/P).
- **Measurement Tools:** Rulers and calipers should use a 0.5pt stroke weight in primary cyan to ensure they do not obscure the underlying pathology.