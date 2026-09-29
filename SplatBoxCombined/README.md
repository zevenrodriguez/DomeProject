# SplatBox Combined Plugin (Render + Relight)

All-in-one precompiled package: SplatBox Gaussian Splat editing tools **and** the Relight lighting add-on.

**Build Info:**
- Unity Version: 6000.6.0f1
- URP Support: NO - GaussianSplatURPFeature NOT included (rebuild with URP for full support)
- Build Date: 2026-09-28 20:30

## Installation

### Option 1: Unity Package Manager
1. Open Unity Package Manager (Window > Package Manager)
2. Click '+' and select 'Add package from disk...'
3. Navigate to this folder and select `package.json`

### Option 2: Manual
1. Copy this folder into your project's `Packages/` directory.
2. Rename to `com.gaussiansplatting.splatbox.combined`.

## Requirements

- **Unity 6** (6000.0+) — compiled with Unity 6000.6.0f1
- Universal Render Pipeline (URP) for splat rendering
- Auto-installed dependencies: Mathematics, Collections, Burst

**IMPORTANT:** Use with the same major Unity version it was built with.

## URP Setup

The **Gaussian Splat URP Feature** is automatically added to all URP renderers when the editor loads.
If needed, you can re-run it manually via **SplatBox > Setup URP Renderer Features**.

(Optional) Add **Gaussian Tilt Shift Renderer Feature** manually to your URP Renderer Asset for tilt-shift effects.

## PC VR (Stereo Rendering)

PCVR stereo rendering is supported via **Multi-Pass** mode:

1. In **Project Settings > XR Plug-in Management**, enable your headset plug-in (OpenXR, Oculus, etc.)
2. In your **URP Renderer Asset**, set **Rendering Path** to **Multi-Pass**
3. Splats will render correctly to both eyes automatically

> **Note:** Single Pass Instanced is not currently supported for Gaussian Splats. Use Multi-Pass.

## Android / Quest 3 Standalone

The GPU sort (`BoxUtilSort.compute` / `DeviceRadixSort.hlsl`) uses HLSL wave
intrinsics that require Vulkan. OpenGL ES does not support them, so you MUST
build with Vulkan:

1. **Edit > Project Settings > Player > Android > Other Settings**
2. Uncheck **Auto Graphics API**
3. Make **Vulkan** the first (and ideally only) Graphics API — remove OpenGLES3
4. Set **Minimum API Level** to **Android 10 (API 29)** or higher
5. Set **Scripting Backend** to **IL2CPP** and **Target Architectures** to **ARM64**
6. Quest 3 / Quest Pro supports Vulkan subgroup ops natively — no extra setup

If you leave OpenGLES3 in the Graphics API list, the build will fail while
compiling the sort compute shader because wave intrinsics cannot be translated
to GLSL ES.

## Contents

- **Plugins/** — Precompiled DLLs
  - `GaussianSplatting.SplatBox.Core.dll` — Core rendering and data structures
  - `GaussianSplatting.SplatBox.Runtime.dll` — Runtime components, brushes, URP features
  - `GaussianSplatting.SplatBox.Relight.dll` — Relight runtime (custom lights)
  - **Editor/**
    - `GaussianSplatting.SplatBox.Editor.dll` — Editor tools and inspectors
    - `GaussianSplatting.SplatBox.Relight.Editor.dll` — Relight editor (Relight window, custom inspectors)
- **Editor/Icons/** — Editor icon assets
- **Shaders/** — All shader and compute files (SplatBox + Relight)
- **Resources/** — Runtime-loadable assets

## Relight Usage

- Menu: **SplatToolbox > Relight > Relight Editor**
- In Splat Editor, enable the Relight toggle and 'Use Relight shader'.
- Add **Custom Light Manager** to the scene, then add Point, Spot, Directional, or Planar lights as children.
- Adjust intensity, range, attenuation, etc. in each light's inspector.
