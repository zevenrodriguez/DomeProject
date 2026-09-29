Shader "Gaussian Splatting/Render Splats"
{
    Properties
    {
        _ColorExposure ("Exposure", Float) = 0.0
        _ColorContrast ("Contrast", Float) = 1.0
        _ColorFilter ("Color Filter", Color) = (1,1,1,1)
        _ColorHueShift ("Hue Shift", Float) = 0.0
        _ColorSaturation ("Saturation", Float) = 1.0
        _SelectionColor ("Selection Color", Color) = (1,0,1,1)
        _TargetSelectionColor ("Target Selection Color", Color) = (0,1,1,1)
    }
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }

        Pass
        {
            ZWrite Off
            Blend OneMinusDstAlpha One
            Cull Off
            
CGPROGRAM
#pragma vertex vert
#pragma fragment frag
// No #pragma use_dxc: nothing here needs SM6, and DXC's auto-binding collides on Vulkan
// ("resource '_OrderBuffer' shares binding N with another resource"), which leaves the splat
// buffers unbound and the draw skipped.
#pragma require compute
#pragma multi_compile _ STEREO_MULTIVIEW_ON

#include "UnityCG.cginc"
#include "SplatBox.hlsl"
#include "EquirectangularProjection.hlsl"
#include "MeshSplatTexture.hlsl"

StructuredBuffer<uint> _OrderBuffer;
uint _SplatCount; // valid splat count; guards against out-of-range sort indices
// Scene-depth occlusion (see RenderSplatsRelight.shader): the splat RT has no depth attachment, so opaque
// geometry is hidden by comparing against the camera depth texture in the fragment instead.
UNITY_DECLARE_DEPTH_TEXTURE(_CameraDepthTexture);
uint _SceneDepthOcclusion;  // 1 = fade out splats that lie behind opaque scene geometry
float _SceneDepthFade;      // metres over which a splat fades as it passes behind geometry
uint _SceneDepthTextureValid;
float4 _SplatRenderTargetSize; // xy = splat RT size, zw = reciprocal size

// Color adjustments
float _ColorExposure;
float _ColorContrast;
float4 _ColorFilter;
float _ColorHueShift;
float _ColorSaturation;

// Selection colors
float4 _SelectionColor;
float4 _TargetSelectionColor;

struct v2f
{
    half4 col : COLOR0;
    float2 pos : TEXCOORD0;
    float eyeDepth : TEXCOORD1;
    float3 projectionPositionWS : TEXCOORD2;
    float2 meshUv : TEXCOORD3;
    float4 vertex : SV_POSITION;
    UNITY_VERTEX_OUTPUT_STEREO
};

StructuredBuffer<SplatViewData> _SplatViewData;
ByteAddressBuffer _SplatSelectedBits;
ByteAddressBuffer _SplatTargetSelectedBits;
uint _SplatBitsValid;

// Convert RGB to HSV
float3 RGBToHSV(float3 c)
{
    float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    float4 p = lerp(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
    float4 q = lerp(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));
    
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

// Convert HSV to RGB
float3 HSVToRGB(float3 c)
{
    float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * lerp(K.xxx, saturate(p - K.xxx), c.y);
}

// Apply color adjustments to a color
float3 ApplyColorAdjustments(float3 color)
{
    // Apply exposure (brightness) - 0 is neutral
    if (abs(_ColorExposure) > 0.001)
        color *= exp2(_ColorExposure);
    
    // Apply contrast - 1.0 is neutral
    if (abs(_ColorContrast - 1.0) > 0.001)
        color = lerp(0.5, color, _ColorContrast);
    
    // Apply hue shift and saturation - only if needed
    if (abs(_ColorHueShift) > 0.001 || abs(_ColorSaturation - 1.0) > 0.001)
    {
        float3 hsv = RGBToHSV(color);
        hsv.x = frac(hsv.x + _ColorHueShift / 360.0);
        hsv.y *= _ColorSaturation;
        color = HSVToRGB(hsv);
    }
    
    // Apply color filter only if not white (1,1,1)
    bool filterIsNeutral = _ColorFilter.r > 0.99 && _ColorFilter.g > 0.99 && _ColorFilter.b > 0.99;
    if (!filterIsNeutral)
        color *= _ColorFilter.rgb;
    
    return color;
}

v2f vert (uint vtxID : SV_VertexID, uint instID : SV_InstanceID)
{
    v2f o = (v2f)0;
    UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
    instID = _OrderBuffer[instID];
    // Discard out-of-range / garbage payload indices the GPU radix sort can emit on some mobile
    // (Adreno/Vulkan) GPUs, which otherwise read garbage _SplatViewData and draw noise splats.
    if (instID >= _SplatCount)
    {
        o.vertex = asfloat(0x7fc00000); // NaN discards the primitive
        return o;
    }
    SplatViewData view = _SplatViewData[instID];
    float4 centerClipPos = view.pos;
    bool behindCam = centerClipPos.w <= 0;
    if (behindCam)
    {
        o.vertex = asfloat(0x7fc00000); // NaN discards the primitive
    }
    else
    {
        o.col.r = f16tof32(view.color.x >> 16);
        o.col.g = f16tof32(view.color.x);
        o.col.b = f16tof32(view.color.y >> 16);
        o.col.a = f16tof32(view.color.y);

        uint idx = vtxID;
        float2 quadPos = float2(idx&1, (idx>>1)&1) * 2.0 - 1.0;
        quadPos *= 2;

        o.pos = quadPos;
        o.meshUv = ComputeSplatMeshUv(instID, quadPos);

        float2 axis1 = float2(f16tof32(view.axis1 >> 16), f16tof32(view.axis1));
        float2 axis2 = float2(f16tof32(view.axis2 >> 16), f16tof32(view.axis2));
        // _ScreenParams describes the camera target, which can be larger than the
        // low-resolution splat target. Keep quad expansion aligned with CalcViewData.
        float2 splatTargetSize = _SplatRenderTargetSize.x > 0
            ? _SplatRenderTargetSize.xy
            : _ScreenParams.xy;
        float2 deltaScreenPos = (quadPos.x * axis1 + quadPos.y * axis2) * 2 / splatTargetSize;
        o.vertex = centerClipPos;
        o.vertex.xy += deltaScreenPos * centerClipPos.w;

        float2 viewOffset = float2(
            deltaScreenPos.x * centerClipPos.w / UNITY_MATRIX_P._m00,
            deltaScreenPos.y * centerClipPos.w / UNITY_MATRIX_P._m11);
        o.projectionPositionWS = view.worldPos.xyz +
            mul((float3x3)UNITY_MATRIX_I_V, float3(viewOffset, 0.0));

        // Perspective clip.w is eye depth; orthographic projections need view-space z.
        float eyeDepth = unity_OrthoParams.w > 0.5
            ? -mul(UNITY_MATRIX_V, float4(view.worldPos.xyz, 1.0)).z
            : centerClipPos.w;
        o.eyeDepth = eyeDepth;

        // Check selection state
        if (_SplatBitsValid)
        {
            uint wordIdx = instID / 32;
            uint bitIdx = instID & 31;
            uint selVal = _SplatSelectedBits.Load(wordIdx * 4);
            uint targetSelVal = _SplatTargetSelectedBits.Load(wordIdx * 4);
            bool isSourceSelected = (selVal & (1 << bitIdx)) != 0;
            bool isTargetSelected = (targetSelVal & (1 << bitIdx)) != 0;
            
            // Encode selection state in alpha:
            // -1 = source selected only
            // -2 = target selected only
            // -3 = both selected
            if (isSourceSelected && isTargetSelected)
                o.col.a = -3;
            else if (isTargetSelected)
                o.col.a = -2;
            else if (isSourceSelected)
                o.col.a = -1;
        }
    }
    return o;
}

half4 frag (v2f i) : SV_Target
{
    UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
    float power = -dot(i.pos, i.pos);
    half alpha = exp(power);
    
    if (i.col.a >= 0)
    {
        alpha = saturate(alpha * i.col.a);
    }
    else
    {
        // Selection state encoded in negative alpha:
        // -1 = source selected, -2 = target selected, -3 = both
        half3 selColor = _SelectionColor.rgb;
        if (i.col.a < -2.5) // Both selected (-3)
        {
            selColor = lerp(_SelectionColor.rgb, _TargetSelectionColor.rgb, 0.5);
        }
        else if (i.col.a < -1.5) // Target selected (-2)
        {
            selColor = _TargetSelectionColor.rgb;
        }
        
        if (alpha > 7.0/255.0)
        {
            if (alpha < 10.0/255.0)
            {
                alpha = 1;
                i.col.rgb = selColor;
            }
            alpha = saturate(alpha + 0.3);
        }
        i.col.rgb = lerp(i.col.rgb, selColor, 0.5);
    }

    // Scene-depth occlusion: fade the splat out where opaque geometry is nearer than its centre.
    if (_SceneDepthOcclusion != 0 && _SceneDepthTextureValid != 0)
    {
        // SV_POSITION is a pixel coordinate in the actual splat RT. Normalizing that
        // coordinate avoids API-dependent clip-space Y orientation.
        float2 depthUv = i.vertex.xy * _SplatRenderTargetSize.zw;
        float rawDepth = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, depthUv);
        float sceneEyeDepth;
        if (unity_OrthoParams.w > 0.5)
        {
            #if UNITY_REVERSED_Z
            rawDepth = 1.0 - rawDepth;
            #endif
            sceneEyeDepth = lerp(_ProjectionParams.y, _ProjectionParams.z, rawDepth);
        }
        else
        {
            sceneEyeDepth = LinearEyeDepth(rawDepth);
        }
        // Keep the splat fully visible when it is on or in front of the opaque surface,
        // then soften only the portion that moves behind it. The previous fade window was
        // reversed: equal depth produced zero alpha, which cut a hard primitive-shaped hole.
        float depthGap = sceneEyeDepth - i.eyeDepth;
        alpha *= saturate(1.0 + depthGap / max(_SceneDepthFade, 1e-4));
    }
    
    if (alpha < 1.0/255.0)
        discard;

    // Apply color adjustments
    half3 finalColor = ApplyColorAdjustments(ApplyMeshSplatAlbedo(i.col.rgb, i.meshUv));
    
    half4 res = half4(finalColor * alpha, alpha);
    res = ApplyEquirectangularProjection(i.projectionPositionWS, res);
    return res;
}
ENDCG
        }

        Pass
        {
            Name "BlitToScreen"
            ZWrite Off
            ZTest Always
            Cull Off
            Blend SrcAlpha OneMinusSrcAlpha

CGPROGRAM
#pragma vertex vertComposite
#pragma fragment fragComposite
// No #pragma use_dxc: nothing here needs SM6, and DXC's auto-binding collides on Vulkan
// ("resource '_OrderBuffer' shares binding N with another resource"), which leaves the splat
// buffers unbound and the draw skipped.
#pragma require compute
#pragma multi_compile _ STEREO_MULTIVIEW_ON
#include "UnityCG.cginc"

struct v2fComposite
{
    float4 vertex : SV_POSITION;
    UNITY_VERTEX_OUTPUT_STEREO
};

v2fComposite vertComposite(uint vtxID : SV_VertexID)
{
    v2fComposite o;
    UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
    float2 quadPos = float2(vtxID&1, (vtxID>>1)&1) * 4.0 - 1.0;
    o.vertex = float4(quadPos, 1, 1);
    return o;
}

Texture2D _GaussianSplatRT;
SamplerState sampler_GaussianSplatRT;

half4 fragComposite(v2fComposite i) : SV_Target
{
    UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
    // Sample (not Load) so the splat RT can be rendered below screen resolution
    // (SplatBoxRenderManager.renderScale) and bilinearly upscaled here.
    float2 uv = i.vertex.xy / _ScreenParams.xy;
    half4 col = _GaussianSplatRT.Sample(sampler_GaussianSplatRT, uv);
    return col;
}
ENDCG
        }
    }
}
