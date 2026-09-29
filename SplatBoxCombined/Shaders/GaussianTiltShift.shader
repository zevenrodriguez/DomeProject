// SPDX-License-Identifier: MIT
// Tilt-Shift Miniature Effect + Depth of Field for Gaussian Splats
Shader "Hidden/Gaussian Splatting/TiltShift"
{
    Properties
    {
        _MainTex ("Source", 2D) = "white" {}
        [HideInInspector] _BlitTexture ("Blit Texture", 2D) = "white" {}
        _FocusPosition ("Focus Position", Range(0, 1)) = 0.5
        _FocusWidth ("Focus Width", Range(0, 1)) = 0.2
        _BlurStrength ("Blur Strength", Range(0, 10)) = 3.0
        _TopBlurStrength ("Top Blur Strength", Range(0, 2)) = 1.0
        _BottomBlurStrength ("Bottom Blur Strength", Range(0, 2)) = 1.0
        _TopBlurOffset ("Top Blur Offset", Range(-0.5, 0.5)) = 0.0
        _BottomBlurOffset ("Bottom Blur Offset", Range(-0.5, 0.5)) = 0.0
        _BlurSamples ("Blur Samples", Range(3, 16)) = 8
        _TiltAngle ("Tilt Angle", Range(-45, 45)) = 0
        _SaturationBoost ("Saturation Boost", Range(0, 2)) = 1.3
        _ContrastBoost ("Contrast Boost", Range(0.5, 2)) = 1.15
        _VignetteStrength ("Vignette Strength", Range(0, 1)) = 0.3
        _BokehShape ("Bokeh Shape", Range(0, 1)) = 0.5
        _BlurMode ("Blur Mode", Float) = 0
        _FocusDistance ("Focus Distance", Float) = 10.0
        _FocusRange ("Focus Range", Float) = 5.0
        _NearBlurStrength ("Near Blur Strength", Range(0, 1)) = 0.5
        _FarBlurStrength ("Far Blur Strength", Range(0, 1)) = 0.8
    }

    HLSLINCLUDE
    #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
    #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"
    #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"

    // _BlitTexture_TexelSize: declared by Blit.hlsl in Unity 6+ but missing in older URP.
    // Guard with UNITY_VERSION so it compiles on both.
    #if UNITY_VERSION < 600000
    float4 _BlitTexture_TexelSize;
    #endif

    float _FocusPosition;
    float _FocusWidth;
    float _BlurStrength;
    float _TopBlurStrength;
    float _BottomBlurStrength;
    float _TopBlurOffset;
    float _BottomBlurOffset;
    float _BlurSamples;
    float _TiltAngle;
    float _SaturationBoost;
    float _ContrastBoost;
    float _VignetteStrength;
    float _BokehShape;
    float _BlurMode;
    float _FocusDistance;
    float _FocusRange;
    float _NearBlurStrength;
    float _FarBlurStrength;

    // _BlurStrength is authored in virtual 720p pixels so the same setting has the same
    // screen-space/angular size in the Editor and on a high-resolution XR eye texture.
    // Quest 3 currently renders this project at 1680x1760 per eye; treating the value as
    // literal pixels made the headset blur about 2.4x weaker than the Game view.
    float GetBlurResolutionScale(float4 texelSize)
    {
        return max(texelSize.w, 1.0) / 720.0;
    }

    float GetTiltShiftBlurFactor(float2 uv)
    {
        float angleRad = _TiltAngle * 0.0174533;
        float2 center = float2(0.5, 0.5);
        float2 rotatedUV = uv - center;
        float cs = cos(angleRad);
        float sn = sin(angleRad);
        float rotatedY = rotatedUV.x * sn + rotatedUV.y * cs + 0.5;
        
        // Per-side sharp half-extent. Offsets push each blur edge away from (positive) or toward
        // (negative) the focus center, so the top and bottom blur boundaries can be aligned
        // independently — needed in VR where the comfortable focus band is not symmetric.
        float focusHalf = _FocusWidth * 0.5;
        float signedDist = rotatedY - _FocusPosition; // >0 above focus center, <0 below
        float topHalf = max(focusHalf + _TopBlurOffset, 0.0);
        float bottomHalf = max(focusHalf + _BottomBlurOffset, 0.0);
        float edge = (signedDist >= 0.0) ? topHalf : bottomHalf;
        float blur = smoothstep(0, focusHalf + 0.3, abs(signedDist) - edge);
        float sideStrength = lerp(_BottomBlurStrength, _TopBlurStrength, step(_FocusPosition, rotatedY));
        return blur * blur * sideStrength;
    }

    float GetDepthBlurFactor(float2 uv)
    {
        // Sample depth using URP's proper depth sampling
        float depth = SampleSceneDepth(uv);
        float linearDepth = LinearEyeDepth(depth, _ZBufferParams);
        
        float focusHalf = _FocusRange * 0.5;
        float nearEdge = max(_FocusDistance - focusHalf, 0.01);
        float farEdge = _FocusDistance + focusHalf;
        
        float blurFactor = 0;
        if (linearDepth < nearEdge)
        {
            // Near blur - objects closer than focus
            float nearBlend = saturate((nearEdge - linearDepth) / nearEdge);
            blurFactor = nearBlend * _NearBlurStrength;
        }
        else if (linearDepth > farEdge)
        {
            // Far blur - objects farther than focus
            float farBlend = saturate((linearDepth - farEdge) / (_FocusDistance * 2.0));
            blurFactor = farBlend * _FarBlurStrength;
        }
        
        return saturate(blurFactor);
    }

    float GetBlurFactor(float2 uv)
    {
        // Mode 0 = TiltShift, Mode 1 = DoF, Mode 2 = Both
        if (_BlurMode < 0.5)
            return GetTiltShiftBlurFactor(uv);
        else if (_BlurMode < 1.5)
            return GetDepthBlurFactor(uv);
        else
            return max(GetTiltShiftBlurFactor(uv), GetDepthBlurFactor(uv));
    }

    float3 RGBToHSV(float3 c)
    {
        float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
        float4 p = lerp(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
        float4 q = lerp(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));
        float d = q.x - min(q.w, q.y);
        float e = 1.0e-10;
        return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
    }

    float3 HSVToRGB(float3 c)
    {
        float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
        float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
        return c.z * lerp(K.xxx, saturate(p - K.xxx), c.y);
    }
    ENDHLSL

    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline" = "UniversalPipeline" }
        LOD 100
        ZWrite Off ZTest Always Cull Off

        // Pass 0: Horizontal blur (fixed 9-tap Gaussian)
        Pass
        {
            Name "TiltShift_BlurH"
            
            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment FragBlurH

            // Fixed Gaussian weights for 9-tap blur
            static const float kWeights[5] = { 0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216 };

            half4 FragBlurH(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                float2 uv = input.texcoord;
                float4 texelSize = _BlitTexture_TexelSize;
                
                float blurFactor = GetBlurFactor(uv);
                float blurRadius = blurFactor * _BlurStrength *
                    GetBlurResolutionScale(texelSize) * texelSize.x;
                
                if (blurRadius < 0.0001)
                    return SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv);
                
                // Center sample
                half4 color = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv) * kWeights[0];
                
                // Symmetric samples
                [unroll]
                for (int i = 1; i < 5; i++)
                {
                    float offset = (float)i * blurRadius * 0.25;
                    color += SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv + float2(offset, 0)) * kWeights[i];
                    color += SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv - float2(offset, 0)) * kWeights[i];
                }
                
                return color;
            }
            ENDHLSL
        }

        // Pass 1: Vertical blur (fixed 9-tap Gaussian)
        Pass
        {
            Name "TiltShift_BlurV"
            
            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment FragBlurV

            static const float kWeightsV[5] = { 0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216 };

            half4 FragBlurV(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                float2 uv = input.texcoord;
                float4 texelSize = _BlitTexture_TexelSize;
                
                float blurFactor = GetBlurFactor(uv);
                float blurRadius = blurFactor * _BlurStrength *
                    GetBlurResolutionScale(texelSize) * texelSize.y;
                
                if (blurRadius < 0.0001)
                    return SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv);
                
                // Center sample
                half4 color = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv) * kWeightsV[0];
                
                // Symmetric samples
                [unroll]
                for (int i = 1; i < 5; i++)
                {
                    float offset = (float)i * blurRadius * 0.25;
                    color += SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv + float2(0, offset)) * kWeightsV[i];
                    color += SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv - float2(0, offset)) * kWeightsV[i];
                }
                
                return color;
            }
            ENDHLSL
        }

        // Pass 2: Color enhancement
        Pass
        {
            Name "TiltShift_ColorEnhance"
            
            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment FragEnhance

            half4 FragEnhance(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                float2 uv = input.texcoord;
                half4 color = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv);
                
                // Saturation boost
                float3 hsv = RGBToHSV(color.rgb);
                hsv.y = saturate(hsv.y * _SaturationBoost);
                color.rgb = HSVToRGB(hsv);
                
                // Contrast
                color.rgb = saturate((color.rgb - 0.5) * _ContrastBoost + 0.5);
                
                // Vignette
                float2 vignetteUV = uv - 0.5;
                float vignette = 1.0 - dot(vignetteUV, vignetteUV) * _VignetteStrength * 2.0;
                color.rgb *= saturate(vignette);
                
                return color;
            }
            ENDHLSL
        }

        // Pass 3: Combined single-pass
        Pass
        {
            Name "TiltShift_Combined"
            
            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment FragCombined

            // Fixed 16-point circular blur pattern
            static const float2 kBlurOffsets[16] = {
                float2(1.0, 0.0), float2(0.9239, 0.3827), float2(0.7071, 0.7071), float2(0.3827, 0.9239),
                float2(0.0, 1.0), float2(-0.3827, 0.9239), float2(-0.7071, 0.7071), float2(-0.9239, 0.3827),
                float2(-1.0, 0.0), float2(-0.9239, -0.3827), float2(-0.7071, -0.7071), float2(-0.3827, -0.9239),
                float2(0.0, -1.0), float2(0.3827, -0.9239), float2(0.7071, -0.7071), float2(0.9239, -0.3827)
            };

            half4 FragCombined(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                float2 uv = input.texcoord;
                float4 texelSize = _BlitTexture_TexelSize;
                
                float blurFactor = GetBlurFactor(uv);
                float2 blurRadius = blurFactor * _BlurStrength *
                    GetBlurResolutionScale(texelSize) * texelSize.xy;
                
                half4 color;
                
                if (length(blurRadius) < 0.0001)
                {
                    color = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv);
                }
                else
                {
                    // Center sample
                    color = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv);
                    float totalWeight = 1.0;
                    
                    // Inner ring (8 samples at 50% radius)
                    float innerRadius = 0.5;
                    float innerWeight = 1.0 - innerRadius * _BokehShape;
                    innerWeight = max(innerWeight, 0.2);
                    
                    [unroll]
                    for (int j = 0; j < 8; j++)
                    {
                        float2 offset = kBlurOffsets[j * 2] * innerRadius * blurRadius;
                        half4 s = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv + offset);
                        float b = dot(s.rgb, float3(0.299, 0.587, 0.114));
                        float w = innerWeight * lerp(1.0, 1.5, saturate(b - 0.5) * 2);
                        color += s * w;
                        totalWeight += w;
                    }
                    
                    // Outer ring (16 samples at 100% radius)
                    float outerWeight = 1.0 - _BokehShape;
                    outerWeight = max(outerWeight, 0.2);
                    
                    [unroll]
                    for (int k = 0; k < 16; k++)
                    {
                        float2 offset = kBlurOffsets[k] * blurRadius;
                        half4 s = SAMPLE_TEXTURE2D_X(_BlitTexture, sampler_LinearClamp, uv + offset);
                        float b = dot(s.rgb, float3(0.299, 0.587, 0.114));
                        float w = outerWeight * lerp(1.0, 1.5, saturate(b - 0.5) * 2);
                        color += s * w;
                        totalWeight += w;
                    }
                    
                    color /= totalWeight;
                }
                
                // Saturation
                float3 hsv = RGBToHSV(color.rgb);
                hsv.y = saturate(hsv.y * _SaturationBoost);
                color.rgb = HSVToRGB(hsv);
                
                // Contrast
                color.rgb = saturate((color.rgb - 0.5) * _ContrastBoost + 0.5);
                
                // Vignette
                float2 vignetteUV = uv - 0.5;
                float vignette = 1.0 - dot(vignetteUV, vignetteUV) * _VignetteStrength * 2.0;
                color.rgb *= saturate(vignette);
                
                return color;
            }
            ENDHLSL
        }
    }

    // Fallback for Built-in pipeline and editor
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100
        ZWrite Off ZTest Always Cull Off

        Pass
        {
            Name "TiltShift_Builtin"
            
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            float4 _MainTex_TexelSize;
            sampler2D _CameraDepthTexture;

            float _FocusPosition;
            float _FocusWidth;
            float _BlurStrength;
            float _TopBlurStrength;
            float _BottomBlurStrength;
            float _TopBlurOffset;
            float _BottomBlurOffset;
            float _BlurSamples;
            float _TiltAngle;
            float _SaturationBoost;
            float _ContrastBoost;
            float _VignetteStrength;
            float _BokehShape;
            float _BlurMode;
            float _FocusDistance;
            float _FocusRange;
            float _NearBlurStrength;
            float _FarBlurStrength;

            float GetBlurResolutionScale(float4 texelSize)
            {
                return max(texelSize.w, 1.0) / 720.0;
            }

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            float GetTiltShiftBlurFactor(float2 uv)
            {
                float angleRad = _TiltAngle * 0.0174533;
                float2 center = float2(0.5, 0.5);
                float2 rotatedUV = uv - center;
                float cs = cos(angleRad);
                float sn = sin(angleRad);
                float rotatedY = rotatedUV.x * sn + rotatedUV.y * cs + 0.5;
                
                // Per-side sharp half-extent so the top and bottom blur edges can be nudged
                // independently to align the focus band in VR.
                float focusHalf = _FocusWidth * 0.5;
                float signedDist = rotatedY - _FocusPosition; // >0 above focus center, <0 below
                float topHalf = max(focusHalf + _TopBlurOffset, 0.0);
                float bottomHalf = max(focusHalf + _BottomBlurOffset, 0.0);
                float edge = (signedDist >= 0.0) ? topHalf : bottomHalf;
                float blur = smoothstep(0, focusHalf + 0.3, abs(signedDist) - edge);
                float sideStrength = lerp(_BottomBlurStrength, _TopBlurStrength, step(_FocusPosition, rotatedY));
                return blur * blur * sideStrength;
            }

            float GetDepthBlurFactor(float2 uv)
            {
                float depth = tex2D(_CameraDepthTexture, uv).r;
                float linearDepth = LinearEyeDepth(depth);
                
                float focusHalf = _FocusRange * 0.5;
                float nearEdge = _FocusDistance - focusHalf;
                float farEdge = _FocusDistance + focusHalf;
                
                float blurFactor = 0;
                if (linearDepth < nearEdge)
                {
                    blurFactor = saturate((nearEdge - linearDepth) / max(nearEdge, 0.001)) * _NearBlurStrength;
                }
                else if (linearDepth > farEdge)
                {
                    blurFactor = saturate((linearDepth - farEdge) / max(_FocusDistance, 1.0)) * _FarBlurStrength;
                }
                
                return blurFactor;
            }

            float GetBlurFactor(float2 uv)
            {
                // Mode 0 = TiltShift, Mode 1 = DoF, Mode 2 = Both
                if (_BlurMode < 0.5)
                    return GetTiltShiftBlurFactor(uv);
                else if (_BlurMode < 1.5)
                    return GetDepthBlurFactor(uv);
                else
                    return max(GetTiltShiftBlurFactor(uv), GetDepthBlurFactor(uv));
            }

            float3 RGBToHSV(float3 c)
            {
                float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
                float4 p = lerp(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
                float4 q = lerp(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));
                float d = q.x - min(q.w, q.y);
                float e = 1.0e-10;
                return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
            }

            float3 HSVToRGB(float3 c)
            {
                float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
                return c.z * lerp(K.xxx, saturate(p - K.xxx), c.y);
            }

            // Fixed 16-point circular blur pattern (avoids dynamic loop issues)
            static const float2 blurOffsets[16] = {
                float2(1.0, 0.0), float2(0.9239, 0.3827), float2(0.7071, 0.7071), float2(0.3827, 0.9239),
                float2(0.0, 1.0), float2(-0.3827, 0.9239), float2(-0.7071, 0.7071), float2(-0.9239, 0.3827),
                float2(-1.0, 0.0), float2(-0.9239, -0.3827), float2(-0.7071, -0.7071), float2(-0.3827, -0.9239),
                float2(0.0, -1.0), float2(0.3827, -0.9239), float2(0.7071, -0.7071), float2(0.9239, -0.3827)
            };

            half4 frag(v2f i) : SV_Target
            {
                float blurFactor = GetBlurFactor(i.uv);
                float2 blurRadius = blurFactor * _BlurStrength *
                    GetBlurResolutionScale(_MainTex_TexelSize) * _MainTex_TexelSize.xy;
                
                half4 color;
                
                if (length(blurRadius) < 0.0001)
                {
                    color = tex2D(_MainTex, i.uv);
                }
                else
                {
                    // Center sample
                    color = tex2D(_MainTex, i.uv);
                    float totalWeight = 1.0;
                    
                    // Inner ring (8 samples at 50% radius)
                    float innerRadius = 0.5;
                    float innerWeight = 1.0 - innerRadius * _BokehShape;
                    innerWeight = max(innerWeight, 0.2);
                    
                    [unroll]
                    for (int j = 0; j < 8; j++)
                    {
                        float2 offset = blurOffsets[j * 2] * innerRadius * blurRadius;
                        half4 s = tex2D(_MainTex, i.uv + offset);
                        float b = dot(s.rgb, float3(0.299, 0.587, 0.114));
                        float w = innerWeight * lerp(1.0, 1.5, saturate(b - 0.5) * 2);
                        color += s * w;
                        totalWeight += w;
                    }
                    
                    // Outer ring (16 samples at 100% radius)
                    float outerWeight = 1.0 - _BokehShape;
                    outerWeight = max(outerWeight, 0.2);
                    
                    [unroll]
                    for (int k = 0; k < 16; k++)
                    {
                        float2 offset = blurOffsets[k] * blurRadius;
                        half4 s = tex2D(_MainTex, i.uv + offset);
                        float b = dot(s.rgb, float3(0.299, 0.587, 0.114));
                        float w = outerWeight * lerp(1.0, 1.5, saturate(b - 0.5) * 2);
                        color += s * w;
                        totalWeight += w;
                    }
                    
                    color /= totalWeight;
                }
                
                float3 hsv = RGBToHSV(color.rgb);
                hsv.y = saturate(hsv.y * _SaturationBoost);
                color.rgb = HSVToRGB(hsv);
                
                color.rgb = saturate((color.rgb - 0.5) * _ContrastBoost + 0.5);
                
                float2 vignetteUV = i.uv - 0.5;
                float vignette = 1.0 - dot(vignetteUV, vignetteUV) * _VignetteStrength * 2.0;
                color.rgb *= saturate(vignette);
                
                return color;
            }
            ENDCG
        }
    }
}
