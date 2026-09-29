 
// Player/build copy loaded via Resources.Load("RenderSplatsRelight").
// Unique name (vs Shaders/Relight/RenderGaussianSplats.shader) avoids duplicate-shader-name
// dedupe in player builds, which silently drops one shader and breaks rendering.
Shader "Gaussian Splatting/Render Splats Relight (Player)"
{
    Properties
    {
        _DirectionalIntensity ("Directional Light Intensity", Range(0, 2)) = 1
        _AmbientIntensity ("Ambient Light Intensity", Range(0, 2)) = 1
        _ShadowStrength ("Shadow Strength", Range(0, 1)) = 0.5
        _AmbientShadowStrength ("Ambient Shadow Strength", Range(0, 1)) = 1.0
        _AmbientShadowDistanceScale ("Ambient Shadow Distance Scale", Range(0.1, 5)) = 2.0
        _SSAOStrength ("SSAO Strength", Range(0, 1)) = 0.0
        _SharpenStrength ("Sharpen Strength", Range(0, 2)) = 0.0
        _GaussianSSAO ("SSAO Texture", 2D) = "white" {}
       // _AudioColorTint ("Audio Color Tint", Color) = (1,1,1,1)
    }

    SubShader
    {
        Tags { 
            "RenderType" = "Transparent" 
            "Queue" = "Transparent"
            "RenderPipeline" = "UniversalPipeline"
        }

        Pass
        {
            Name "ForwardLit"
            Tags { "LightMode"="UniversalForward" }
            
            ZWrite Off
            // ZTest Always: splats render into their own offscreen RT (composited on top later),
            // so no depth test is needed. Without this, a default LEqual test against a NON-bound
            // depth attachment discards every fragment on Vulkan/Quest (renders fine on DX11 editor).
            ZTest Always
            Blend OneMinusDstAlpha One
            Cull Off
            
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // No #pragma use_dxc: nothing here needs SM6, and DXC's auto-binding collides on Vulkan
            // ("resource '_OrderBuffer' shares binding N with another resource") once URP's Lighting.hlsl
            // resources are added, which leaves the splat buffers unbound and the draw skipped.
            #pragma require compute
            #pragma multi_compile_local __ DEBUG_SPLAT_RENDER
            #pragma multi_compile_local __ DEBUG_BINDINGS
            // Main-light shadows are sampled in the vertex shader by CalculateGSLighting.
            // These cannot be fragment-only variants because each splat's lighting is evaluated per vertex.
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN
            #pragma multi_compile _ _SHADOWS_SOFT
            // Use Multi-Pass VR rendering instead of Single Pass Instanced (known issue)
            #pragma multi_compile _ STEREO_MULTIVIEW_ON

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/UnityInstancing.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"
            #include "../Shaders/SplatBox.hlsl"
            #include "../Shaders/EquirectangularProjection.hlsl"
            #include "../Shaders/MeshSplatTexture.hlsl"

            StructuredBuffer<uint> _OrderBuffer;
            uint _SplatCount; // valid splat count; guards against out-of-range sort indices
            // Scene-depth occlusion. The splats draw into their own colour-only RT with ZTest Always (see the
            // pass state above), so opaque scene geometry can never hide them through the depth buffer.
            // Instead the fragment compares the splat's eye depth with URP's _CameraDepthTexture, the same
            // way URP soft particles do. The URP feature requests the depth texture when this is on.
            uint _SceneDepthOcclusion;  // 1 = fade out splats that lie behind opaque scene geometry
            float _SceneDepthFade;      // metres over which a splat fades as it passes behind geometry
            uint _SceneDepthTextureValid;
            float4 _SplatRenderTargetSize; // xy = splat RT size, zw = reciprocal size

            CBUFFER_START(UnityPerMaterial)
                float _DirectionalIntensity;
                float _AmbientIntensity;
                float _ShadowStrength;
                float _AmbientShadowStrength;
                float _AmbientShadowDistanceScale;
                float _SSAOStrength;
                float _SharpenStrength;
                float4 _AudioColorTint;
                float _DepthOffset;
                
                // Color adjustments
                float _ColorExposure;
                float _ColorContrast;
                float4 _ColorFilter;
                float _ColorHueShift;
                float _ColorSaturation;
                
                // Selection colors
                float4 _SelectionColor;
                float4 _TargetSelectionColor;
                
                // Decal properties
                int _DecalEnabled;
                float4x4 _DecalWorldToProjector;
                float4x4 _DecalProjectorToWorld;
                float3 _DecalBoxSize;
                int _DecalProjectionMode;
                int _DecalBlendMode;
                float _DecalOpacity;
                float4 _DecalTintColor;
                float _DecalFrustumFOV;
                float _DecalFrustumNear;
                float _DecalFrustumFar;
                int _DecalUseNormalFiltering;
                float _DecalNormalThreshold;
                int _DecalUseDepthFade;
                float _DecalDepthFadeStart;
                float _DecalDepthFadeEnd;
                int _DecalUseCircleMask;
                float2 _DecalMaskCenter;
                float _DecalMaskRadius;
                float _DecalMaskSoftness;
                int _DecalInvertMask;
            CBUFFER_END

            TEXTURE2D(_DecalTexture);
            SAMPLER(sampler_DecalTexture);

            struct v2f
            {
                half4 col : COLOR0;
                float2 pos : TEXCOORD0;
                float4 vertex : SV_POSITION;
                // positionWS / normalWS are still needed per-fragment by ApplyDecal (gradient texture
                // sampling is invalid in the vertex stage, so the decal must stay in the fragment).
                float3 normalWS : TEXCOORD1;
                float3 positionWS : TEXCOORD2;
                // Final per-splat lit RGB, computed once in the vertex shader.
                // TEXCOORD (not COLOR0) so HDR values >1 are not clamped by the interpolator.
                half3 litRgb : TEXCOORD3;
                float eyeDepth : TEXCOORD4;
                float3 projectionPositionWS : TEXCOORD5;
                float2 meshUv : TEXCOORD6;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            StructuredBuffer<SplatViewData> _SplatViewData;
            ByteAddressBuffer _SplatSelectedBits;
            ByteAddressBuffer _SplatTargetSelectedBits;
            uint _SplatBitsValid;

            struct CustomPointLight {
                float4 position;
                float4 color;
                float intensity;
                float range;
                float attenuationMode;
                float _pad;
            };

            struct CustomSpotLight {
                float4 position;
                float4 direction;
                float4 color;
                float intensity;
                float range;
                float innerConeAngle;  // cosine of inner cone half-angle
                float outerConeAngle;  // cosine of outer cone half-angle
                float attenuationMode;
                float _pad0;
                float _pad1;
                float _pad2;
            };

            struct CustomDirectionalLight {
                float4 direction;
                float4 color;
                float intensity;
                float effectRange;
                float rangeSoftness;
                float shadowStrength;
                float useDistanceAttenuation;
                float distanceAttenuationFactor;
                float useAngleAttenuation;
                float angleAttenuationBias;
            };

            struct CustomPlanarLight {
                float4 position;
                float4 normal;
                float4 color;
                float intensity;
                float range;
                float sizeX;
                float sizeY;
                float twoSided;
                float attenuationMode;
                float _pad0;
                float _pad1;
            };

            StructuredBuffer<CustomPointLight> _CustomPointLights;
            StructuredBuffer<CustomSpotLight> _CustomSpotLights;
            StructuredBuffer<CustomDirectionalLight> _CustomDirectionalLights;
            StructuredBuffer<CustomPlanarLight> _CustomPlanarLights;
            uint _CustomPointLightCount;
            uint _CustomSpotLightCount;
            uint _CustomDirectionalLightCount;
            uint _CustomPlanarLightCount;
            TEXTURE2D(_GaussianSSAO);
            SAMPLER(sampler_GaussianSSAO);

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
                // Apply exposure (brightness)
                color *= exp2(_ColorExposure);
                
                // Apply contrast
                color = lerp(0.5, color, _ColorContrast);
                
                // Convert to HSV for hue shift and saturation
                float3 hsv = RGBToHSV(color);
                
                // Apply hue shift (in degrees, convert to 0-1 range)
                hsv.x = frac(hsv.x + _ColorHueShift / 360.0);
                
                // Apply saturation
                hsv.y *= _ColorSaturation;
                
                // Convert back to RGB
                color = HSVToRGB(hsv);
                
                // Apply color filter (multiply)
                color *= _ColorFilter.rgb;
                
                return color;
            }
            
            float4 ApplyDecal(float3 positionWS, float3 normalWS, float4 baseColor)
            {
                if (_DecalEnabled == 0) return baseColor;
                
                // Transform world position to projector local space
                float3 projectorLocalPos = mul(_DecalWorldToProjector, float4(positionWS, 1.0)).xyz;
                
                // Check if position is within projection volume
                bool inVolume = false;
                float2 uv = float2(0, 0);
                
                if (_DecalProjectionMode == 0) // Box
                {
                    float3 absPos = abs(projectorLocalPos);
                    half3 boxHalfSize = _DecalBoxSize * 0.5;
                    inVolume = all(absPos <= boxHalfSize);
                    if (inVolume)
                    {
                        // Map to UV (0-1)
                        uv = (projectorLocalPos.xy / _DecalBoxSize.xy) + 0.5;
                    }
                }
                else // Frustum (1)
                {
                    // For frustum projection, use perspective division
                    float depth = -projectorLocalPos.z;
                    if (depth > _DecalFrustumNear && depth < _DecalFrustumFar)
                    {
                        // Calculate frustum size at this depth
                        float fovRad = _DecalFrustumFOV * 0.5 * 0.01745329; // deg to rad
                        float halfHeight = tan(fovRad) * depth;
                        float halfWidth = halfHeight;
                        
                        if (abs(projectorLocalPos.x) < halfWidth && abs(projectorLocalPos.y) < halfHeight)
                        {
                            inVolume = true;
                            uv = float2(projectorLocalPos.x / (halfWidth * 2.0) + 0.5,
                                       projectorLocalPos.y / (halfHeight * 2.0) + 0.5);
                        }
                    }
                }
                
                if (!inVolume) return baseColor;
                
                float finalOpacity = _DecalOpacity;
                
                // Apply circle mask
                if (_DecalUseCircleMask == 1)
                {
                    float2 maskUV = uv - _DecalMaskCenter;
                    float dist = length(maskUV);
                    float mask = smoothstep(_DecalMaskRadius + _DecalMaskSoftness, 
                                           _DecalMaskRadius - _DecalMaskSoftness, dist);
                    if (_DecalInvertMask == 1) mask = 1.0 - mask;
                    if (mask < 0.01) return baseColor;
                    finalOpacity *= mask;
                }
                
                // Depth fade
                if (_DecalUseDepthFade == 1)
                {
                    float depth = abs(projectorLocalPos.z) / _DecalBoxSize.z;
                    float fade = smoothstep(_DecalDepthFadeStart, _DecalDepthFadeEnd, depth);
                    finalOpacity *= (1.0 - fade);
                }
                
                // Sample decal texture
                float4 decalColor = SAMPLE_TEXTURE2D(_DecalTexture, sampler_DecalTexture, uv);
                decalColor *= _DecalTintColor;
                
                // Apply blend mode
                float4 result = baseColor;
                if (_DecalBlendMode == 0) // Replace
                {
                    result.rgb = lerp(baseColor.rgb, decalColor.rgb, decalColor.a * finalOpacity);
                }
                else if (_DecalBlendMode == 1) // Multiply
                {
                    result.rgb = lerp(baseColor.rgb, baseColor.rgb * decalColor.rgb, decalColor.a * finalOpacity);
                }
                else if (_DecalBlendMode == 2) // Additive
                {
                    result.rgb = baseColor.rgb + decalColor.rgb * decalColor.a * finalOpacity;
                }
                else if (_DecalBlendMode == 3) // Overlay
                {
                    float3 overlay;
                    overlay.r = baseColor.r < 0.5 ? (2.0 * baseColor.r * decalColor.r) : (1.0 - 2.0 * (1.0 - baseColor.r) * (1.0 - decalColor.r));
                    overlay.g = baseColor.g < 0.5 ? (2.0 * baseColor.g * decalColor.g) : (1.0 - 2.0 * (1.0 - baseColor.g) * (1.0 - decalColor.g));
                    overlay.b = baseColor.b < 0.5 ? (2.0 * baseColor.b * decalColor.b) : (1.0 - 2.0 * (1.0 - baseColor.b) * (1.0 - decalColor.b));
                    result.rgb = lerp(baseColor.rgb, overlay, decalColor.a * finalOpacity);
                }
                else // AlphaBlend (4)
                {
                    result.rgb = lerp(baseColor.rgb, decalColor.rgb, decalColor.a * finalOpacity);
                }
                
                return result;
            }

			float CalculateAttenuation(float distance, float range, float mode)
            {
                float normalizedDistance = distance / range;
                
                // Mode 0: Default (Quadratic)
                if (mode < 0.5)
                {
                    float attenuation = saturate(1.0 - normalizedDistance * normalizedDistance);
                    return attenuation * attenuation;
                }
                // Mode 1: Linear
                else if (mode < 1.5)
                {
                    return saturate(1.0 - normalizedDistance);
                }
                // Mode 2: Inverse Square
                else
                {
                    return 1.0 / (1.0 + normalizedDistance * normalizedDistance);
                }
            }

            // World-up normal for relight: stable on mobile (never use camera view direction).
            static const float3 kRelightShadingNormal = float3(0, 1, 0);

            half3 CalculateGSLighting(float3 positionWS, float4 shadowCoord)
            {
                half3 totalLighting = half3(0, 0, 0);
                float3 shadingNormal = kRelightShadingNormal;
                
                Light mainLight = GetMainLight(shadowCoord);
                float NdotL = dot(shadingNormal, mainLight.direction) * 0.5 + 0.5;
               
                float shadowAttenuation = mainLight.shadowAttenuation;
                
                float depth = length(positionWS);
                float distanceAtten = 1.0 / (1.0 + depth * 0.5);  
                float ambientDistanceAtten = 1.0 / (1.0 + depth * _AmbientShadowDistanceScale);
                
                float upDot = dot(mainLight.direction, float3(0, 1, 0));
                float angleAttenuation = saturate(upDot + 0.5);
                
                float finalAttenuation = shadowAttenuation * distanceAtten * angleAttenuation;
                float ambientFinalAttenuation = shadowAttenuation * ambientDistanceAtten * angleAttenuation;
                
                // Shadow/distance darkening factors. 0 -> 1.0 (no darkening, splats keep their original
                // albedo); higher values let distance + shadow attenuation pull the splat down. The ambient
                // term has its own strength (_AmbientShadowStrength) so the distance-based shadow can be
                // pushed much darker than the additive directional contribution.
                float shadowFactor = 1 - (_ShadowStrength * (1 - finalAttenuation));
                float ambientShadowFactor = 1 - (_AmbientShadowStrength * (1 - ambientFinalAttenuation));
                
                // Neutral white ambient fill. We intentionally do NOT use SampleSH here: the environment
                // SH probe is not reliably bound during our procedural splat draw, which produced a random
                // per-session color cast (green/yellow). A neutral fill keeps splats at their ORIGINAL
                // albedo when no relight lights are present; the custom lights below add illumination.
                // The ambient is modulated by ambientShadowFactor so distance-based attenuation darkens the
                // WHOLE splat (far/shadowed splats fall off) instead of being held up by a constant floor.
                half3 ambient = half3(1.0h, 1.0h, 1.0h);
                totalLighting += ambient * _AmbientIntensity * ambientShadowFactor;
               
                half3 directionalLight = mainLight.color * NdotL * _DirectionalIntensity;
                directionalLight *= shadowFactor;
                
                totalLighting += directionalLight;
                
                // Custom point lights — world-space distance falloff only (no view-dependent normal).
                for (uint i = 0; i < _CustomPointLightCount; i++)
                {
                    CustomPointLight light = _CustomPointLights[i];
                    
                    float3 toLight = light.position.xyz - positionWS;
                    float distance = length(toLight);
                    float attenuation = CalculateAttenuation(distance, light.range, light.attenuationMode);
                    
                    totalLighting += light.color.rgb * light.intensity * attenuation;
                }
                
                // Custom spot lights — world-space cone + distance (no view-dependent normal).
                for (uint j = 0; j < _CustomSpotLightCount; j++)
                {
                    CustomSpotLight spotLight = _CustomSpotLights[j];
                    
                    float3 toLight = spotLight.position.xyz - positionWS;
                    float distance = length(toLight);
                    float3 lightDir = toLight / max(distance, 1e-4);
                    
                    float distanceAttenuation = CalculateAttenuation(distance, spotLight.range, spotLight.attenuationMode);
                    
                    float3 spotDirection = normalize(spotLight.direction.xyz);
                    float cosAngle = dot(-lightDir, spotDirection);
                    float coneSpan = spotLight.innerConeAngle - spotLight.outerConeAngle;
                    float angularAttenuation = (coneSpan > 1e-5) ? saturate((cosAngle - spotLight.outerConeAngle) / coneSpan) : saturate(sign(cosAngle - spotLight.outerConeAngle));
                    
                    float totalAttenuation = distanceAttenuation * angularAttenuation;
                    
                    totalLighting += spotLight.color.rgb * spotLight.intensity * totalAttenuation;
                }
                
                // Custom directional lights
                for (uint k = 0; k < _CustomDirectionalLightCount; k++)
                {
                    CustomDirectionalLight dirLight = _CustomDirectionalLights[k];
                    
                    if (dirLight.intensity <= 0.0)
                        continue;
                    
                    float3 lightDir = normalize(dirLight.direction.xyz);
                    float dirNdotL = dot(shadingNormal, lightDir) * 0.5 + 0.5;
                    
                    float distanceFromOrigin = length(positionWS);
                    float rangeStart = max(0.0, dirLight.effectRange - max(0.0, dirLight.rangeSoftness));
                    float rangeAtten = 1.0 - smoothstep(rangeStart, dirLight.effectRange, distanceFromOrigin);
                    
                    float normalInfluence = lerp(1.0, dirNdotL, saturate(dirLight.shadowStrength));
                    
                    float finalDirAttenuation = rangeAtten;
                    
                    if (dirLight.useDistanceAttenuation > 0.5)
                    {
                        float distFactor = saturate(dirLight.distanceAttenuationFactor);
                        float dirDistAtten = 1.0 / (1.0 + distanceFromOrigin * distFactor);
                        finalDirAttenuation *= dirDistAtten;
                    }
                    
                    if (dirLight.useAngleAttenuation > 0.5)
                    {
                        float upDotDir = dot(lightDir, float3(0, 1, 0));
                        float angleAtten = saturate(upDotDir + dirLight.angleAttenuationBias);
                        finalDirAttenuation *= max(0.2, angleAtten);
                    }
                    
                    half3 dirContribution = dirLight.color.rgb * dirLight.intensity * normalInfluence;
                    dirContribution *= finalDirAttenuation;
                    
                    totalLighting += dirContribution;
                }
                
                // Custom planar lights
                for (uint p = 0; p < _CustomPlanarLightCount; p++)
                {
                    CustomPlanarLight planar = _CustomPlanarLights[p];
                    
                    if (planar.intensity <= 0.0)
                        continue;
                    
                    float3 toSplat = positionWS - planar.position.xyz;
                    float distance = length(toSplat);
                    float3 lightDir = normalize(toSplat);
                    
                    float3 planeNormal = normalize(planar.normal.xyz);
                    
                    float dotNormal = dot(toSplat, planeNormal);
                    
                    bool correctSide = (dotNormal > 0.0) || (planar.twoSided > 0.5);
                    if (!correctSide)
                        continue;
                    
                    float3 projectedPos = positionWS - planeNormal * dotNormal;
                    float3 localPos = projectedPos - planar.position.xyz;
                    
                    float3 planeRight = normalize(cross(planeNormal, float3(0, 1, 0)));
                    if (length(planeRight) < 0.1)
                        planeRight = normalize(cross(planeNormal, float3(1, 0, 0)));
                    float3 planeUp = normalize(cross(planeNormal, planeRight));
                    
                    float localX = dot(localPos, planeRight);
                    float localY = dot(localPos, planeUp);
                    
                    float halfSizeX = planar.sizeX * 0.5;
                    float halfSizeY = planar.sizeY * 0.5;
                    
                    bool withinBounds = (abs(localX) <= halfSizeX) && (abs(localY) <= halfSizeY);
                    
                    if (!withinBounds)
                        continue;
                    
                    float distanceAtten = CalculateAttenuation(distance, planar.range, planar.attenuationMode);
                    
                    float edgeFalloffX = 1.0 - smoothstep(halfSizeX * 0.7, halfSizeX, abs(localX));
                    float edgeFalloffY = 1.0 - smoothstep(halfSizeY * 0.7, halfSizeY, abs(localY));
                    float edgeFalloff = edgeFalloffX * edgeFalloffY;
                    
                    half3 planarContribution = planar.color.rgb * planar.intensity * distanceAtten * edgeFalloff;
                    
                    totalLighting += planarContribution;
                }
                
                return totalLighting;
            }

            struct appdata
            {
                uint vtxID : SV_VertexID;
                uint instID : SV_InstanceID;
            };

            v2f vert(appdata input)
            {
                v2f o = (v2f)0;
                
                // Multi-Pass VR support (no instance ID conflicts)
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

                uint vtxID = input.vtxID;
                uint instID = input.instID;
                instID = _OrderBuffer[instID];
                // The GPU radix sort can occasionally emit an out-of-range / garbage payload index on
                // some mobile (Adreno/Vulkan) GPUs. Reading _SplatViewData with it pulls garbage data
                // and draws bright "noise" splats at random positions. Discard those primitives.
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

                    // Calculate vertex position in clip space
                    float2 axis1 = float2(f16tof32(view.axis1 >> 16), f16tof32(view.axis1));
                    float2 axis2 = float2(f16tof32(view.axis2 >> 16), f16tof32(view.axis2));
                    // _ScreenParams describes the camera target, which can be larger than the
                    // low-resolution splat target. Use the actual splat target dimensions so the
                    // covariance compute and quad expansion agree at every Render Scale.
                    float2 splatTargetSize = _SplatRenderTargetSize.x > 0
                        ? _SplatRenderTargetSize.xy
                        : _ScreenParams.xy;
                    float2 deltaScreenPos = (quadPos.x * axis1 + quadPos.y * axis2) * 2 / splatTargetSize;
                    o.vertex = centerClipPos;
                    o.vertex.xy += deltaScreenPos * centerClipPos.w;

                    // Approximate a world position for each point on the camera-facing
                    // Gaussian quad. Interpolation gives the fragment projector a
                    // different UV across large splats instead of one color per splat.
                    float2 viewOffset = float2(
                        deltaScreenPos.x * centerClipPos.w / UNITY_MATRIX_P._m00,
                        deltaScreenPos.y * centerClipPos.w / UNITY_MATRIX_P._m11);
                    o.projectionPositionWS = view.worldPos.xyz +
                        mul((float3x3)UNITY_MATRIX_I_V, float3(viewOffset, 0.0));

                    // Perspective clip.w is eye depth; orthographic projections need view-space z.
                    float eyeDepth = unity_OrthoParams.w > 0.5
                        ? -TransformWorldToView(view.worldPos.xyz).z
                        : centerClipPos.w;
                    o.eyeDepth = eyeDepth;

                    // Stable world position from compute (avoid I_VP unproject — breaks on mobile/Vulkan).
                    o.positionWS = view.worldPos.xyz;
                    o.normalWS = kRelightShadingNormal;

                    // Per-splat relighting. positionWS / normal / base color are constant across the splat
                    // quad, so the full multi-light loop + SH + shadow produce an identical result for every
                    // covered fragment. Evaluate it ONCE here instead of per fragment — under Gaussian
                    // overdraw this is the dominant relight cost on Quest. shadowCoord is a vertex-local
                    // (no longer interpolated) since the fragment never reads it.
                    {
                        float4 shadowCoord = TransformWorldToShadowCoord(o.positionWS);
                        half3 lighting = CalculateGSLighting(o.positionWS, shadowCoord);
                        // Keep only a tiny floor so intentionally dark ambient distance shadows are not
                        // clamped back up. Do not snap near-zero lighting to white here: with strong ambient
                        // shadowing, near-zero can be the intended result.
                        lighting = max(lighting, half3(0.01h, 0.01h, 0.01h));
                        lighting = min(lighting, half3(4.0, 4.0, 4.0));
                        // Only the (expensive) multi-light loop is per-splat. Decal + color grading stay
                        // in the fragment: the decal samples a texture (gradient/bias sampling is invalid
                        // in the vertex stage), and this preserves the original decal -> color-grade order.
                        o.litRgb = o.col.rgb * lighting;
                    }

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

            half4 frag(v2f i) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

                #if defined(DEBUG_BINDINGS)
                return half4(1,0,0,1);
                #endif

                float power = -dot(i.pos, i.pos);
                half alpha = exp(power);
                
                // Apply sharpening after exp for more visible effect
                if (_SharpenStrength > 0.01)
                {
                    // Sharpen by raising alpha to higher power (makes edges crisper)
                    // Higher power = steeper falloff at edges
                    float sharpenPower = 1.0 + (_SharpenStrength * 3.0);
                    alpha = pow(alpha, sharpenPower);
                }

                // Lighting (the expensive multi-light loop) was already applied per-splat in the vertex
                // shader. The fragment resolves the Gaussian alpha + selection, then decal + color grade.
                half3 albedo = ApplyMeshSplatAlbedo(i.col.rgb, i.meshUv);
                half3 safeBaked = max(i.col.rgb, half3(0.04h, 0.04h, 0.04h));
                half3 lighting = i.litRgb / safeBaked;
                half3 outRgb = albedo * lighting;

                // Debug: Tint splats red when sharpen is active
                if (_SharpenStrength > 0.5)
                {
                    outRgb = lerp(outRgb, half3(1,0,0), 0.3);
                }
                
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
                        // Mix both colors
                        selColor = lerp(_SelectionColor.rgb, _TargetSelectionColor.rgb, 0.5);
                    }
                    else if (i.col.a < -1.5) // Target selected (-2)
                    {
                        selColor = _TargetSelectionColor.rgb;
                    }
                    // else: Source selected (-1), use default _SelectionColor
                    
                    if (alpha > 7.0/255.0)
                    {
                        if (alpha < 10.0/255.0)
                        {
                            alpha = 1;
                            outRgb = selColor;
                        }
                        alpha = saturate(alpha + 0.3);
                    }
                    outRgb = lerp(outRgb, selColor, 0.5);
                }

                // Scene-depth occlusion: fade the splat out where opaque geometry is nearer than its centre.
                if (_SceneDepthOcclusion != 0 && _SceneDepthTextureValid != 0)
                {
                    // SV_POSITION is a pixel coordinate in the actual splat RT. Normalizing that
                    // coordinate is stable across Scene/Game cameras, Vulkan/D3D and render scaling.
                    float2 depthUv = i.vertex.xy * _SplatRenderTargetSize.zw;
                    // Convert raster coordinates to URP's normalized-screen convention, then use its
                    // RTHandle-aware loader. This applies the Game camera's render scale and viewport;
                    // indexing by _CameraDepthTexture_TexelSize sampled outside the scaled viewport.
                    TransformNormalizedScreenUV(depthUv);
                    float rawDepth = SampleSceneDepth(depthUv);
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
                        sceneEyeDepth = LinearEyeDepth(rawDepth, _ZBufferParams);
                    }
                    // Keep the splat fully visible when it is on or in front of the opaque surface,
                    // then soften only the portion that moves behind it. The previous fade window was
                    // reversed: equal depth produced zero alpha, which cut a hard primitive-shaped hole.
                    float depthGap = sceneEyeDepth - i.eyeDepth;
                    alpha *= saturate(1.0 + depthGap / max(_SceneDepthFade, 1e-4));
                }
                
                if (alpha < 1.0/255.0)
                    discard;

                // Premultiply by alpha BEFORE decal + color grading, then return — identical order to the
                // original per-fragment path. ApplyColorAdjustments is non-linear (contrast/saturation/
                // color filter), so it must run on the premultiplied color or it shifts hue (green wash).
                // Only the lighting itself was hoisted to the vertex stage.
                half4 res = half4(outRgb * alpha, alpha);
                res = ApplyDecal(i.positionWS, i.normalWS, res);
                res = ApplyEquirectangularProjection(i.projectionPositionWS, res);
                res.rgb = ApplyColorAdjustments(res.rgb);
                return res;
            }
            ENDHLSL
        }

        Pass
        {
            Name "BlitToScreen"
            Tags { "LightMode" = "Always" }
            ZWrite Off
            ZTest Always
            Cull Off
            Blend SrcAlpha OneMinusSrcAlpha

            CGPROGRAM
            #pragma vertex vertComposite
            #pragma fragment fragComposite
            // No #pragma use_dxc: nothing here needs SM6, and DXC's auto-binding collides on Vulkan
            // ("resource '_OrderBuffer' shares binding N with another resource") once URP's Lighting.hlsl
            // resources are added, which leaves the splat buffers unbound and the draw skipped.
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
                float2 quadPos = float2(vtxID & 1, (vtxID >> 1) & 1) * 4.0 - 1.0;
                o.vertex = float4(quadPos, 1, 1);
                return o;
            }

            Texture2D _GaussianSplatRT;
            SamplerState sampler_GaussianSplatRT;

            half4 fragComposite(v2fComposite i) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
                // Sample (not Load) so the splat RT can be rendered at a fraction of the screen
                // resolution (SplatBoxRenderManager.renderScale) and bilinearly upscaled here. The
                // splats are rasterized in clip space, so the low-res RT maps 1:1 in NDC; uv is the
                // destination pixel normalized by the full-res screen.
                float2 uv = i.vertex.xy / _ScreenParams.xy;
                half4 col = _GaussianSplatRT.Sample(sampler_GaussianSplatRT, uv);
                col.rgb = min(col.rgb, half3(4.0, 4.0, 4.0));
                return col;
            }
            ENDCG
        }
    }
}
