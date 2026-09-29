#ifndef SPLATBOX_EQUIRECTANGULAR_PROJECTION_INCLUDED
#define SPLATBOX_EQUIRECTANGULAR_PROJECTION_INCLUDED

int _EquirectEnabled;
Texture2D<float4> _EquirectTexture;
SamplerState sampler_EquirectTexture;
float3 _EquirectProjectorPosition;
float3 _EquirectProjectorScale;
float4 _EquirectProjectorRotation;
int _EquirectBlendMode;
float _EquirectOpacity;
float4 _EquirectTintColor;
int _EquirectFlipHorizontal;
int _EquirectFlipVertical;
float _EquirectUVOffsetX;
float _EquirectUVOffsetY;
float _EquirectTileX;
float _EquirectTileY;
int _EquirectUseDistanceFiltering;
float _EquirectMinDistance;
float _EquirectMaxDistance;

float3 EquirectRotateByQuaternion(float3 value, float4 rotation)
{
    return value + 2.0 * cross(rotation.xyz, cross(rotation.xyz, value) + rotation.w * value);
}

float2 EquirectDirectionToUV(float3 direction)
{
    float theta = atan2(direction.x, direction.z);
    float phi = atan2(direction.y, length(direction.xz));
    return float2(
        0.5 - theta / (2.0 * 3.14159265359),
        0.5 - phi / 3.14159265359);
}

float3 EquirectBlend(float3 baseColor, float4 projectedColor, int blendMode)
{
    if (blendMode == 0)
        return projectedColor.rgb;
    if (blendMode == 1)
        return baseColor * projectedColor.rgb * 2.0;
    if (blendMode == 2)
        return saturate(baseColor + projectedColor.rgb * 0.5);
    if (blendMode == 3)
    {
        return lerp(
            2.0 * baseColor * projectedColor.rgb,
            1.0 - 2.0 * (1.0 - baseColor) * (1.0 - projectedColor.rgb),
            step(0.5, baseColor));
    }

    return lerp(baseColor, projectedColor.rgb, projectedColor.a);
}

float4 ApplyEquirectangularProjection(float3 positionWS, float4 premultipliedColor)
{
    if (_EquirectEnabled == 0 || _EquirectOpacity <= 0.0)
        return premultipliedColor;

    float3 offset = positionWS - _EquirectProjectorPosition;
    float distanceToProjector = length(offset);
    if (_EquirectUseDistanceFiltering != 0 &&
        (distanceToProjector < _EquirectMinDistance || distanceToProjector > _EquirectMaxDistance))
        return premultipliedColor;

    float3 direction = offset / max(distanceToProjector, 0.0001);
    float4 inverseRotation = float4(-_EquirectProjectorRotation.xyz, _EquirectProjectorRotation.w);
    direction = EquirectRotateByQuaternion(direction, inverseRotation);

    // Scale is relative to the transform scale captured when the projector was
    // initialized, so Unity's Scale gizmo acts as a predictable video zoom.
    float3 safeScale = max(abs(_EquirectProjectorScale), 0.0001);

    float2 uv = EquirectDirectionToUV(direction);
    uv = (uv - 0.5) / safeScale.xy + 0.5;
    if (_EquirectFlipHorizontal != 0)
        uv.x = 1.0 - uv.x;
    if (_EquirectFlipVertical != 0)
        uv.y = 1.0 - uv.y;
    uv = uv * float2(_EquirectTileX, _EquirectTileY) +
        float2(_EquirectUVOffsetX, _EquirectUVOffsetY);

    float4 projectedColor = _EquirectTexture.SampleLevel(sampler_EquirectTexture, uv, 0);
    projectedColor *= _EquirectTintColor;

    float alpha = premultipliedColor.a;
    float3 baseColor = alpha > 0.0001 ? premultipliedColor.rgb / alpha : 0.0;
    float3 blended = EquirectBlend(baseColor, projectedColor, _EquirectBlendMode);
    float blendAmount = saturate(_EquirectOpacity);
    if (_EquirectBlendMode == 4)
        blendAmount *= projectedColor.a;

    premultipliedColor.rgb = lerp(baseColor, blended, blendAmount) * alpha;
    return premultipliedColor;
}

#endif
