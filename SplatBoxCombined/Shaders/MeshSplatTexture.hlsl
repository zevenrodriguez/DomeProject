#ifndef SPLATBOX_MESH_SPLAT_TEXTURE_INCLUDED
#define SPLATBOX_MESH_SPLAT_TEXTURE_INCLUDED

// High-resolution mesh albedo sampled across each Gaussian quad.
// One baked dc0 color cannot represent an 8K source; UVs + derivatives can.

int _MeshTexEnabled;
Texture2D _MeshTex;
SamplerState sampler_MeshTex;

struct SplatMeshUv
{
    float2 uv;
    float2 uvSigmaX;
    float2 uvSigmaY;
    float2 _pad;
};

StructuredBuffer<SplatMeshUv> _SplatMeshUvs;

float2 ComputeSplatMeshUv(uint splatIndex, float2 quadPos)
{
    if (_MeshTexEnabled == 0)
        return 0;

    SplatMeshUv data = _SplatMeshUvs[splatIndex];
    // quadPos is in [-2, 2] = two Gaussian sigmas. Map back to sigma units.
    return data.uv + quadPos.x * 0.5 * data.uvSigmaX + quadPos.y * 0.5 * data.uvSigmaY;
}

half3 ApplyMeshSplatAlbedo(half3 bakedRgb, float2 meshUv)
{
    if (_MeshTexEnabled == 0)
        return bakedRgb;
    return (half3)_MeshTex.SampleLevel(sampler_MeshTex, meshUv, 0).rgb;
}

#endif
