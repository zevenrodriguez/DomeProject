// SPDX-License-Identifier: MIT
 
// For compute shader code, see BoxUtil.compute
#ifndef SPLATBOX_HLSL
#define SPLATBOX_HLSL

// View data structure for rendering splats
struct SplatViewData
{
    float4 pos;
    uint axis1, axis2; // screen-space conic axes, each packed as half2 (f16 x in high 16 bits, y in low)
    uint2 color; // 4xFP16
    // Keep this as float4 so the StructuredBuffer stride remains 16-byte aligned on Vulkan/Adreno.
    // A 44-byte stride with float3 worked on desktop but caused unstable reads / flickering on Quest.
    float4 worldPos; // xyz = stable world-space splat center (from compute)
};

#endif // SPLATBOX_HLSL
