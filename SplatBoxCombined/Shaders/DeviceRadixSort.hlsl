/******************************************************************************
 * Device Level 8-bit LSD Radix Sort using reduce then scan
 * 
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 5/17/2024
 * https://github.com/b0nes164/GPUSorting
 *  
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 ******************************************************************************/
// Specialised for Splat Box: uint keys, uint payloads, ascending order, key/payload pairs.
//
// This is the wave-size-agnostic revision of GPUSorting. Qualcomm/Adreno Vulkan drivers (and any
// SPIR-V below 1.6) report a subgroup size through WaveGetLaneCount() that does not match the wave
// that actually executes the kernel. Every shared-memory offset in a radix sort is derived from the
// wave size, so a wrong value silently scatters keys and payloads to the wrong slots. The resulting
// garbage payload indices are what made splats flicker and vanish on Quest. The fix is to never call
// WaveGetLaneCount() and instead count the real active lanes with a ballot, then thread that value
// through every helper as an argument.

//General macros 
#define PART_SIZE       3840U       //size of a partition tile

#define US_DIM          128U        //The number of threads in a Upsweep threadblock
#define SCAN_DIM        128U        //The number of threads in a Scan threadblock
#define DS_DIM          256U        //The number of threads in a Downsweep threadblock

#define RADIX           256U        //Number of digit bins
#define RADIX_MASK      255U        //Mask of digit bins
#define RADIX_LOG       8U          //log2(RADIX)

#define HALF_RADIX      128U        //For smaller waves where bit packing is necessary
#define HALF_MASK       127U        // '' 

//For the downsweep kernels
#define DS_KEYS_PER_THREAD  15U     //The number of keys per thread in a Downsweep Threadblock
#define MAX_DS_SMEM         4096U   //shared memory for downsweep kernel

cbuffer cbParallelSort : register(b0)
{
    uint e_numKeys;
    uint e_radixShift;
    uint e_threadBlocks;
    uint padding0;
};

RWStructuredBuffer<uint> b_sort;            //buffer to be sorted
RWStructuredBuffer<uint> b_alt;             //double buffer
RWStructuredBuffer<uint> b_sortPayload;     //payload buffer
RWStructuredBuffer<uint> b_altPayload;      //double buffer payload
RWStructuredBuffer<uint> b_globalHist;      //buffer holding global device level offsets 
                                            //for each digit during a binning pass
RWStructuredBuffer<uint> b_passHist;        //buffer used to store device level offsets for 
                                            //each partition tile for each digit during a binning pass

groupshared uint g_us[RADIX * 2];           //Shared memory for upsweep kernel
groupshared uint g_scan[SCAN_DIM];          //Shared memory for the scan kernel
groupshared uint g_ds[MAX_DS_SMEM];         //Shared memory for the downsweep kernel

struct KeyStruct
{
    uint k[DS_KEYS_PER_THREAD];
};

struct OffsetStruct
{
    uint o[DS_KEYS_PER_THREAD];
};

struct DigitStruct
{
    uint d[DS_KEYS_PER_THREAD];
};

//*****************************************************************************
//HELPER FUNCTIONS
//*****************************************************************************
//Never use WaveGetLaneCount(): see the note at the top of this file. Ballot over a fully converged
//wave and count the bits to get the number of lanes that are genuinely executing together.
inline uint getWaveSize()
{
    GroupMemoryBarrierWithGroupSync(); //make absolutely sure the wave is not diverged here
    const uint4 ballot = WaveActiveBallot(true);
    return countbits(ballot.x) + countbits(ballot.y) + countbits(ballot.z) + countbits(ballot.w);
}

inline uint getWaveIndex(uint gtid, uint waveSize)
{
    return gtid / waveSize;
}

inline uint ExtractDigit(uint key)
{
    return key >> e_radixShift & RADIX_MASK;
}

inline uint ExtractPackedIndex(uint key)
{
    return key >> (e_radixShift + 1) & HALF_MASK;
}

inline uint ExtractPackedShift(uint key)
{
    return (key >> e_radixShift & 1) ? 16 : 0;
}

inline uint ExtractPackedValue(uint packed, uint key)
{
    return packed >> ExtractPackedShift(key) & 0xffff;
}

inline uint SubPartSizeWGE16(uint waveSize)
{
    return DS_KEYS_PER_THREAD * waveSize;
}

inline uint SharedOffsetWGE16(uint gtid, uint waveSize)
{
    return WaveGetLaneIndex() + getWaveIndex(gtid, waveSize) * SubPartSizeWGE16(waveSize);
}

inline uint DeviceOffsetWGE16(uint gtid, uint waveSize, uint partIndex)
{
    return SharedOffsetWGE16(gtid, waveSize) + partIndex * PART_SIZE;
}

inline uint SubPartSizeWLT16(uint waveSize, uint _serialIterations)
{
    return DS_KEYS_PER_THREAD * waveSize * _serialIterations;
}

inline uint SharedOffsetWLT16(uint gtid, uint waveSize, uint _serialIterations)
{
    return WaveGetLaneIndex() +
        (getWaveIndex(gtid, waveSize) / _serialIterations * SubPartSizeWLT16(waveSize, _serialIterations)) +
        (getWaveIndex(gtid, waveSize) % _serialIterations * waveSize);
}

inline uint DeviceOffsetWLT16(uint gtid, uint waveSize, uint partIndex, uint _serialIterations)
{
    return SharedOffsetWLT16(gtid, waveSize, _serialIterations) + partIndex * PART_SIZE;
}

inline uint GlobalHistOffset()
{
    return e_radixShift << 5;
}

inline uint WaveHistsSizeWGE16(uint waveSize)
{
    return DS_DIM / waveSize * RADIX;
}

inline uint WaveHistsSizeWLT16()
{
    return MAX_DS_SMEM;
}

//*****************************************************************************
//INIT KERNEL
//*****************************************************************************
//Clear the global histogram, as we will be adding to it atomically
[numthreads(1024, 1, 1)]
void InitDeviceRadixSort(int3 id : SV_DispatchThreadID)
{
    b_globalHist[id.x] = 0;
}

//*****************************************************************************
//UPSWEEP KERNEL
//*****************************************************************************
//histogram, 64 threads to a histogram
inline void HistogramDigitCounts(uint gtid, uint gid)
{
    const uint histOffset = gtid / 64 * RADIX;
    const uint partitionEnd = gid == e_threadBlocks - 1 ?
        e_numKeys : (gid + 1) * PART_SIZE;
    for (uint i = gtid + gid * PART_SIZE; i < partitionEnd; i += US_DIM)
        InterlockedAdd(g_us[ExtractDigit(b_sort[i]) + histOffset], 1);
}

//reduce and pass to tile histogram
inline void ReduceWriteDigitCounts(uint gtid, uint gid)
{
    for (uint i = gtid; i < RADIX; i += US_DIM)
    {
        g_us[i] += g_us[i + RADIX];
        b_passHist[i * e_threadBlocks + gid] = g_us[i];
        g_us[i] += WavePrefixSum(g_us[i]);
    }
}

//Exclusive scan over digit counts, then atomically add to global hist
inline void GlobalHistExclusiveScanWGE16(uint gtid, uint waveSize)
{
    GroupMemoryBarrierWithGroupSync();

    if (gtid < (RADIX / waveSize))
    {
        g_us[(gtid + 1) * waveSize - 1] +=
            WavePrefixSum(g_us[(gtid + 1) * waveSize - 1]);
    }
    GroupMemoryBarrierWithGroupSync();

    //atomically add to global histogram
    const uint globalHistOffset = GlobalHistOffset();
    const uint laneMask = waveSize - 1;
    const uint circularLaneShift = WaveGetLaneIndex() + 1 & laneMask;
    for (uint i = gtid; i < RADIX; i += US_DIM)
    {
        const uint index = circularLaneShift + (i & ~laneMask);
        uint t = WaveGetLaneIndex() != laneMask ? g_us[i] : 0;
        if (i >= waveSize)
            t += WaveReadLaneAt(g_us[i - 1], 0);
        InterlockedAdd(b_globalHist[index + globalHistOffset], t);
    }
}

inline void GlobalHistExclusiveScanWLT16(uint gtid, uint waveSize)
{
    const uint globalHistOffset = GlobalHistOffset();
    if (gtid < waveSize)
    {
        const uint circularLaneShift = WaveGetLaneIndex() + 1 &
            waveSize - 1;
        InterlockedAdd(b_globalHist[circularLaneShift + globalHistOffset],
            circularLaneShift ? g_us[gtid] : 0);
    }
    GroupMemoryBarrierWithGroupSync();

    const uint laneLog = countbits(waveSize - 1);
    uint offset = laneLog;
    uint j = waveSize;
    for (; j < (RADIX >> 1); j <<= laneLog)
    {
        if (gtid < (RADIX >> offset))
        {
            g_us[((gtid + 1) << offset) - 1] +=
                WavePrefixSum(g_us[((gtid + 1) << offset) - 1]);
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint i = gtid + j; i < RADIX; i += US_DIM)
        {
            if ((i & ((j << laneLog) - 1)) >= j)
            {
                if (i < (j << laneLog))
                {
                    InterlockedAdd(b_globalHist[i + globalHistOffset],
                        WaveReadLaneAt(g_us[((i >> offset) << offset) - 1], 0) +
                        ((i & (j - 1)) ? g_us[i - 1] : 0));
                }
                else
                {
                    if ((i + 1) & (j - 1))
                    {
                        g_us[i] +=
                            WaveReadLaneAt(g_us[((i >> offset) << offset) - 1], 0);
                    }
                }
            }
        }
        offset += laneLog;
    }
    GroupMemoryBarrierWithGroupSync();

    //If RADIX is not a power of lanecount
    for (uint i = gtid + j; i < RADIX; i += US_DIM)
    {
        InterlockedAdd(b_globalHist[i + globalHistOffset],
            WaveReadLaneAt(g_us[((i >> offset) << offset) - 1], 0) +
            ((i & (j - 1)) ? g_us[i - 1] : 0));
    }
}

[numthreads(US_DIM, 1, 1)]
void Upsweep(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    //get the wave size
    const uint waveSize = getWaveSize();

    //clear shared memory
    const uint histsEnd = RADIX * 2;
    for (uint i = gtid.x; i < histsEnd; i += US_DIM)
        g_us[i] = 0;
    GroupMemoryBarrierWithGroupSync();

    HistogramDigitCounts(gtid.x, gid.x);
    GroupMemoryBarrierWithGroupSync();

    ReduceWriteDigitCounts(gtid.x, gid.x);

    if (waveSize >= 16)
        GlobalHistExclusiveScanWGE16(gtid.x, waveSize);

    if (waveSize < 16)
        GlobalHistExclusiveScanWLT16(gtid.x, waveSize);
}

//*****************************************************************************
//SCAN KERNEL
//*****************************************************************************
inline void ExclusiveThreadBlockScanFullWGE16(
    uint gtid,
    uint laneMask,
    uint circularLaneShift,
    uint partEnd,
    uint deviceOffset,
    uint waveSize,
    inout uint reduction)
{
    for (uint i = gtid; i < partEnd; i += SCAN_DIM)
    {
        g_scan[gtid] = b_passHist[i + deviceOffset];
        g_scan[gtid] += WavePrefixSum(g_scan[gtid]);
        GroupMemoryBarrierWithGroupSync();

        if (gtid < SCAN_DIM / waveSize)
        {
            g_scan[(gtid + 1) * waveSize - 1] +=
                WavePrefixSum(g_scan[(gtid + 1) * waveSize - 1]);
        }
        GroupMemoryBarrierWithGroupSync();

        uint t = (WaveGetLaneIndex() != laneMask ? g_scan[gtid] : 0) + reduction;
        if (gtid >= waveSize)
            t += WaveReadLaneAt(g_scan[gtid - 1], 0);
        b_passHist[circularLaneShift + (i & ~laneMask) + deviceOffset] = t;

        reduction += g_scan[SCAN_DIM - 1];
        GroupMemoryBarrierWithGroupSync();
    }
}

inline void ExclusiveThreadBlockScanPartialWGE16(
    uint gtid,
    uint laneMask,
    uint circularLaneShift,
    uint partEnd,
    uint deviceOffset,
    uint waveSize,
    uint reduction)
{
    uint i = gtid + partEnd;
    if (i < e_threadBlocks)
        g_scan[gtid] = b_passHist[deviceOffset + i];
    g_scan[gtid] += WavePrefixSum(g_scan[gtid]);
    GroupMemoryBarrierWithGroupSync();

    if (gtid < SCAN_DIM / waveSize)
    {
        g_scan[(gtid + 1) * waveSize - 1] +=
            WavePrefixSum(g_scan[(gtid + 1) * waveSize - 1]);
    }
    GroupMemoryBarrierWithGroupSync();

    const uint index = circularLaneShift + (i & ~laneMask);
    if (index < e_threadBlocks)
    {
        uint t = (WaveGetLaneIndex() != laneMask ? g_scan[gtid] : 0) + reduction;
        if (gtid >= waveSize)
            t += g_scan[(gtid & ~laneMask) - 1];
        b_passHist[index + deviceOffset] = t;
    }
}

inline void ExclusiveThreadBlockScanWGE16(uint gtid, uint gid, uint waveSize)
{
    uint reduction = 0;
    const uint laneMask = waveSize - 1;
    const uint circularLaneShift = WaveGetLaneIndex() + 1 & laneMask;
    const uint partionsEnd = e_threadBlocks / SCAN_DIM * SCAN_DIM;
    const uint deviceOffset = gid * e_threadBlocks;

    ExclusiveThreadBlockScanFullWGE16(
        gtid,
        laneMask,
        circularLaneShift,
        partionsEnd,
        deviceOffset,
        waveSize,
        reduction);

    ExclusiveThreadBlockScanPartialWGE16(
        gtid,
        laneMask,
        circularLaneShift,
        partionsEnd,
        deviceOffset,
        waveSize,
        reduction);
}

inline void ExclusiveThreadBlockScanFullWLT16(
    uint gtid,
    uint partitions,
    uint deviceOffset,
    uint laneLog,
    uint circularLaneShift,
    uint waveSize,
    inout uint reduction)
{
    for (uint k = 0; k < partitions; ++k)
    {
        g_scan[gtid] = b_passHist[gtid + k * SCAN_DIM + deviceOffset];
        g_scan[gtid] += WavePrefixSum(g_scan[gtid]);
        GroupMemoryBarrierWithGroupSync();
        if (gtid < waveSize)
        {
            b_passHist[circularLaneShift + k * SCAN_DIM + deviceOffset] =
                (circularLaneShift ? g_scan[gtid] : 0) + reduction;
        }

        uint offset = laneLog;
        uint j = waveSize;
        for (; j < (SCAN_DIM >> 1); j <<= laneLog)
        {
            if (gtid < (SCAN_DIM >> offset))
            {
                g_scan[((gtid + 1) << offset) - 1] +=
                    WavePrefixSum(g_scan[((gtid + 1) << offset) - 1]);
            }
            GroupMemoryBarrierWithGroupSync();

            if ((gtid & ((j << laneLog) - 1)) >= j)
            {
                if (gtid < (j << laneLog))
                {
                    b_passHist[gtid + k * SCAN_DIM + deviceOffset] =
                        WaveReadLaneAt(g_scan[((gtid >> offset) << offset) - 1], 0) +
                        ((gtid & (j - 1)) ? g_scan[gtid - 1] : 0) + reduction;
                }
                else
                {
                    if ((gtid + 1) & (j - 1))
                    {
                        g_scan[gtid] +=
                            WaveReadLaneAt(g_scan[((gtid >> offset) << offset) - 1], 0);
                    }
                }
            }
            offset += laneLog;
        }
        GroupMemoryBarrierWithGroupSync();

        //If SCAN_DIM is not a power of lanecount
        for (uint i = gtid + j; i < SCAN_DIM; i += SCAN_DIM)
        {
            b_passHist[i + k * SCAN_DIM + deviceOffset] =
                WaveReadLaneAt(g_scan[((i >> offset) << offset) - 1], 0) +
                ((i & (j - 1)) ? g_scan[i - 1] : 0) + reduction;
        }

        reduction += WaveReadLaneAt(g_scan[SCAN_DIM - 1], 0) +
            WaveReadLaneAt(g_scan[(((SCAN_DIM - 1) >> offset) << offset) - 1], 0);
        GroupMemoryBarrierWithGroupSync();
    }
}

inline void ExclusiveThreadBlockScanParitalWLT16(
    uint gtid,
    uint partitions,
    uint deviceOffset,
    uint laneLog,
    uint circularLaneShift,
    uint waveSize,
    uint reduction)
{
    const uint finalPartSize = e_threadBlocks - partitions * SCAN_DIM;
    if (gtid < finalPartSize)
    {
        g_scan[gtid] = b_passHist[gtid + partitions * SCAN_DIM + deviceOffset];
        g_scan[gtid] += WavePrefixSum(g_scan[gtid]);
    }
    GroupMemoryBarrierWithGroupSync();
    if (gtid < waveSize && circularLaneShift < finalPartSize)
    {
        b_passHist[circularLaneShift + partitions * SCAN_DIM + deviceOffset] =
            (circularLaneShift ? g_scan[gtid] : 0) + reduction;
    }

    uint offset = laneLog;
    for (uint j = waveSize; j < finalPartSize; j <<= laneLog)
    {
        if (gtid < (finalPartSize >> offset))
        {
            g_scan[((gtid + 1) << offset) - 1] +=
                WavePrefixSum(g_scan[((gtid + 1) << offset) - 1]);
        }
        GroupMemoryBarrierWithGroupSync();

        if ((gtid & ((j << laneLog) - 1)) >= j && gtid < finalPartSize)
        {
            if (gtid < (j << laneLog))
            {
                b_passHist[gtid + partitions * SCAN_DIM + deviceOffset] =
                    WaveReadLaneAt(g_scan[((gtid >> offset) << offset) - 1], 0) +
                    ((gtid & (j - 1)) ? g_scan[gtid - 1] : 0) + reduction;
            }
            else
            {
                if ((gtid + 1) & (j - 1))
                {
                    g_scan[gtid] +=
                        WaveReadLaneAt(g_scan[((gtid >> offset) << offset) - 1], 0);
                }
            }
        }
        offset += laneLog;
    }
}

inline void ExclusiveThreadBlockScanWLT16(uint gtid, uint gid, uint waveSize)
{
    uint reduction = 0;
    const uint partitions = e_threadBlocks / SCAN_DIM;
    const uint deviceOffset = gid * e_threadBlocks;
    const uint laneLog = countbits(waveSize - 1);
    const uint circularLaneShift = WaveGetLaneIndex() + 1 & waveSize - 1;

    ExclusiveThreadBlockScanFullWLT16(
        gtid,
        partitions,
        deviceOffset,
        laneLog,
        circularLaneShift,
        waveSize,
        reduction);

    ExclusiveThreadBlockScanParitalWLT16(
        gtid,
        partitions,
        deviceOffset,
        laneLog,
        circularLaneShift,
        waveSize,
        reduction);
}

//Scan along the spine of the upsweep
[numthreads(SCAN_DIM, 1, 1)]
void Scan(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint waveSize = getWaveSize();

    if (waveSize >= 16)
        ExclusiveThreadBlockScanWGE16(gtid.x, gid.x, waveSize);

    if (waveSize < 16)
        ExclusiveThreadBlockScanWLT16(gtid.x, gid.x, waveSize);
}

//*****************************************************************************
//DOWNSWEEP KERNEL
//*****************************************************************************
//If the size of a wave is too small, we do not have enough space in
//shared memory to assign a histogram to each wave, so instead,
//some operations are performed serially.
inline uint SerialIterations(uint waveSize)
{
    return (DS_DIM / waveSize + 31) >> 5;
}

inline void ClearWaveHists(uint gtid, uint waveSize)
{
    const uint histsEnd = waveSize >= 16 ?
        WaveHistsSizeWGE16(waveSize) : WaveHistsSizeWLT16();
    for (uint i = gtid; i < histsEnd; i += DS_DIM)
        g_ds[i] = 0;
}

inline KeyStruct LoadKeysWGE16(uint gtid, uint waveSize, uint partIndex)
{
    KeyStruct keys;
    [unroll]
    for (uint i = 0, t = DeviceOffsetWGE16(gtid, waveSize, partIndex);
        i < DS_KEYS_PER_THREAD;
        ++i, t += waveSize)
    {
        keys.k[i] = b_sort[t];
    }
    return keys;
}

inline KeyStruct LoadKeysWLT16(uint gtid, uint waveSize, uint partIndex, uint serialIterations)
{
    KeyStruct keys;
    [unroll]
    for (uint i = 0, t = DeviceOffsetWLT16(gtid, waveSize, partIndex, serialIterations);
        i < DS_KEYS_PER_THREAD;
        ++i, t += waveSize * serialIterations)
    {
        keys.k[i] = b_sort[t];
    }
    return keys;
}

inline KeyStruct LoadKeysPartialWGE16(uint gtid, uint waveSize, uint partIndex)
{
    KeyStruct keys;
    [unroll]
    for (uint i = 0, t = DeviceOffsetWGE16(gtid, waveSize, partIndex);
        i < DS_KEYS_PER_THREAD;
        ++i, t += waveSize)
    {
        keys.k[i] = t < e_numKeys ? b_sort[t] : 0xffffffff;
    }
    return keys;
}

inline KeyStruct LoadKeysPartialWLT16(uint gtid, uint waveSize, uint partIndex, uint serialIterations)
{
    KeyStruct keys;
    [unroll]
    for (uint i = 0, t = DeviceOffsetWLT16(gtid, waveSize, partIndex, serialIterations);
        i < DS_KEYS_PER_THREAD;
        ++i, t += waveSize * serialIterations)
    {
        keys.k[i] = t < e_numKeys ? b_sort[t] : 0xffffffff;
    }
    return keys;
}

inline uint WaveFlagsWGE16(uint waveSize)
{
    return (waveSize & 31) ? (1U << waveSize) - 1 : 0xffffffff;
}

inline uint WaveFlagsWLT16(uint waveSize)
{
    return (1U << waveSize) - 1;
}

inline void WarpLevelMultiSplitWGE16(uint key, inout uint4 waveFlags)
{
    [unroll]
    for (uint k = 0; k < RADIX_LOG; ++k)
    {
        const uint currentBit = 1u << (k + e_radixShift);
        const bool t = (key & currentBit) != 0;
        GroupMemoryBarrierWithGroupSync();  //play on the safe side, throw in a barrier for convergence
        const uint4 ballot = WaveActiveBallot(t);
        if (t)
            waveFlags &= ballot;
        else
            waveFlags &= (~ballot);
    }
}

//Returns the number of preceding peers in .x and the total peer count in .y. The countbits() call is
//hoisted out of the divergent branch on purpose: Adreno miscompiles it when it is buried inside a
//conditional that also tests WaveGetLaneIndex().
inline uint2 CountBitsWGE16(uint waveSize, uint ltMask, uint4 waveFlags)
{
    uint2 count = uint2(0, 0);

    for (uint wavePart = 0; wavePart < waveSize; wavePart += 32)
    {
        uint t = countbits(waveFlags[wavePart >> 5]);
        if (WaveGetLaneIndex() >= wavePart)
        {
            if (WaveGetLaneIndex() >= wavePart + 32)
                count.x += t;
            else
                count.x += countbits(waveFlags[wavePart >> 5] & ltMask);
        }
        count.y += t;
    }

    return count;
}

inline void WarpLevelMultiSplitWLT16(uint key, inout uint waveFlags)
{
    [unroll]
    for (uint k = 0; k < RADIX_LOG; ++k)
    {
        const bool t = key >> (k + e_radixShift) & 1;
        waveFlags &= (t ? 0 : 0xffffffff) ^ (uint) WaveActiveBallot(t);
    }
}

inline OffsetStruct RankKeysWGE16(uint waveSize, uint waveOffset, KeyStruct keys)
{
    OffsetStruct offsets;
    const uint initialFlags = WaveFlagsWGE16(waveSize);
    const uint ltMask = (1U << (WaveGetLaneIndex() & 31)) - 1;

    [unroll]
    for (uint i = 0; i < DS_KEYS_PER_THREAD; ++i)
    {
        uint4 waveFlags = initialFlags;
        WarpLevelMultiSplitWGE16(keys.k[i], waveFlags);

        const uint index = ExtractDigit(keys.k[i]) + waveOffset;
        const uint2 bitCount = CountBitsWGE16(waveSize, ltMask, waveFlags);

        offsets.o[i] = g_ds[index] + bitCount.x;
        GroupMemoryBarrierWithGroupSync();
        if (bitCount.x == 0)
            g_ds[index] += bitCount.y;
        GroupMemoryBarrierWithGroupSync();
    }

    return offsets;
}

inline OffsetStruct RankKeysWLT16(uint waveSize, uint waveIndex, KeyStruct keys, uint serialIterations)
{
    OffsetStruct offsets;
    const uint ltMask = (1U << WaveGetLaneIndex()) - 1;
    const uint initialFlags = WaveFlagsWLT16(waveSize);

    [unroll]
    for (uint i = 0; i < DS_KEYS_PER_THREAD; ++i)
    {
        uint waveFlags = initialFlags;
        WarpLevelMultiSplitWLT16(keys.k[i], waveFlags);

        const uint index = ExtractPackedIndex(keys.k[i]) +
            (waveIndex / serialIterations * HALF_RADIX);

        const uint peerBits = countbits(waveFlags & ltMask);
        for (uint k = 0; k < serialIterations; ++k)
        {
            if (waveIndex % serialIterations == k)
                offsets.o[i] = ExtractPackedValue(g_ds[index], keys.k[i]) + peerBits;

            GroupMemoryBarrierWithGroupSync();
            if (waveIndex % serialIterations == k && peerBits == 0)
            {
                InterlockedAdd(g_ds[index],
                    countbits(waveFlags) << ExtractPackedShift(keys.k[i]));
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    return offsets;
}

inline uint WaveHistInclusiveScanCircularShiftWGE16(uint gtid, uint waveSize)
{
    uint histReduction = g_ds[gtid];
    for (uint i = gtid + RADIX; i < WaveHistsSizeWGE16(waveSize); i += RADIX)
    {
        histReduction += g_ds[i];
        g_ds[i] = histReduction - g_ds[i];
    }
    return histReduction;
}

inline uint WaveHistInclusiveScanCircularShiftWLT16(uint gtid)
{
    uint histReduction = g_ds[gtid];
    for (uint i = gtid + HALF_RADIX; i < WaveHistsSizeWLT16(); i += HALF_RADIX)
    {
        histReduction += g_ds[i];
        g_ds[i] = histReduction - g_ds[i];
    }
    return histReduction;
}

inline void WaveHistReductionExclusiveScanWGE16(uint gtid, uint waveSize, uint histReduction)
{
    if (gtid < RADIX)
    {
        const uint laneMask = waveSize - 1;
        g_ds[((WaveGetLaneIndex() + 1) & laneMask) + (gtid & ~laneMask)] = histReduction;
    }
    GroupMemoryBarrierWithGroupSync();

    if (gtid < RADIX / waveSize)
    {
        g_ds[gtid * waveSize] =
            WavePrefixSum(g_ds[gtid * waveSize]);
    }
    GroupMemoryBarrierWithGroupSync();

    uint t = WaveReadLaneAt(g_ds[gtid], 0);
    if (gtid < RADIX && WaveGetLaneIndex())
        g_ds[gtid] += t;
}

//inclusive/exclusive prefix sum up the histograms,
//use a blelloch scan for in place packed exclusive
inline void WaveHistReductionExclusiveScanWLT16(uint gtid)
{
    uint shift = 1;
    for (uint j = RADIX >> 2; j > 0; j >>= 1)
    {
        GroupMemoryBarrierWithGroupSync();
        if (gtid < j)
        {
            g_ds[((((gtid << 1) + 2) << shift) - 1) >> 1] +=
                g_ds[((((gtid << 1) + 1) << shift) - 1) >> 1] & 0xffff0000;
        }
        shift++;
    }
    GroupMemoryBarrierWithGroupSync();

    if (gtid == 0)
        g_ds[HALF_RADIX - 1] &= 0xffff;

    for (uint j2 = 1; j2 < RADIX >> 1; j2 <<= 1)
    {
        --shift;
        GroupMemoryBarrierWithGroupSync();
        if (gtid < j2)
        {
            const uint t = ((((gtid << 1) + 1) << shift) - 1) >> 1;
            const uint t2 = ((((gtid << 1) + 2) << shift) - 1) >> 1;
            const uint t3 = g_ds[t];
            g_ds[t] = (g_ds[t] & 0xffff) | (g_ds[t2] & 0xffff0000);
            g_ds[t2] += t3 & 0xffff0000;
        }
    }

    GroupMemoryBarrierWithGroupSync();
    if (gtid < HALF_RADIX)
    {
        const uint t = g_ds[gtid];
        g_ds[gtid] = (t >> 16) + (t << 16) + (t & 0xffff0000);
    }
}

inline void UpdateOffsetsWGE16(uint gtid, uint waveSize, inout OffsetStruct offsets, KeyStruct keys)
{
    if (gtid >= waveSize)
    {
        const uint t = getWaveIndex(gtid, waveSize) * RADIX;
        [unroll]
        for (uint i = 0; i < DS_KEYS_PER_THREAD; ++i)
        {
            const uint t2 = ExtractDigit(keys.k[i]);
            offsets.o[i] += g_ds[t2 + t] + g_ds[t2];
        }
    }
    else
    {
        [unroll]
        for (uint i = 0; i < DS_KEYS_PER_THREAD; ++i)
            offsets.o[i] += g_ds[ExtractDigit(keys.k[i])];
    }
}

inline void UpdateOffsetsWLT16(
    uint gtid,
    uint waveSize,
    uint serialIterations,
    inout OffsetStruct offsets,
    KeyStruct keys)
{
    if (gtid >= waveSize * serialIterations)
    {
        const uint t = getWaveIndex(gtid, waveSize) / serialIterations * HALF_RADIX;
        [unroll]
        for (uint i = 0; i < DS_KEYS_PER_THREAD; ++i)
        {
            const uint t2 = ExtractPackedIndex(keys.k[i]);
            offsets.o[i] += ExtractPackedValue(g_ds[t2 + t] + g_ds[t2], keys.k[i]);
        }
    }
    else
    {
        [unroll]
        for (uint i = 0; i < DS_KEYS_PER_THREAD; ++i)
            offsets.o[i] += ExtractPackedValue(g_ds[ExtractPackedIndex(keys.k[i])], keys.k[i]);
    }
}

inline void ScatterKeysShared(OffsetStruct offsets, KeyStruct keys)
{
    [unroll]
    for (uint i = 0; i < DS_KEYS_PER_THREAD; ++i)
        g_ds[offsets.o[i]] = keys.k[i];
}

inline void LoadThreadBlockReductions(uint gtid, uint gid, uint exclusiveHistReduction)
{
    if (gtid < RADIX)
    {
        g_ds[gtid + PART_SIZE] = b_globalHist[gtid + GlobalHistOffset()] +
            b_passHist[gtid * e_threadBlocks + gid] - exclusiveHistReduction;
    }
}

inline void LoadPayloadsWGE16(uint gtid, uint waveSize, uint partIndex, inout KeyStruct payloads)
{
    [unroll]
    for (uint i = 0, t = DeviceOffsetWGE16(gtid, waveSize, partIndex);
        i < DS_KEYS_PER_THREAD;
        ++i, t += waveSize)
    {
        payloads.k[i] = b_sortPayload[t];
    }
}

inline void LoadPayloadsWLT16(
    uint gtid,
    uint waveSize,
    uint partIndex,
    uint serialIterations,
    inout KeyStruct payloads)
{
    [unroll]
    for (uint i = 0, t = DeviceOffsetWLT16(gtid, waveSize, partIndex, serialIterations);
        i < DS_KEYS_PER_THREAD;
        ++i, t += waveSize * serialIterations)
    {
        payloads.k[i] = b_sortPayload[t];
    }
}

inline void LoadPayloadsPartialWGE16(uint gtid, uint waveSize, uint partIndex, inout KeyStruct payloads)
{
    [unroll]
    for (uint i = 0, t = DeviceOffsetWGE16(gtid, waveSize, partIndex);
        i < DS_KEYS_PER_THREAD;
        ++i, t += waveSize)
    {
        if (t < e_numKeys)
            payloads.k[i] = b_sortPayload[t];
    }
}

inline void LoadPayloadsPartialWLT16(
    uint gtid,
    uint waveSize,
    uint partIndex,
    uint serialIterations,
    inout KeyStruct payloads)
{
    [unroll]
    for (uint i = 0, t = DeviceOffsetWLT16(gtid, waveSize, partIndex, serialIterations);
        i < DS_KEYS_PER_THREAD;
        ++i, t += waveSize * serialIterations)
    {
        if (t < e_numKeys)
            payloads.k[i] = b_sortPayload[t];
    }
}

//*****************************************************************************
//SCATTERING
//*****************************************************************************
inline void ScatterDevice(uint gtid, uint waveSize, uint partIndex, OffsetStruct offsets)
{
    //Scatter the keys, caching the digit so the histogram lookup survives the payload pass
    //overwriting the key half of shared memory.
    DigitStruct digits;
    [unroll]
    for (uint i = 0, t = gtid; i < DS_KEYS_PER_THREAD; ++i, t += DS_DIM)
    {
        digits.d[i] = ExtractDigit(g_ds[t]);
        b_alt[g_ds[digits.d[i] + PART_SIZE] + t] = g_ds[t];
    }
    GroupMemoryBarrierWithGroupSync();

    KeyStruct payloads;
    if (waveSize >= 16)
        LoadPayloadsWGE16(gtid, waveSize, partIndex, payloads);

    if (waveSize < 16)
        LoadPayloadsWLT16(gtid, waveSize, partIndex, SerialIterations(waveSize), payloads);

    ScatterKeysShared(offsets, payloads);
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint p = 0, u = gtid; p < DS_KEYS_PER_THREAD; ++p, u += DS_DIM)
        b_altPayload[g_ds[digits.d[p] + PART_SIZE] + u] = g_ds[u];
}

inline void ScatterDevicePartial(uint gtid, uint waveSize, uint partIndex, OffsetStruct offsets)
{
    DigitStruct digits;
    const uint finalPartSize = e_numKeys - partIndex * PART_SIZE;
    [unroll]
    for (uint i = 0, t = gtid; i < DS_KEYS_PER_THREAD; ++i, t += DS_DIM)
    {
        if (t < finalPartSize)
        {
            digits.d[i] = ExtractDigit(g_ds[t]);
            b_alt[g_ds[digits.d[i] + PART_SIZE] + t] = g_ds[t];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    KeyStruct payloads;
    if (waveSize >= 16)
        LoadPayloadsPartialWGE16(gtid, waveSize, partIndex, payloads);

    if (waveSize < 16)
        LoadPayloadsPartialWLT16(gtid, waveSize, partIndex, SerialIterations(waveSize), payloads);

    ScatterKeysShared(offsets, payloads);
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint p = 0, u = gtid; p < DS_KEYS_PER_THREAD; ++p, u += DS_DIM)
    {
        if (u < finalPartSize)
            b_altPayload[g_ds[digits.d[p] + PART_SIZE] + u] = g_ds[u];
    }
}

[numthreads(DS_DIM, 1, 1)]
void Downsweep(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    KeyStruct keys;
    OffsetStruct offsets;
    const uint waveSize = getWaveSize();

    ClearWaveHists(gtid.x, waveSize);
    GroupMemoryBarrierWithGroupSync();

    if (gid.x < e_threadBlocks - 1)
    {
        if (waveSize >= 16)
            keys = LoadKeysWGE16(gtid.x, waveSize, gid.x);

        if (waveSize < 16)
            keys = LoadKeysWLT16(gtid.x, waveSize, gid.x, SerialIterations(waveSize));
    }

    if (gid.x == e_threadBlocks - 1)
    {
        if (waveSize >= 16)
            keys = LoadKeysPartialWGE16(gtid.x, waveSize, gid.x);

        if (waveSize < 16)
            keys = LoadKeysPartialWLT16(gtid.x, waveSize, gid.x, SerialIterations(waveSize));
    }

    uint exclusiveHistReduction = 0;

    if (waveSize >= 16)
    {
        offsets = RankKeysWGE16(waveSize, getWaveIndex(gtid.x, waveSize) * RADIX, keys);
        GroupMemoryBarrierWithGroupSync();

        uint histReduction = 0;
        if (gtid.x < RADIX)
        {
            histReduction = WaveHistInclusiveScanCircularShiftWGE16(gtid.x, waveSize);
            histReduction += WavePrefixSum(histReduction); //take advantage of barrier to begin scan
        }
        GroupMemoryBarrierWithGroupSync();

        WaveHistReductionExclusiveScanWGE16(gtid.x, waveSize, histReduction);
        GroupMemoryBarrierWithGroupSync();

        UpdateOffsetsWGE16(gtid.x, waveSize, offsets, keys);
        if (gtid.x < RADIX)
            exclusiveHistReduction = g_ds[gtid.x]; //take advantage of barrier to grab value
        GroupMemoryBarrierWithGroupSync();
    }

    if (waveSize < 16)
    {
        offsets = RankKeysWLT16(waveSize, getWaveIndex(gtid.x, waveSize), keys, SerialIterations(waveSize));

        if (gtid.x < HALF_RADIX)
        {
            uint histReduction = WaveHistInclusiveScanCircularShiftWLT16(gtid.x);
            g_ds[gtid.x] = histReduction + (histReduction << 16); //take advantage of barrier to begin scan
        }

        WaveHistReductionExclusiveScanWLT16(gtid.x);
        GroupMemoryBarrierWithGroupSync();

        UpdateOffsetsWLT16(gtid.x, waveSize, SerialIterations(waveSize), offsets, keys);
        if (gtid.x < RADIX) //take advantage of barrier to grab value
            exclusiveHistReduction = g_ds[gtid.x >> 1] >> ((gtid.x & 1) ? 16 : 0) & 0xffff;
        GroupMemoryBarrierWithGroupSync();
    }

    ScatterKeysShared(offsets, keys);
    LoadThreadBlockReductions(gtid.x, gid.x, exclusiveHistReduction);
    GroupMemoryBarrierWithGroupSync();

    if (gid.x < e_threadBlocks - 1)
        ScatterDevice(gtid.x, waveSize, gid.x, offsets);

    if (gid.x == e_threadBlocks - 1)
        ScatterDevicePartial(gtid.x, waveSize, gid.x, offsets);
}
