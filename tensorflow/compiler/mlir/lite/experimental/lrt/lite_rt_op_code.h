// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_LITE_RT_OP_CODE_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_LITE_RT_OP_CODE_H_

#include "tensorflow/lite/builtin_ops.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum {
  kLrtOpCodeTflAdd = kTfLiteBuiltinAdd,
  kLrtOpCodeTflAveragePool2d = kTfLiteBuiltinAveragePool2d,
  kLrtOpCodeTflConcatenation = kTfLiteBuiltinConcatenation,
  kLrtOpCodeTflConv2d = kTfLiteBuiltinConv2d,
  kLrtOpCodeTflDepthwiseConv2d = kTfLiteBuiltinDepthwiseConv2d,
  kLrtOpCodeTflDepthToSpace = kTfLiteBuiltinDepthToSpace,
  kLrtOpCodeTflDequantize = kTfLiteBuiltinDequantize,
  kLrtOpCodeTflEmbeddingLookup = kTfLiteBuiltinEmbeddingLookup,
  kLrtOpCodeTflFloor = kTfLiteBuiltinFloor,
  kLrtOpCodeTflFullyConnected = kTfLiteBuiltinFullyConnected,
  kLrtOpCodeTflHashtableLookup = kTfLiteBuiltinHashtableLookup,
  kLrtOpCodeTflL2Normalization = kTfLiteBuiltinL2Normalization,
  kLrtOpCodeTflL2Pool2d = kTfLiteBuiltinL2Pool2d,
  kLrtOpCodeTflLocalResponseNormalization =
      kTfLiteBuiltinLocalResponseNormalization,
  kLrtOpCodeTflLogistic = kTfLiteBuiltinLogistic,
  kLrtOpCodeTflLshProjection = kTfLiteBuiltinLshProjection,
  kLrtOpCodeTflLstm = kTfLiteBuiltinLstm,
  kLrtOpCodeTflMaxPool2d = kTfLiteBuiltinMaxPool2d,
  kLrtOpCodeTflMul = kTfLiteBuiltinMul,
  kLrtOpCodeTflRelu = kTfLiteBuiltinRelu,
  kLrtOpCodeTflReluN1To1 = kTfLiteBuiltinReluN1To1,
  kLrtOpCodeTflRelu6 = kTfLiteBuiltinRelu6,
  kLrtOpCodeTflReshape = kTfLiteBuiltinReshape,
  kLrtOpCodeTflResizeBilinear = kTfLiteBuiltinResizeBilinear,
  kLrtOpCodeTflRnn = kTfLiteBuiltinRnn,
  kLrtOpCodeTflSoftmax = kTfLiteBuiltinSoftmax,
  kLrtOpCodeTflSpaceToDepth = kTfLiteBuiltinSpaceToDepth,
  kLrtOpCodeTflSvdf = kTfLiteBuiltinSvdf,
  kLrtOpCodeTflTanh = kTfLiteBuiltinTanh,
  kLrtOpCodeTflConcatEmbeddings = kTfLiteBuiltinConcatEmbeddings,
  kLrtOpCodeTflSkipGram = kTfLiteBuiltinSkipGram,
  kLrtOpCodeTflCall = kTfLiteBuiltinCall,
  kLrtOpCodeTflCustom = kTfLiteBuiltinCustom,
  kLrtOpCodeTflEmbeddingLookupSparse = kTfLiteBuiltinEmbeddingLookupSparse,
  kLrtOpCodeTflPad = kTfLiteBuiltinPad,
  kLrtOpCodeTflUnidirectionalSequenceRnn =
      kTfLiteBuiltinUnidirectionalSequenceRnn,
  kLrtOpCodeTflGather = kTfLiteBuiltinGather,
  kLrtOpCodeTflBatchToSpaceNd = kTfLiteBuiltinBatchToSpaceNd,
  kLrtOpCodeTflSpaceToBatchNd = kTfLiteBuiltinSpaceToBatchNd,
  kLrtOpCodeTflTranspose = kTfLiteBuiltinTranspose,
  kLrtOpCodeTflMean = kTfLiteBuiltinMean,
  kLrtOpCodeTflSuv = kTfLiteBuiltinSub,
  kLrtOpCodeTflDiv = kTfLiteBuiltinDiv,
  kLrtOpCodeTflSqueeze = kTfLiteBuiltinSqueeze,
  kLrtOpCodeTflUnidirectionalSequenceLstm =
      kTfLiteBuiltinUnidirectionalSequenceLstm,
  kLrtOpCodeTflStridedSlice = kTfLiteBuiltinStridedSlice,
  kLrtOpCodeTflBidirectionalSequenceRnn =
      kTfLiteBuiltinBidirectionalSequenceRnn,
  kLrtOpCodeTflExp = kTfLiteBuiltinExp,
  kLrtOpCodeTflTopkV2 = kTfLiteBuiltinTopkV2,
  kLrtOpCodeTflSplit = kTfLiteBuiltinSplit,
  kLrtOpCodeTflLogSoftmax = kTfLiteBuiltinLogSoftmax,
  kLrtOpCodeTflDelegate = kTfLiteBuiltinDelegate,
  kLrtOpCodeTflBidirectionalSequenceLstm =
      kTfLiteBuiltinBidirectionalSequenceLstm,
  kLrtOpCodeTflCast = kTfLiteBuiltinCast,
  kLrtOpCodeTflPrelu = kTfLiteBuiltinPrelu,
  kLrtOpCodeTflMaximum = kTfLiteBuiltinMaximum,
  kLrtOpCodeTflArgMax = kTfLiteBuiltinArgMax,
  kLrtOpCodeTflMinimum = kTfLiteBuiltinMinimum,
  kLrtOpCodeTflLess = kTfLiteBuiltinLess,
  kLrtOpCodeTflNeg = kTfLiteBuiltinNeg,
  kLrtOpCodeTflPadv2 = kTfLiteBuiltinPadv2,
  kLrtOpCodeTflGreater = kTfLiteBuiltinGreater,
  kLrtOpCodeTflGreaterEqual = kTfLiteBuiltinGreaterEqual,
  kLrtOpCodeTflLessEqual = kTfLiteBuiltinLessEqual,
  kLrtOpCodeTflSelect = kTfLiteBuiltinSelect,
  kLrtOpCodeTflSlice = kTfLiteBuiltinSlice,
  kLrtOpCodeTflSin = kTfLiteBuiltinSin,
  kLrtOpCodeTflTransposeConv = kTfLiteBuiltinTransposeConv,
  kLrtOpCodeTflSparseToDense = kTfLiteBuiltinSparseToDense,
  kLrtOpCodeTflTile = kTfLiteBuiltinTile,
  kLrtOpCodeTflExpandDims = kTfLiteBuiltinExpandDims,
  kLrtOpCodeTflEqual = kTfLiteBuiltinEqual,
  kLrtOpCodeTflNotEqual = kTfLiteBuiltinNotEqual,
  kLrtOpCodeTflLog = kTfLiteBuiltinLog,
  kLrtOpCodeTflSum = kTfLiteBuiltinSum,
  kLrtOpCodeTflSqrt = kTfLiteBuiltinSqrt,
  kLrtOpCodeTflRsqrt = kTfLiteBuiltinRsqrt,
  kLrtOpCodeTflShape = kTfLiteBuiltinShape,
  kLrtOpCodeTflPow = kTfLiteBuiltinPow,
  kLrtOpCodeTflArgMin = kTfLiteBuiltinArgMin,
  kLrtOpCodeTflFakeQuant = kTfLiteBuiltinFakeQuant,
  kLrtOpCodeTflReduceProd = kTfLiteBuiltinReduceProd,
  kLrtOpCodeTflReduceMax = kTfLiteBuiltinReduceMax,
  kLrtOpCodeTflPack = kTfLiteBuiltinPack,
  kLrtOpCodeTflLogicalOr = kTfLiteBuiltinLogicalOr,
  kLrtOpCodeTflOneHot = kTfLiteBuiltinOneHot,
  kLrtOpCodeTflLogicalAnd = kTfLiteBuiltinLogicalAnd,
  kLrtOpCodeTflLogicalNot = kTfLiteBuiltinLogicalNot,
  kLrtOpCodeTflUnpack = kTfLiteBuiltinUnpack,
  kLrtOpCodeTflReduceMin = kTfLiteBuiltinReduceMin,
  kLrtOpCodeTflFloorDiv = kTfLiteBuiltinFloorDiv,
  kLrtOpCodeTflReduceAny = kTfLiteBuiltinReduceAny,
  kLrtOpCodeTflSquare = kTfLiteBuiltinSquare,
  kLrtOpCodeTflZerosLike = kTfLiteBuiltinZerosLike,
  kLrtOpCodeTflFill = kTfLiteBuiltinFill,
  kLrtOpCodeTflFloorMod = kTfLiteBuiltinFloorMod,
  kLrtOpCodeTflRange = kTfLiteBuiltinRange,
  kLrtOpCodeTflResizeNearestNeighbor = kTfLiteBuiltinResizeNearestNeighbor,
  kLrtOpCodeTflLeakyRelu = kTfLiteBuiltinLeakyRelu,
  kLrtOpCodeTflSquaredDifference = kTfLiteBuiltinSquaredDifference,
  kLrtOpCodeTflMirrorPad = kTfLiteBuiltinMirrorPad,
  kLrtOpCodeTflAbs = kTfLiteBuiltinAbs,
  kLrtOpCodeTflSplitV = kTfLiteBuiltinSplitV,
  kLrtOpCodeTflUnique = kTfLiteBuiltinUnique,
  kLrtOpCodeTflCeil = kTfLiteBuiltinCeil,
  kLrtOpCodeTflReverseV2 = kTfLiteBuiltinReverseV2,
  kLrtOpCodeTflAddN = kTfLiteBuiltinAddN,
  kLrtOpCodeTflGatherNd = kTfLiteBuiltinGatherNd,
  kLrtOpCodeTflCos = kTfLiteBuiltinCos,
  kLrtOpCodeTflWhere = kTfLiteBuiltinWhere,
  kLrtOpCodeTflRank = kTfLiteBuiltinRank,
  kLrtOpCodeTflElu = kTfLiteBuiltinElu,
  kLrtOpCodeTflReverseSequence = kTfLiteBuiltinReverseSequence,
  kLrtOpCodeTflMatrixDiag = kTfLiteBuiltinMatrixDiag,
  kLrtOpCodeTflQuantize = kTfLiteBuiltinQuantize,
  kLrtOpCodeTflMatrixSetDiag = kTfLiteBuiltinMatrixSetDiag,
  kLrtOpCodeTflRound = kTfLiteBuiltinRound,
  kLrtOpCodeTflHardSwish = kTfLiteBuiltinHardSwish,
  kLrtOpCodeTflIf = kTfLiteBuiltinIf,
  kLrtOpCodeTflWhile = kTfLiteBuiltinWhile,
  kLrtOpCodeTflNonMaxSuppressionV4 = kTfLiteBuiltinNonMaxSuppressionV4,
  kLrtOpCodeTflNonMaxSuppressionV5 = kTfLiteBuiltinNonMaxSuppressionV5,
  kLrtOpCodeTflScatterNd = kTfLiteBuiltinScatterNd,
  kLrtOpCodeTflSelectV2 = kTfLiteBuiltinSelectV2,
  kLrtOpCodeTflDensify = kTfLiteBuiltinDensify,
  kLrtOpCodeTflSegmentSum = kTfLiteBuiltinSegmentSum,
  kLrtOpCodeTflBatchMatmul = kTfLiteBuiltinBatchMatmul,
  kLrtOpCodeTflPlaceholderForGreaterOpCodeTfls =
      kTfLiteBuiltinPlaceholderForGreaterOpCodes,
  kLrtOpCodeTflCumsum = kTfLiteBuiltinCumsum,
  kLrtOpCodeTflCallOnce = kTfLiteBuiltinCallOnce,
  kLrtOpCodeTflBroadcastTo = kTfLiteBuiltinBroadcastTo,
  kLrtOpCodeTflRfft2d = kTfLiteBuiltinRfft2d,
  kLrtOpCodeTflConv3d = kTfLiteBuiltinConv3d,
  kLrtOpCodeTflImag = kTfLiteBuiltinImag,
  kLrtOpCodeTflReal = kTfLiteBuiltinReal,
  kLrtOpCodeTflComplexAbs = kTfLiteBuiltinComplexAbs,
  kLrtOpCodeTflHashtable = kTfLiteBuiltinHashtable,
  kLrtOpCodeTflHashtableFind = kTfLiteBuiltinHashtableFind,
  kLrtOpCodeTflHashtableImport = kTfLiteBuiltinHashtableImport,
  kLrtOpCodeTflHashtableSize = kTfLiteBuiltinHashtableSize,
  kLrtOpCodeTflReduceAll = kTfLiteBuiltinReduceAll,
  kLrtOpCodeTflConv3dTranspose = kTfLiteBuiltinConv3dTranspose,
  kLrtOpCodeTflVarHandle = kTfLiteBuiltinVarHandle,
  kLrtOpCodeTflReadVariable = kTfLiteBuiltinReadVariable,
  kLrtOpCodeTflAssignVariable = kTfLiteBuiltinAssignVariable,
  kLrtOpCodeTflBroadcastArgs = kTfLiteBuiltinBroadcastArgs,
  kLrtOpCodeTflRandomStandardNormal = kTfLiteBuiltinRandomStandardNormal,
  kLrtOpCodeTflBucketize = kTfLiteBuiltinBucketize,
  kLrtOpCodeTflRandomUniform = kTfLiteBuiltinRandomUniform,
  kLrtOpCodeTflMultinomial = kTfLiteBuiltinMultinomial,
  kLrtOpCodeTflGelu = kTfLiteBuiltinGelu,
  kLrtOpCodeTflDynamicUpdateSlice = kTfLiteBuiltinDynamicUpdateSlice,
  kLrtOpCodeTflRelu0To1 = kTfLiteBuiltinRelu0To1,
  kLrtOpCodeTflUnsortedSegmentProd = kTfLiteBuiltinUnsortedSegmentProd,
  kLrtOpCodeTflUnsortedSegmentMax = kTfLiteBuiltinUnsortedSegmentMax,
  kLrtOpCodeTflUnsortedSegmentSum = kTfLiteBuiltinUnsortedSegmentSum,
  kLrtOpCodeTflAtan2 = kTfLiteBuiltinAtan2,
  kLrtOpCodeTflUnsortedSegmentMin = kTfLiteBuiltinUnsortedSegmentMin,
  kLrtOpCodeTflSign = kTfLiteBuiltinSign,
  kLrtOpCodeTflBitcast = kTfLiteBuiltinBitcast,
  kLrtOpCodeTflBitwiseXor = kTfLiteBuiltinBitwiseXor,
  kLrtOpCodeTflRightShift = kTfLiteBuiltinRightShift,
  kLrtOpCodeShloLogistic = kTfLiteBuiltinStablehloLogistic,
  kLrtOpCodeShloAdd = kTfLiteBuiltinStablehloAdd,
  kLrtOpCodeShloDivide = kTfLiteBuiltinStablehloDivide,
  kLrtOpCodeShloMultiply = kTfLiteBuiltinStablehloMultiply,
  kLrtOpCodeShloMaximum = kTfLiteBuiltinStablehloMaximum,
  kLrtOpCodeShloReshape = kTfLiteBuiltinStablehloReshape,
  kLrtOpCodeShloClamp = kTfLiteBuiltinStablehloClamp,
  kLrtOpCodeShloConcatenate = kTfLiteBuiltinStablehloConcatenate,
  kLrtOpCodeShloBroadcastInDim = kTfLiteBuiltinStablehloBroadcastInDim,
  kLrtOpCodeShloConvolution = kTfLiteBuiltinStablehloConvolution,
  kLrtOpCodeShloSlice = kTfLiteBuiltinStablehloSlice,
  kLrtOpCodeShloCustomCall = kTfLiteBuiltinStablehloCustomCall,
  kLrtOpCodeShloReduce = kTfLiteBuiltinStablehloReduce,
  kLrtOpCodeShloAbs = kTfLiteBuiltinStablehloAbs,
  kLrtOpCodeShloAnd = kTfLiteBuiltinStablehloAnd,
  kLrtOpCodeShloCosine = kTfLiteBuiltinStablehloCosine,
  kLrtOpCodeShloExponential = kTfLiteBuiltinStablehloExponential,
  kLrtOpCodeShloFloor = kTfLiteBuiltinStablehloFloor,
  kLrtOpCodeShloLog = kTfLiteBuiltinStablehloLog,
  kLrtOpCodeShloMinimum = kTfLiteBuiltinStablehloMinimum,
  kLrtOpCodeShloNegate = kTfLiteBuiltinStablehloNegate,
  kLrtOpCodeShloOr = kTfLiteBuiltinStablehloOr,
  kLrtOpCodeShloPower = kTfLiteBuiltinStablehloPower,
  kLrtOpCodeShloRemainder = kTfLiteBuiltinStablehloRemainder,
  kLrtOpCodeShloRsqrt = kTfLiteBuiltinStablehloRsqrt,
  kLrtOpCodeShloSelect = kTfLiteBuiltinStablehloSelect,
  kLrtOpCodeShloSubtract = kTfLiteBuiltinStablehloSubtract,
  kLrtOpCodeShloTanh = kTfLiteBuiltinStablehloTanh,
  kLrtOpCodeShloScatter = kTfLiteBuiltinStablehloScatter,
  kLrtOpCodeShloCompare = kTfLiteBuiltinStablehloCompare,
  kLrtOpCodeShloConvert = kTfLiteBuiltinStablehloConvert,
  kLrtOpCodeShloDynamicSlice = kTfLiteBuiltinStablehloDynamicSlice,
  kLrtOpCodeShloDynamicUpdateSlice = kTfLiteBuiltinStablehloDynamicUpdateSlice,
  kLrtOpCodeShloPad = kTfLiteBuiltinStablehloPad,
  kLrtOpCodeShloIota = kTfLiteBuiltinStablehloIota,
  kLrtOpCodeShloGeneral = kTfLiteBuiltinStablehloDotGeneral,
  kLrtOpCodeShloWindow = kTfLiteBuiltinStablehloReduceWindow,
  kLrtOpCodeShloSort = kTfLiteBuiltinStablehloSort,
  kLrtOpCodeShloWhile = kTfLiteBuiltinStablehloWhile,
  kLrtOpCodeShloGather = kTfLiteBuiltinStablehloGather,
  kLrtOpCodeShloTranspose = kTfLiteBuiltinStablehloTranspose,
  kLrtOpCodeTflDilate = kTfLiteBuiltinDilate,
  kLrtOpCodeShloRngBitGenerator = kTfLiteBuiltinStablehloRngBitGenerator,
  kLrtOpCodeTflReduceWindow = kTfLiteBuiltinReduceWindow,
  kLrtOpCodeShloComposite = kTfLiteBuiltinStablehloComposite,
} LrtOpCode;

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_LITE_RT_OP_CODE_H_
