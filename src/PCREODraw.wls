#!/usr/bin/env wolframscript

ImportPCREO[path_String] := With[
	{data = SplitBy[Import[path], # === {}&]},
	{Flatten[data[[1]]], data[[3]], data[[5]], data[[7]]}
];

ValenceColor[4] = Yellow;
ValenceColor[5] = Red;
ValenceColor[6] = Green;
ValenceColor[7] = Blue;
ValenceColor[_] = White;

PlotPCREOPrimitives[{_, points_List, facets_List, _}] := Block[
	{vertexColors, facetCenters, neighboringFacets, triangles, dist, dots},
	vertexColors = ValenceColor /@ Lookup[
		Counts[Join @@ facets], Range@Length[points], 0];
	facetCenters = Normalize@Mean[points[[#]]]& /@ facets;
	neighboringFacets = Select[Subsets[Range@Length[facets], {2}],
		Apply[Length@Intersection[facets[[#1]], facets[[#2]]] >= 2&]];
	triangles = Join @@ Table[Block[{i, j, a, b},
		{i, j} = pair;
		{a, b} = Intersection[facets[[i]], facets[[j]]];
		{
			{EdgeForm[], vertexColors[[a]], Triangle[
				{points[[a]], facetCenters[[i]], facetCenters[[j]]}]},
			{EdgeForm[], vertexColors[[b]], Triangle[
				{points[[b]], facetCenters[[i]], facetCenters[[j]]}]},
			{Thick, Line[{facetCenters[[i]], facetCenters[[j]]}]}
		}],
		{pair, neighboringFacets}];
	dist = Min[EuclideanDistance @@@ Subsets[points, {2}]];
	dots = {Black, MapThread[Cylinder[{#1, #2}, Min[dist / 8, 0.05]]&,
		{0.5 * points, points}]};
	Append[triangles, dots]
];

$DatabasePath = "D:\\Data\\PCREODatabase";
$ImagePath = "D:\\Data\\PCREOImages";
LaunchKernels[16];

Do[
	If[!FileExistsQ[FileNameJoin[{$ImagePath, FileBaseName[dir] <> ".gif"}]],
		Print["Loading data for ", FileBaseName[dir], "..."];
		With[{selected = First@MinimalBy[
			ImportPCREO /@ FileNames[All, dir],
			Length[#[[3]]]&
		]},
		Print["Rendering image for ", FileBaseName[dir], "..."];
		With[{primitives = PlotPCREOPrimitives[selected]},
		With[{frames = Table[
			Graphics3D[primitives,
				Boxed -> False,
				ViewVector -> {
					100 * Normalize[{2*Cos[t], 2*Sin[t], Cos[2*t]}],
					{0, 0, 0}
				},
				ViewVertical -> {0, 0, 1},
				ViewAngle -> 0.021],
			{t, N[2*Pi * Range[0, 199] / 200]}
		]},
		Export[
			FileNameJoin[{$ImagePath, FileBaseName[dir] <> ".gif"}],
			ParallelMap[
				Rasterize[#, Background -> None, ImageResolution -> 50]&,
				frames, Method -> "FinestGrained"
			],
			"DisplayDurations" -> 1/30,
			"AnimationRepetitions" -> Infinity
		]]]]
	],
	{dir, FileNames[All, $DatabasePath]}
];
