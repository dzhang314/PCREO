#!/usr/bin/env wolframscript
(* ::Package:: *)

ImportPCREO[path_String] := With[
	{entries = StringSplit[Import[path, "Text"], "\n\n"]},
	{
		N[Map[ToExpression] /@ StringTrim /@ StringSplit[
			StringSplit[entries[[3]], "\n"], ","]],
		Map[FromDigits] /@ StringTrim /@ StringSplit[
			StringSplit[entries[[4]], "\n"], ","]
	}
];

SphericalTriangle[{a_List, b_List, c_List}, 0] :=
	If[Cross[b - a, c - a] . a >= 0,
		Triangle[{a, b, c}, VertexNormals -> {a, b, c}],
		Triangle[{a, c, b}, VertexNormals -> {a, c, b}]
	];

SphericalTriangle[{a_List, b_List, c_List}, n_Integer] := With[{
	ab = Normalize[a + b],
	bc = Normalize[b + c],
	ca = Normalize[c + a]},
	{
		SphericalTriangle[{a, ab, ca}, n - 1],
		SphericalTriangle[{b, bc, ab}, n - 1],
		SphericalTriangle[{c, ca, bc}, n - 1],
		SphericalTriangle[{ab, bc, ca}, n - 1]
	}
];

SphericalLine[{a_List, b_List}, 0] := Line[{a, b}];

SphericalLine[{a_List, b_List}, n_Integer] := With[
	{ab = Normalize[a + b]},
	{
		SphericalLine[{a, ab}, n - 1],
		SphericalLine[{ab, b}, n - 1]
	}
];

Circumcenter[{a_, b_, c_}] := Block[
	{normal, norm2, denom, alpha, beta, gamma},
	normal = Cross[a - b, b - c];
	norm2 = normal . normal;
	denom = norm2 + norm2;
	alpha = ((b - c) . (b - c)) * ((a - b) . (a - c));
	beta = ((a - c) . (a - c)) * ((b - a) . (b - c));
	gamma = ((a - b) . (a - b)) * ((c - a) . (c - b));
	(alpha * a + beta * b + gamma * c) / denom
];

SphericalCircumcenter[pts_List] :=
	Normalize@Mean[Circumcenter /@ Subsets[pts, {3}]];

ValenceColor[3] = Orange;
ValenceColor[4] = Yellow;
ValenceColor[5] = Red;
ValenceColor[6] = Green;
ValenceColor[7] = Blue;
ValenceColor[8] = Purple;
ValenceColor[_] = White;

PlotPCREOPrimitives[points_List, facets_List] := Block[
	{vertexColors, facetCenters, neighboringFacets, dist, k, triangles, dots},
	vertexColors = ValenceColor /@ Lookup[
		Counts[Join @@ facets], Range@Length[points], 0];
	facetCenters = SphericalCircumcenter[points[[#]]]& /@ facets;
	neighboringFacets = Select[Subsets[Range@Length[facets], {2}],
		Apply[Length@Intersection[facets[[#1]], facets[[#2]]] >= 2&]];
	dist = Min[EuclideanDistance @@@ Subsets[points, {2}]];
	k = 0;
	While[dist / 2^k > 1/5, ++k];
	triangles = Join @@ Table[Block[{i, j, a, b},
		{i, j} = pair;
		{a, b} = Intersection[facets[[i]], facets[[j]]];
		{
			{EdgeForm[], vertexColors[[a]], SphericalTriangle[
				{points[[a]], facetCenters[[i]], facetCenters[[j]]}, k]},
			{EdgeForm[], vertexColors[[b]], SphericalTriangle[
				{points[[b]], facetCenters[[i]], facetCenters[[j]]}, k]},
			{Thick, SphericalLine[{facetCenters[[i]], facetCenters[[j]]}, k]}
		}],
		{pair, neighboringFacets}];
	dots = {Black, MapThread[Cylinder[{#1, #2}, Min[dist / 8, 0.05]]&,
		{0.5 * points, points}]};
	Append[triangles, dots]
];

PlotPCREOPrimitives[{points_List, facets_List}] := PlotPCREOPrimitives[points, facets];


$DatabasePath = "/Users/dzhang314/PCREODatabase";
$ImagePath = "/Users/dzhang314/data/PCREOImages";
LaunchKernels[16];


Do[
	If[!FileExistsQ[FileNameJoin[{$ImagePath, FileBaseName[dir] <> ".gif"}]],
		Print["Loading data for ", FileBaseName[dir], "..."];
(*
		With[{selected = First@MinimalBy[
			ImportPCREO /@ FileNames[All, dir],
			Length[#[[4]]]&
		]},
*)
		With[{selected = ImportPCREO@RandomChoice@FileNames[All, dir]},
		Print["Rendering image for ", FileBaseName[dir], "..."];
		With[{primitives = PlotPCREOPrimitives[selected]},
		With[{frames = Table[
			Graphics3D[primitives,
				Boxed -> False,
				ViewVector -> {
					100 * Normalize[{2*Cos[t], 2*Sin[t], 1.2 * Cos[2*t]}],
					{0, 0, 0}
				},
				ViewVertical -> {0, 0, 1},
				ViewAngle -> 0.021],
			{t, N[2*Pi * Range[0, 199] / 200]}
		]},
		Export[
			FileNameJoin[{$ImagePath,
				"PCREO-03-" <> FileBaseName[numDir] <>
				"-" <> FileBaseName[dir] <> ".gif"}],
			ParallelMap[
				Rasterize[#, Background -> None, ImageResolution -> 50]&,
				frames, Method -> "FinestGrained"
			],
			(* Map[
				Rasterize[#, Background -> None, ImageResolution -> 50]&,
				frames
			], *)
			"DisplayDurations" -> 1/30,
			"AnimationRepetitions" -> Infinity
		]]]]
	],
	{numDir, FileNames["0*", $DatabasePath]},
	{dir, FileNames["*-*-*-*-*", numDir]}
];
