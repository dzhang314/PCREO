#!/usr/bin/env wolframscript

ImportPCREO[path_String] := With[{
	data = SplitBy[Import[path], # === {}&]},
	{Flatten[data[[1]]], data[[3]], data[[5]], data[[7]]}
];

NNGraph[{_, _, _, facets_List}] := With[
	{edges = Select[Intersection @@@ Subsets[facets, {2}],
		GreaterThan[1]@*Length]},
	If[!AllTrue[edges, EqualTo[2]@*Length], Throw[edges]];
	Graph[UndirectedEdge @@@ edges]
];

G6String[path_] := ExportString[
	CanonicalGraph@NNGraph@ImportPCREO[path],
	"Graph6"
];

$PCREODatabaseDirectory = "D:\\Data\\PCREODatabase";
$PCREOGraphDirectory = "D:\\Data\\PCREOGraphs";

LaunchKernels[16];
Do[
	With[{outputFile = FileNameJoin[{$PCREOGraphDirectory,
		"PCREO-03-" <> FileBaseName[numDir] <>
		"-" <> FileBaseName[dir] <> ".g6"}
	]},
	If[!FileExistsQ[outputFile],
		Print["Calculating graphs for ", FileBaseName[dir], "..."];
		With[{g6strings = ParallelMap[
			G6String, FileNames[All, dir],
			Method -> "FinestGrained"
		]},
		Export[outputFile, StringJoin @@ Union[g6strings], "Text"]
	]]],
	{numDir, FileNames[All, $PCREODatabaseDirectory]},
	{dir, FileNames[All, numDir]}
];
