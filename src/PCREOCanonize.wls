#!/usr/bin/env wolframscript

ImportPCREO[path_String] := With[{
	data = SplitBy[Import[path], # === {}&]},
	{Flatten[data[[1]]], data[[3]], data[[5]], data[[7]]}];

NNGraph[{_, _, facets_List, _}] := Graph[
	Union @@ (UndirectedEdge @@@ Subsets[Sort[#], {2}]& /@ facets)];

G6String[path_] := ExportString[
	CanonicalGraph@NNGraph@ImportPCREO[path],
	"Graph6"];

$DatabasePath = "D:\\Data\\PCREODatabase";
$GraphPath = "D:\\Data\\PCREOGraphs";

Do[
	Print["Calculating graphs for ", FileBaseName[dir], "..."];
	With[{g6strings = ParallelMap[G6String, FileNames[All, dir],
			Method -> "FinestGrained"]},
		Export[
			FileNameJoin[{$GraphPath, FileBaseName[dir] <> ".g6"}],
			StringJoin @@ Union[g6strings], "Text"
		]
	],
	{dir, FileNames[All, $DatabasePath]}
];
