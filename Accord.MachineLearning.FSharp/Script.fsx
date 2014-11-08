#r @"C:\Users\Mathias\Documents\github\Accord-FSharp\Accord.MachineLearning.FSharp\packages\Accord.2.13.1\lib\net40\Accord.dll"
#r @"C:\Users\Mathias\Documents\github\Accord-FSharp\Accord.MachineLearning.FSharp\packages\Accord.Math.2.13.1\lib\net40\Accord.Math.dll"
#r @"C:\Users\Mathias\Documents\github\Accord-FSharp\Accord.MachineLearning.FSharp\packages\Accord.Statistics.2.13.1\lib\net40\Accord.Statistics.dll"
#r @"C:\Users\Mathias\Documents\github\Accord-FSharp\Accord.MachineLearning.FSharp\packages\Accord.MachineLearning.2.13.1\lib\net40\Accord.MachineLearning.dll"

open System
open System.IO


open Accord.Statistics.Kernels
open Accord.MachineLearning
open Accord.MachineLearning.VectorMachines
open Accord.MachineLearning.VectorMachines.Learning

let labels, observations = 
    @"C:\Users\Mathias\Documents\github\Accord-FSharp\Accord.MachineLearning.FSharp\trainingsample.csv"
    |> File.ReadAllLines
    |> fun x -> x.[1..500]
    |> Array.map (fun line -> line.Split(','))
    |> Array.map (fun line -> 
        line.[0] |> int, 
        line.[1..] |> Array.map float)
    |> Array.unzip

let algorithm = 
    fun (svm: KernelSupportVectorMachine) 
        (classInputs: float[][]) 
        (classOutputs: int[]) (i: int) (j: int) -> 
        let strategy = SequentialMinimalOptimization(svm, classInputs, classOutputs)
        strategy :> ISupportVectorMachineLearning

let features = 28 * 28
let classes = 10

let kernel = Linear()

// clarify api?
let svm = new MulticlassSupportVectorMachine(features, kernel, classes)
let learner = MulticlassSupportVectorLearning(svm, observations, labels)
let config = SupportVectorMachineLearningConfigurationFunction(algorithm)
learner.Algorithm <- config
 
let error = learner.Run()