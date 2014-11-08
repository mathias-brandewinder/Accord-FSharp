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
    |> fun x -> x.[1..]
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

let kernel = Gaussian() // Linear() -> why is this failing now???
// let linKernel = Linear ()

// clarify api?
let svm = new MulticlassSupportVectorMachine(features, kernel, classes)
let learner = MulticlassSupportVectorLearning(svm, observations, labels)
let config = SupportVectorMachineLearningConfigurationFunction(algorithm)
learner.Algorithm <- config
 
let error = learner.Run()

// Validation

let validation = 
    @"C:\Users\Mathias\Documents\github\Accord-FSharp\Accord.MachineLearning.FSharp\validationsample.csv"
    |> File.ReadAllLines
    |> fun x -> x.[1..]
    |> Array.map (fun line -> line.Split(','))
    |> Array.map (fun line -> 
        line.[0] |> int, 
        line.[1..] |> Array.map float)

validation 
|> Array.averageBy (fun (label,obs) -> 
    if label = svm.Compute(obs) then 1. else 0.)

let learn (dataset:int[]*float[][]) (kernel:IKernel) =
    let labels,observations = dataset
    let features = observations.[0].Length
    let classes = 
        labels 
        |> Seq.distinct 
        |> Seq.length
    
    let algorithm = 
        fun (svm: KernelSupportVectorMachine) 
            (classInputs: float[][]) 
            (classOutputs: int[]) (i: int) (j: int) -> 
            let strategy = SequentialMinimalOptimization(svm, classInputs, classOutputs)
            strategy.Complexity <- 5.0
            strategy :> ISupportVectorMachineLearning
    
    let svm = new MulticlassSupportVectorMachine(features, kernel, classes)
    let learner = MulticlassSupportVectorLearning(svm, observations, labels)
    let config = SupportVectorMachineLearningConfigurationFunction(algorithm)
    learner.Algorithm <- config    
    
    learner.Run () |> ignore
    svm

type SMOConfig = { Epsilon:float; Complexity:float; }

let smo (labels:int[]) (obs:float[][]) (svm:KernelSupportVectorMachine) =
    SequentialMinimalOptimization(svm,obs,labels)

let withComplexity (smo:SequentialMinimalOptimization) complexity = 
    smo.Complexity <- complexity
    smo

type SMOConfig = SequentialMinimalOptimization -> SequentialMinimalOptimization

let setComplexity x (smo:SequentialMinimalOptimization) =
    smo.Complexity <- x
    smo

let setEpsilon x (smo:SequentialMinimalOptimization) =
    smo.Epsilon <- x
    smo

let configure (configs:SMOConfig seq) (smo:SequentialMinimalOptimization) =
    configs |> Seq.fold (fun currentSmo config -> config currentSmo) smo

let test = configure [ setComplexity 4.0; setEpsilon 0.1; ]

type LearningStrategy = KernelSupportVectorMachine -> float[][] -> int[] -> int -> int -> ISupportVectorMachineLearning

let SMO = 
    fun (svm: KernelSupportVectorMachine) 
        (classInputs: float[][]) 
        (classOutputs: int[]) (i: int) (j: int) -> 
        let strategy = SequentialMinimalOptimization(svm, classInputs, classOutputs)
        strategy :> ISupportVectorMachineLearning

let configureWith (strategy:LearningStrategy) (kernel:IKernel) =
    (kernel,strategy)

let learn2 (dataset:int[]*float[][]) ((kernel,strategy):IKernel*LearningStrategy) =

    let labels,observations = dataset
    let features = observations.[0].Length
    let classes = 
        labels 
        |> Seq.distinct 
        |> Seq.length

    let svm = new MulticlassSupportVectorMachine(features, kernel, classes)
    let learner = MulticlassSupportVectorLearning(svm, observations, labels)
    let config = SupportVectorMachineLearningConfigurationFunction(strategy)
    learner.Algorithm <- config    
    
    learner.Run () |> ignore
    svm

Gaussian () 
|> configureWith SMO 
|> learn2 (labels,observations)

type Result =
    | Success of float
    | Failure of string

let f (x:float) (y:float) =
    if y = 0. then "Divide by Zero" |> Failure
    else x / y |> Success

let z = 
    match (f 1. 0.) with
    | Success(x) -> printfn "%f" x
    | Failure(msg) -> printfn "%s" msg

let f x y = x + y
let f = 42
let f () = 42