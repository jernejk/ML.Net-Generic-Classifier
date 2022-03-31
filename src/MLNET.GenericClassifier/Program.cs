// This console application is made to test out 2 different multi-class classifiers:
// 1. Predicting gitmoji based on a git commit message
// 2. Predicting bank transaction category based on transaction data.
// 
// The code shows you how you can make multi-class classifiers generic.
// Check Gitmo.cs and TransactionClassifier.cs for trainers and data inputs and the 2 methods below for training and predicting.
using System.Text;

// Add UTF-8 support to show emojis.
Console.OutputEncoding = Encoding.UTF8;
Console.Write("\xfeff"); // bom = byte order mark

// If true, this will run AutoML (~60 seconds of training) every time even if the ML model already exists.
bool retrain = false;

// First example is generated bank transaction data and is by design pretty good ML data.
// We should see good accuracies.
string transactionTrainingDataPath = "Data/transactions.csv";
string transactionModelPath = "bank.mlnet";
if (!File.Exists(transactionModelPath) || retrain)
{
    Console.WriteLine("Training bank transactions model (~60 seconds)...");
    TrainModel<TransactionData>(new TransactionTrainer(), transactionTrainingDataPath, transactionModelPath);
}

Console.WriteLine();
Console.WriteLine("Predictions for bank transactions model");
var transactionPredictor = new Predictor<TransactionData>();
transactionPredictor.LoadFromFilePath(transactionModelPath);

Predict(transactionPredictor, new TransactionData { TransactionName = "Runolfsson" }); // Games
Predict(transactionPredictor, new TransactionData { TransactionName = "PAYPAL *Gerlach 123456789" }, true); // Music

Console.WriteLine();
Console.WriteLine();

// Second example is for predicting gitmojis but from a dataset that was not cleaned and therefore we won't get as good results as the first example.
string gitmoTrainingDataPath = "Data/CommitMessages.csv";
string gitmoModelPath = "gitmo.mlnet";
if (!File.Exists(gitmoTrainingDataPath) || retrain)
{
    Console.WriteLine("Training gitmo model (~120 seconds)...");
    TrainModel<GitComment>(new GitmoTrainer(), gitmoTrainingDataPath, gitmoModelPath, 120);
}

Console.WriteLine();
Console.WriteLine("Predictions for gitmo model");
var gitmoPredictor = new Predictor<GitComment>();
gitmoPredictor.LoadFromFilePath(gitmoModelPath);

Predict(gitmoPredictor, new GitComment { CommitMessage = "Initial commit" }); // 🎉
Predict(gitmoPredictor, new GitComment { CommitMessage = "Updated button UI" }, true); // 💄


/// <summary>
/// Use generic trainer.
/// </summary>
static void TrainModel<TModelInput>(IBaseTrainer trainer, string dataPath, string modelPath, uint trainingTime = 60)
{
    var trainingData = trainer.MlContext.Data.LoadFromTextFile<TModelInput>(dataPath, separatorChar: ',', hasHeader: false);

    RunDetail<MulticlassClassificationMetrics> result = trainer.AutoTrain(trainingData, trainingTime);
    if (result.ValidationMetrics.MacroAccuracy < 0.7)
    {
        Console.WriteLine("The model accuracy is quite low at " + result.ValidationMetrics.MacroAccuracy);
    }

    trainer.SaveModel(modelPath, result.Model);
}

/// <summary>
/// Use generic predictor.
/// </summary>
static void Predict<TModelInput>(IPredictor<TModelInput> predictor, TModelInput input, bool showDetails = false)
    where TModelInput : class
{
    // Predict most likely label.
    Prediction result = predictor.Predict(input);
    float score = result.Score.Max();
    Console.WriteLine($"Prediction {input} -> {result.Label.Trim()} ({Math.Round(score * 100, 2)}%)");

    if (showDetails)
    {
        // Alternatively, we can get key pair value of value predictions.
        Dictionary<string, float>? allPredictions = predictor.PredictionList(input);
        foreach (var prediction in allPredictions)
        {
            Console.WriteLine($"  {prediction.Key} -> {Math.Round(prediction.Value * 100, 2)}%");
        }
    }
}